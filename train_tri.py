#!/usr/bin/env python3
"""
train_tri.py — 三角 GCN 跨图训练脚本

从 tri_convert.py 产出的 .tri 数据集加载数据,
用 PyTorch GCN 做端到端训练 (RAW → RGB),
支持跨图批量训练 (解决逐图训练的局限)。

用法:
    # 训练
    python train_tri.py --data ./tri_output/ --epochs 200 --batch 64

    # 推理
    python train_tri.py --data ./tri_output/ --infer ./test_img.jpg

    # 导出 ONNX (准备部署到手机)
    python train_tri.py --data ./tri_output/ --export onnx
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tri_dataset import TriDataset
from triangle_engine import (
    assigned_channel, borrow_neighbors, correct_triangular_isp,
    render_triangles, sample_single_channels,
)


# ============================================================
#  PyTorch GCN 模型
# ============================================================

class TriGCN(nn.Module):
    """
    三角图卷积网络 — PyTorch 版。

    每层: H' = ReLU( (H + mean(neighbor_H)) / 2 @ W + b )

    输入: [N, 4] — [value/255, is_R, is_G, is_B]
    输出: [N, 3] — [R, G, B] in [0,1]
    """

    def __init__(self, in_dim=4, hidden=32, out_dim=3):
        super().__init__()
        self.conv1 = nn.Linear(in_dim, hidden)
        self.conv2 = nn.Linear(hidden, hidden)
        self.conv3 = nn.Linear(hidden, out_dim)

    def _graph_aggregate(self, X, adj):
        """
        X: [N, D]
        adj: [N, 3] — neighbor indices
        Returns: [N, D] — (X + mean_of_3_neighbors) / 2
        """
        neigh = (X[adj[:, 0]] + X[adj[:, 1]] + X[adj[:, 2]]) / 3.0
        return (X + neigh) / 2.0

    def forward(self, X, adj):
        agg1 = self._graph_aggregate(X, adj)
        h1 = F.relu(self.conv1(agg1))
        agg2 = self._graph_aggregate(h1, adj)
        h2 = F.relu(self.conv2(agg2))
        agg3 = self._graph_aggregate(h2, adj)
        out = torch.sigmoid(self.conv3(agg3))
        return out


# ============================================================
#  训练
# ============================================================

def build_adjacency_torch(n_rows, n_cols):
    """构建三角网格邻接矩阵 [N, 3]."""
    from triangle_engine import neighbors_of
    N = n_rows * n_cols
    adj = np.zeros((N, 3), dtype=np.int64)
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            nbrs = neighbors_of(r, c)
            for k, (nr, nc2) in enumerate(nbrs):
                if 0 <= nr < n_rows and 0 <= nc2 < n_cols:
                    adj[idx, k] = nr * n_cols + nc2
                else:
                    adj[idx, k] = idx  # 边界自环
    return torch.from_numpy(adj)


def prepare_batch(raw_batch, gt_batch, adj, n_rows, n_cols, device):
    """
    将 batch 的三角 RAW 转换为 GCN 输入特征。

    raw_batch: [B, 1, H, W] — 三角 RAW (归一化)
    gt_batch:  [B, 3, H, W] — GT RGB (归一化)
    返回: X [B*N, 4], y [B*N, 3]
    """
    B = raw_batch.size(0)
    N = n_rows * n_cols

    X_list = []
    y_list = []

    for b in range(B):
        raw = raw_batch[b, 0].numpy()  # [H, W]
        gt = gt_batch[b].permute(1, 2, 0).numpy()  # [H, W, 3]

        feat = np.zeros((N, 4), dtype=np.float32)
        target = np.zeros((N, 3), dtype=np.float32)

        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                ch = assigned_channel(r, c)
                feat[idx, 0] = raw[r, c]
                feat[idx, ch + 1] = 1.0  # one-hot channel
                target[idx] = gt[r, c]

        X_list.append(torch.from_numpy(feat))
        y_list.append(torch.from_numpy(target))

    X = torch.cat(X_list, dim=0).to(device)
    y = torch.cat(y_list, dim=0).to(device)
    return X, y


def train_epoch(model, loader, adj, n_rows, n_cols, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for raw_batch, gt_batch in loader:
        raw_batch = raw_batch.to(device)
        gt_batch = gt_batch.to(device)

        X, y = prepare_batch(raw_batch, gt_batch, adj, n_rows, n_cols, device)

        optimizer.zero_grad()
        pred = model(X, adj.to(device))
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, adj, n_rows, n_cols, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for raw_batch, gt_batch in loader:
        raw_batch = raw_batch.to(device)
        gt_batch = gt_batch.to(device)

        X, y = prepare_batch(raw_batch, gt_batch, adj, n_rows, n_cols, device)
        pred = model(X, adj.to(device))
        loss = F.mse_loss(pred, y)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================
#  推理
# ============================================================

def infer_image(model, adj, n_rows, n_cols, S, h, image_path, device):
    """对单张图片做端到端三角推理."""
    from PIL import Image
    from triangle_sensor import TriangleSensor

    scene = Image.open(image_path).convert("RGB")
    W, H = scene.size

    # 用传感器模拟器捕获三角 RAW (无噪声)
    sensor = TriangleSensor(triangle_side=int(S), iso=0)
    tri_raw, meta = sensor.capture(scene)
    nr, nc = meta["n_rows"], meta["n_cols"]

    # 如果网格尺寸不匹配, 重新构建 adj
    if nr != n_rows or nc != n_cols:
        adj = build_adjacency_torch(nr, nc)
        n_rows, n_cols = nr, nc

    N = n_rows * n_cols
    feat = np.zeros((N, 4), dtype=np.float32)
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            ch = assigned_channel(r, c)
            feat[idx, 0] = tri_raw[r, c] / 255.0
            feat[idx, ch + 1] = 1.0

    X = torch.from_numpy(feat).to(device)
    adj_t = adj.to(device)

    model.eval()
    with torch.no_grad():
        pred = model(X, adj_t).cpu().numpy()

    # 重塑为图像
    rgb = (np.clip(pred, 0, 1) * 255).astype(np.uint8).reshape(n_rows, n_cols, 3)
    result = render_triangles(rgb.astype(np.float32), S, h, n_rows, n_cols, W, H)
    return result, tri_raw


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="三角 GCN 跨图训练")
    parser.add_argument("--data", required=True, help=".tri 数据集目录")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch", type=int, default=64, help="batch size (图片数)")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--hidden", type=int, default=32, help="隐藏层维度")
    parser.add_argument("--save", default="./tri_model.pt", help="模型保存路径")
    parser.add_argument("--infer", help="推理: 图片路径")
    parser.add_argument("--device", default="cpu", help="cpu | cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据集
    ds = TriDataset(args.data, mode="image")
    params = ds.get_grid_params()
    n_rows, n_cols, S, h = params["n_rows"], params["n_cols"], params["S"], params["h"]
    print(f"网格: {n_rows}x{n_cols}, S={S:.1f}")

    # 构建邻接矩阵
    adj = build_adjacency_torch(n_rows, n_cols)
    print(f"节点数: {n_rows * n_cols}")

    # 推理模式
    if args.infer:
        model = TriGCN(hidden=args.hidden).to(device)
        if os.path.exists(args.save):
            model.load_state_dict(torch.load(args.save, map_location=device))
            print(f"加载模型: {args.save}")
        else:
            print("[警告] 未找到预训练模型, 使用随机权重")

        result, _ = infer_image(model, adj, n_rows, n_cols, S, h,
                                args.infer, device)
        out_path = args.infer.replace(".jpg", "_tri_ai.jpg").replace(".png", "_tri_ai.png")
        result.save(out_path)
        print(f"推理结果: {out_path}")
        return

    # 训练模式
    # 简单划分: 前 80% 训练, 后 20% 验证
    n = len(ds)
    n_train = max(1, int(n * 0.8))
    n_val = max(1, n - n_train)
    train_ds = torch.utils.data.Subset(ds, range(n_train))
    val_ds = torch.utils.data.Subset(ds, range(n_train, n))

    batch = min(args.batch, n_train)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=0)

    print(f"训练: {len(train_ds)} 样本, 验证: {len(val_ds)} 样本")

    model = TriGCN(hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print(f"参数: {sum(p.numel() for p in model.parameters())}")
    print()

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, adj, n_rows, n_cols,
                                 optimizer, device)
        val_loss = evaluate(model, val_loader, adj, n_rows, n_cols, device)
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), args.save)

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:4d}: train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  best={best_loss:.6f}  "
                  f"[{elapsed:.1f}s]")

    print(f"\n完成: {args.epochs} 轮, best_loss={best_loss:.6f}")
    print(f"模型: {args.save}")


if __name__ == "__main__":
    main()
