#!/usr/bin/env python3
"""
三角网格 AI 原生处理 — Phase 5

图神经网络直接在三角网格上运行，不经 RGB 转换。
纯 numpy 实现，CPU 可训练。

核心:
  TriGCN — 三角图卷积网络 (2层 + ReLU)
  train_denoiser — 自监督去噪训练
  inference — 模型推理 + 可视化对比
"""

import math
import time
import numpy as np
from PIL import Image

from triangle_engine import (
    assigned_channel, neighbors_of,
    sample_single_channels, borrow_neighbors,
    correct_triangular_isp, render_triangles,
)


# ============================================================
#  邻接矩阵预计算
# ============================================================

def build_adjacency(n_rows, n_cols):
    """
    构建三角网格的邻接表。

    每个节点 (r,c) 连接到其 3 个邻居。
    返回: adj[r*n_cols + c] = [node_id1, node_id2, node_id3]
    """
    N = n_rows * n_cols
    adj = np.zeros((N, 3), dtype=np.int32)

    for r in range(n_rows):
        for c in range(n_cols):
            node_id = r * n_cols + c
            nbr_ids = []
            for nr, nc in neighbors_of(r, c):
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    nbr_ids.append(nr * n_cols + nc)
                else:
                    nbr_ids.append(node_id)  # self-loop for boundary
            adj[node_id] = nbr_ids

    return adj


# ============================================================
#  TriGCN — 三角图卷积网络
# ============================================================

class TriGCN:
    """2-layer Graph Convolutional Network for triangular mesh"""

    def __init__(self, in_dim=4, hidden_dim=16, out_dim=1, seed=42):
        rng = np.random.RandomState(seed)

        # He initialization
        self.W1 = rng.randn(in_dim, hidden_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.randn(hidden_dim, out_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(out_dim)

        # Adam state
        self._init_adam()

    def _init_adam(self):
        self.m = {}  # first moment
        self.v = {}  # second moment
        for name, param in [
            ("W1", self.W1), ("b1", self.b1),
            ("W2", self.W2), ("b2", self.b2),
            ("W3", self.W3), ("b3", self.b3),
        ]:
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

    # ---- 前向传播 ----

    def _graph_conv(self, X, adj, W, b, activation=True):
        """
        图卷积：H' = σ(mean(X_i + neighbors) @ W + b)

        X: [N, in_dim]
        adj: [N, 3]
        W: [in_dim, out_dim]
        b: [out_dim]
        """
        N = X.shape[0]
        # 邻居特征平均
        neigh_X = (X[adj[:, 0]] + X[adj[:, 1]] + X[adj[:, 2]]) / 3.0

        # 自身 + 邻居 的平均
        agg = (X + neigh_X) / 2.0

        out = agg @ W + b
        if activation:
            out = np.maximum(0, out)  # ReLU
        return out, agg  # 返回聚合结果用于反向传播

    def forward(self, X, adj):
        """
        前向传播。

        X: [N, 4] 输入特征 [value, is_R, is_G, is_B]
        adj: [N, 3] 邻接表

        Returns:
            y_pred: [N, 1] 预测的干净单通道值
        """
        h1, _ = self._graph_conv(X, adj, self.W1, self.b1, activation=True)
        h2, _ = self._graph_conv(h1, adj, self.W2, self.b2, activation=True)
        h3, _ = self._graph_conv(h2, adj, self.W3, self.b3, activation=False)
        return h3

    # ---- 训练 ----

    def train_step(self, X, adj, y_true, lr=0.001):
        """
        单步训练（手动反向传播）。

        X: [N, 4]  输入 (带噪声)
        y_true: [N, 1]  目标 (干净)
        """
        N = X.shape[0]

        # --- Forward ---
        h1, agg1 = self._graph_conv(X, adj, self.W1, self.b1, activation=True)
        h2, agg2 = self._graph_conv(h1, adj, self.W2, self.b2, activation=True)
        h3, agg3 = self._graph_conv(h2, adj, self.W3, self.b3, activation=False)

        y_pred = h3

        # --- Loss: MSE ---
        error = y_pred - y_true
        loss = np.mean(error * error)

        # --- Backward (layer 3: linear) ---
        d_h3 = 2.0 * error / N
        d_W3 = agg3.T @ d_h3
        d_b3 = d_h3.sum(axis=0)
        d_agg2 = d_h3 @ self.W3.T
        d_h2 = d_agg2 * (h2 > 0)
        d_W2 = agg2.T @ d_h2
        d_b2 = d_h2.sum(axis=0)
        d_agg1 = d_h2 @ self.W2.T
        d_h1 = d_agg1 * (h1 > 0)
        d_W1 = agg1.T @ d_h1
        d_b1 = d_h1.sum(axis=0)

        # --- Adam update ---
        updates = {
            "W1": d_W1, "b1": d_b1,
            "W2": d_W2, "b2": d_b2,
            "W3": d_W3, "b3": d_b3,
        }

        for name, grad in updates.items():
            param = getattr(self, name)
            self.m[name] = 0.9 * self.m[name] + 0.1 * grad
            self.v[name] = 0.999 * self.v[name] + 0.001 * grad * grad
            param -= lr * self.m[name] / (np.sqrt(self.v[name]) + 1e-8)
            setattr(self, name, param)

        return float(loss)


# ============================================================
#  特征构建
# ============================================================

def build_features(single, n_rows, n_cols):
    """
    从单通道数据构建节点特征。

    X[i, 0] = 归一化单通道值
    X[i, 1:4] = one-hot 通道类型 (R/G/B)
    """
    N = n_rows * n_cols
    X = np.zeros((N, 4), dtype=np.float32)

    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            ch = assigned_channel(r, c)
            X[idx, 0] = single[r, c] / 255.0  # 归一化
            X[idx, 1 + ch] = 1.0  # one-hot channel

    return X


# ============================================================
#  训练管线
# ============================================================

def train_denoiser(image_paths, triangle_side=16,
                   epochs=200, lr=0.001, noise_std=0.15,
                   progress_callback=None):
    """
    自监督去噪训练。

    对每张图：原图→单通道(干净)→加噪声(输入)→训练模型预测干净值。

    Args:
        image_paths: 训练图片路径列表
        triangle_side: 三角边长
        epochs: 训练轮数
        noise_std: 噪声标准差 (相对于 [0,1] 归一化值)
        progress_callback: fn(epoch, loss)

    Returns:
        model: 训练好的 TriGCN
        adj_ref: 参考邻接表
    """
    model = TriGCN(in_dim=4, hidden_dim=16, out_dim=1)

    # 用第一张图确定网格尺寸
    img = Image.open(image_paths[0]).convert("RGB")
    W, H = img.size
    S = float(triangle_side)
    h = S * math.sqrt(3) / 2.0
    n_cols = int(W / (S / 2.0)) + 3
    n_rows = int(H / h) + 2
    adj = build_adjacency(n_rows, n_cols)
    adj_ref = (n_rows, n_cols, S, h)

    if progress_callback:
        progress_callback(0, 0)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            # 调整到统一尺寸
            img = img.resize((W, H), Image.LANCZOS)
            pixels = np.array(img).astype(np.float32)

            # 干净单通道
            clean_single = sample_single_channels(
                pixels, S, h, n_rows, n_cols, sample_radius=1.0
            )

            # 加噪声
            noisy_single = clean_single.copy()
            noise = np.random.randn(*noisy_single.shape) * noise_std * 255
            noisy_single = np.clip(noisy_single + noise, 0, 255)

            # 构建特征
            X = build_features(noisy_single, n_rows, n_cols)

            # 目标：干净值（只预测单通道值，通道类型已知）
            y_true = clean_single.reshape(-1, 1) / 255.0

            # 训练一步
            loss = model.train_step(X, adj, y_true, lr=lr)
            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if progress_callback:
            progress_callback(epoch + 1, avg_loss)

    return model, adj_ref


# ============================================================
#  推理
# ============================================================

def denoise_image(model, pil_image, adj_ref):
    """
    用训练好的模型对单张图去噪。

    Returns:
        denoised RGB PIL Image
    """
    n_rows, n_cols, S, h = adj_ref
    W, H = pil_image.size
    pixels = np.array(pil_image.convert("RGB")).astype(np.float32)

    # 采样单通道
    single = sample_single_channels(pixels, S, h, n_rows, n_cols, sample_radius=1.0)

    # 构建特征
    X = build_features(single, n_rows, n_cols)

    # 构建邻接表
    adj = build_adjacency(n_rows, n_cols)

    # 推理
    y_pred = model.forward(X, adj)
    denoised_single = np.clip(y_pred.reshape(n_rows, n_cols) * 255.0, 0, 255)

    # borrow + ISP
    borrowed = borrow_neighbors(denoised_single, n_rows, n_cols, edge_mode="mirror")
    corrected = correct_triangular_isp(
        denoised_single, borrowed, n_rows, n_cols,
        iterations=3, edge_sensitivity=40,
    )

    return render_triangles(corrected, S, h, n_rows, n_cols, W, H)


# ============================================================
#  CLI 测试
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python triangle_ai.py train <img1> [img2...]")
        print("      python triangle_ai.py test <img>")
        sys.exit(1)

    if sys.argv[1] == "train":
        paths = sys.argv[2:]
        if not paths:
            paths = ["test_edge.png"]

        print(f"Training on {len(paths)} images, epochs=100...")

        losses = []
        def cb(epoch, loss):
            losses.append(loss)
            if epoch % 20 == 0:
                print(f"  epoch {epoch:3d}: loss={loss:.6f}")

        t0 = time.time()
        model, adj_ref = train_denoiser(paths, epochs=100, noise_std=0.15,
                                        progress_callback=cb)
        print(f"Training done in {time.time()-t0:.1f}s, final loss={losses[-1]:.6f}")

        # 保存权重
        np.savez("tri_gcn_weights.npz",
                 W1=model.W1, b1=model.b1,
                 W2=model.W2, b2=model.b2,
                 W3=model.W3, b3=model.b3)
        np.savez("tri_gcn_adj.npz",
                 n_rows=adj_ref[0], n_cols=adj_ref[1],
                 S=adj_ref[2], h=adj_ref[3])
        print("Weights saved: tri_gcn_weights.npz")

    elif sys.argv[1] == "test":
        # 加载权重
        w = np.load("tri_gcn_weights.npz")
        a = np.load("tri_gcn_adj.npz")
        adj_ref = (int(a["n_rows"]), int(a["n_cols"]), float(a["S"]), float(a["h"]))

        model = TriGCN()
        model.W1 = w["W1"]; model.b1 = w["b1"]
        model.W2 = w["W2"]; model.b2 = w["b2"]
        model.W3 = w["W3"]; model.b3 = w["b3"]

        img = Image.open(sys.argv[2]).convert("RGB")
        result = denoise_image(model, img, adj_ref)
        result.save("ai_denoised.png")
        print(f"Denoised: ai_denoised.png ({result.size})")
