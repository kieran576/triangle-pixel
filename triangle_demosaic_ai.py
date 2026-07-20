#!/usr/bin/env python3
"""
端到端 AI 去马赛克 — 三角 GCN 替代整个 ISP 管线

三角 RAW → GCN → RGB (一次前向传播)
无需 borrow + ISP 迭代

训练数据由三角传感器模拟器生成。
"""

import math, time
import numpy as np
from PIL import Image

from triangle_engine import (
    assigned_channel, neighbors_of,
    sample_single_channels, render_triangles,
)


# ============================================================
#  TriDemosaicGCN — 端到端去马赛克
# ============================================================

class TriDemosaicGCN:
    """
    端到端三角去马赛克图卷积网络。

    输入: [value, is_R, is_G, is_B]  (4D)
    输出: [R, G, B]  (3D)

    架构: 4→hidden→hidden→3
    图卷积自动聚合邻居信息 → 隐式"借用"
    """

    def __init__(self, hidden=32, seed=42):
        rng = np.random.RandomState(seed)

        self.W1 = rng.randn(4, hidden) * np.sqrt(2.0 / 4)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, hidden) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = rng.randn(hidden, 3) * np.sqrt(2.0 / hidden)
        self.b3 = np.zeros(3)

        # Adam state
        self.m = {}; self.v = {}
        for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            param = getattr(self, name)
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

    # ---- 前向传播 ----

    def forward(self, X, adj):
        """
        X: [N, 4] — [value/255, is_R, is_G, is_B]
        adj: [N, 3] — neighbor indices
        Returns: [N, 3] — predicted [R, G, B] in [0,1]
        """
        # Layer 1
        neigh = (X[adj[:, 0]] + X[adj[:, 1]] + X[adj[:, 2]]) / 3.0
        agg1 = (X + neigh) / 2.0
        h1 = np.maximum(0, agg1 @ self.W1 + self.b1)

        # Layer 2
        neigh2 = (h1[adj[:, 0]] + h1[adj[:, 1]] + h1[adj[:, 2]]) / 3.0
        agg2 = (h1 + neigh2) / 2.0
        h2 = np.maximum(0, agg2 @ self.W2 + self.b2)

        # Layer 3 (output, sigmoid)
        neigh3 = (h2[adj[:, 0]] + h2[adj[:, 1]] + h2[adj[:, 2]]) / 3.0
        agg3 = (h2 + neigh3) / 2.0
        h3 = agg3 @ self.W3 + self.b3

        # Sigmoid squash to [0,1]
        return 1.0 / (1.0 + np.exp(-h3))

    # ---- 训练 ----

    def train_step(self, X, adj, y_true, lr=0.001):
        """
        X: [N, 4] features
        y_true: [N, 3] ground truth RGB [0,1]
        """
        N = X.shape[0]

        # Forward
        neigh1 = (X[adj[:, 0]] + X[adj[:, 1]] + X[adj[:, 2]]) / 3.0
        agg1 = (X + neigh1) / 2.0
        h1_raw = agg1 @ self.W1 + self.b1
        h1 = np.maximum(0, h1_raw)

        neigh2 = (h1[adj[:, 0]] + h1[adj[:, 1]] + h1[adj[:, 2]]) / 3.0
        agg2 = (h1 + neigh2) / 2.0
        h2_raw = agg2 @ self.W2 + self.b2
        h2 = np.maximum(0, h2_raw)

        neigh3 = (h2[adj[:, 0]] + h2[adj[:, 1]] + h2[adj[:, 2]]) / 3.0
        agg3 = (h2 + neigh3) / 2.0
        h3_raw = agg3 @ self.W3 + self.b3
        y_pred = 1.0 / (1.0 + np.exp(-h3_raw))

        # MSE loss
        error = y_pred - y_true
        loss = np.mean(error * error)

        # Backward: layer 3
        d_h3 = 2.0 * error / N * y_pred * (1 - y_pred)  # sigmoid grad
        d_W3 = agg3.T @ d_h3
        d_b3 = d_h3.sum(axis=0)

        # Backward: layer 2
        d_agg3 = d_h3 @ self.W3.T
        d_h2 = d_agg3 * (h2 > 0)
        d_W2 = agg2.T @ d_h2
        d_b2 = d_h2.sum(axis=0)

        # Backward: layer 1
        d_agg2 = d_h2 @ self.W2.T
        d_h1 = d_agg2 * (h1 > 0)
        d_W1 = agg1.T @ d_h1
        d_b1 = d_h1.sum(axis=0)

        # Adam update
        updates = {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2,
                   "W3": d_W3, "b3": d_b3}
        for name, grad in updates.items():
            param = getattr(self, name)
            self.m[name] = 0.9 * self.m[name] + 0.1 * grad
            self.v[name] = 0.999 * self.v[name] + 0.001 * grad * grad
            param -= lr * self.m[name] / (np.sqrt(self.v[name]) + 1e-8)
            setattr(self, name, param)

        return float(loss)


# ============================================================
#  数据生成
# ============================================================

def generate_training_data(scene_paths, triangle_side=16,
                           noise_iso=200, num_samples=2000):
    """
    用传感器模拟器生成训练数据。

    对每张图:
    1. 模拟三角传感器捕获 → RAW (带噪声)
    2. 从原图直接采样全 RGB → ground truth
    3. 随机选取三角形作为训练样本

    Returns:
        X_train: [total_samples, 4] features
        y_train: [total_samples, 3] ground truth RGB
        adj_ref: for inference
    """
    from triangle_sensor import optical_blur, add_sensor_noise

    all_X = []
    all_y = []
    adj_ref = None

    samples_per_image = max(1, num_samples // len(scene_paths))

    for path in scene_paths:
        scene = Image.open(path).convert("RGB")
        W, H = scene.size

        # 传感器参数
        S = float(triangle_side)
        h = S * math.sqrt(3) / 2.0
        n_cols = int(W / (S / 2.0)) + 3
        n_rows = int(H / h) + 2

        if adj_ref is None:
            adj_ref = (n_rows, n_cols, S, h)

        # 光学模糊
        blurred = optical_blur(scene, 0.5)
        pixels = np.array(blurred).astype(np.float32)

        # 三角 RAW (带噪声)
        raw = sample_single_channels(pixels, S, h, n_rows, n_cols, sample_radius=1.0)
        raw = add_sensor_noise(raw, base_iso=noise_iso)

        # Ground truth: 每个三角中心的全 RGB
        gt_rgb = np.zeros((n_rows, n_cols, 3), dtype=np.float32)
        for r in range(n_rows):
            for c in range(n_cols):
                x = c * S / 2.0
                y = r * h + (2.0 * h / 3.0 if (r + c) % 2 == 0 else h / 3.0)
                ix = int(np.clip(x, 0, W - 1))
                iy = int(np.clip(y, 0, H - 1))
                gt_rgb[r, c] = pixels[iy, ix] / 255.0

        # 特征
        for s in range(samples_per_image):
            r = np.random.randint(0, n_rows)
            c = np.random.randint(0, n_cols)

            ch = assigned_channel(r, c)
            feat = np.zeros(4, dtype=np.float32)
            feat[0] = raw[r, c] / 255.0
            feat[1 + ch] = 1.0

            all_X.append(feat)
            all_y.append(gt_rgb[r, c])

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)

    # Shuffle
    perm = np.random.permutation(len(X))
    return X[perm], y[perm], adj_ref


# ============================================================
#  训练
# ============================================================

def train_demosaic_gcn(scene_paths, triangle_side=16,
                       epochs=200, batch_size=256, lr=0.001,
                       progress_callback=None):
    """训练端到端去马赛克 GCN"""

    model = TriDemosaicGCN(hidden=32)

    # 生成数据
    X, y, adj_ref = generate_training_data(
        scene_paths, triangle_side, num_samples=2000
    )

    N = len(X)
    losses = []

    for epoch in range(epochs):
        # GCN 需要完整邻接表, 不做 mini-batch
        # (随机采样会让每个 batch 重新构建邻接, 反而比全图慢)
        pass

    # 在全图上训练
    n_rows, n_cols, S, h = adj_ref
    adj = _build_adjacency(n_rows, n_cols)

    # 构建全图特征
    X_full = np.zeros((n_rows * n_cols, 4), dtype=np.float32)
    y_full = np.zeros((n_rows * n_cols, 3), dtype=np.float32)

    # 取第一张图
    scene = Image.open(scene_paths[0]).convert("RGB")
    pixels = np.array(scene).astype(np.float32)
    from triangle_sensor import optical_blur, add_sensor_noise
    blurred = optical_blur(scene, 0.5)
    pixels_b = np.array(blurred).astype(np.float32)
    raw = sample_single_channels(pixels_b, S, h, n_rows, n_cols, sample_radius=1.0)
    raw = add_sensor_noise(raw, base_iso=200)

    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            ch = assigned_channel(r, c)
            X_full[idx, 0] = raw[r, c] / 255.0
            X_full[idx, 1 + ch] = 1.0
            x = c * S / 2.0
            y = r * h + (2.0 * h / 3.0 if (r + c) % 2 == 0 else h / 3.0)
            ix = int(np.clip(x, 0, pixels_b.shape[1] - 1))
            iy = int(np.clip(y, 0, pixels_b.shape[0] - 1))
            y_full[idx] = pixels_b[iy, ix] / 255.0

    for epoch in range(epochs):
        loss = model.train_step(X_full, adj, y_full, lr=lr)
        losses.append(loss)

        if progress_callback:
            progress_callback(epoch + 1, loss)

    return model, adj_ref, losses


def _build_adjacency(n_rows, n_cols):
    """构建三角网格邻接表"""
    N = n_rows * n_cols
    adj = np.zeros((N, 3), dtype=np.int32)
    for r in range(n_rows):
        for c in range(n_cols):
            nid = r * n_cols + c
            nbrs = neighbors_of(r, c)
            ids = []
            for nr, nc in nbrs:
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    ids.append(nr * n_cols + nc)
                else:
                    ids.append(nid)
            adj[nid] = ids
    return adj


# ============================================================
#  推理
# ============================================================

def demosaic_image(model, pil_image, adj_ref):
    """端到端 AI 去马赛克：RAW → RGB"""
    n_rows, n_cols, S, h = adj_ref
    W, H = pil_image.size

    # 采样 RAW
    pixels = np.array(pil_image.convert("RGB")).astype(np.float32)
    raw = sample_single_channels(pixels, S, h, n_rows, n_cols, sample_radius=1.0)

    # 特征
    X = np.zeros((n_rows * n_cols, 4), dtype=np.float32)
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            ch = assigned_channel(r, c)
            X[idx, 0] = raw[r, c] / 255.0
            X[idx, 1 + ch] = 1.0

    # 推理
    adj = _build_adjacency(n_rows, n_cols)
    y_pred = model.forward(X, adj)  # [N, 3] in [0,1]

    rgb = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            rgb[r, c] = np.clip(y_pred[idx] * 255, 0, 255).astype(np.uint8)

    return render_triangles(rgb, S, h, n_rows, n_cols, W, H)


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python triangle_demosaic_ai.py train")
        print("      python triangle_demosaic_ai.py test <image>")
        sys.exit(1)

    if sys.argv[1] == "train":
        paths = ["test_edge.png", "test_input.png"]

        losses = []
        def cb(epoch, loss):
            losses.append(loss)
            if epoch % 20 == 0:
                print(f"  epoch {epoch:3d}: loss={loss:.6f}")

        print("Training end-to-end demosaic GCN...")
        t0 = time.time()
        model, adj_ref, losses = train_demosaic_gcn(
            paths, triangle_side=16, epochs=150,
            progress_callback=cb,
        )
        print(f"Done in {time.time()-t0:.1f}s, final loss={losses[-1]:.6f}")

        # Save
        np.savez("tri_demosaic_weights.npz",
                 W1=model.W1, b1=model.b1,
                 W2=model.W2, b2=model.b2,
                 W3=model.W3, b3=model.b3)
        np.savez("tri_demosaic_adj.npz",
                 n_rows=adj_ref[0], n_cols=adj_ref[1],
                 S=adj_ref[2], h=adj_ref[3])
        print("Weights saved")

    elif sys.argv[1] == "test":
        w = np.load("tri_demosaic_weights.npz")
        a = np.load("tri_demosaic_adj.npz")
        adj_ref = (int(a["n_rows"]), int(a["n_cols"]),
                   float(a["S"]), float(a["h"]))

        model = TriDemosaicGCN(hidden=32)
        model.W1=w["W1"]; model.b1=w["b1"]
        model.W2=w["W2"]; model.b2=w["b2"]
        model.W3=w["W3"]; model.b3=w["b3"]

        img = Image.open(sys.argv[2]).convert("RGB")
        result = demosaic_image(model, img, adj_ref)
        result.save("ai_demosaic.png")
        print(f"Saved: ai_demosaic.png ({result.size})")
