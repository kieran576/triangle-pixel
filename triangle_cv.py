#!/usr/bin/env python3
"""
三角网格原生 CV 基元 — Phase 2

三个核心操作，全部在三角网格上原生执行，
不需要转 RGB 或矩形网格：

1. luminance — 单通道→亮度估计（一次 borrow 取均值）
2. edges — 3邻域梯度 + 非极大值抑制
3. multiscale_edges — Sierpinski 金字塔跨尺度融合
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from triangle_engine import (
    assigned_channel, is_upward, tri_center, tri_vertices,
    neighbors_of, sample_single_channels, borrow_neighbors,
    render_triangles,
)


# ============================================================
#  1. 亮度估计
# ============================================================

def estimate_luminance(single, n_tri_rows, n_tri_cols):
    """
    从单通道三角网格估计每个三角的亮度。

    方法：用 borrow 获取完整 RGB，然后取亮度 L = (R+G+B)/3。
    borrow 只需一遍遍历，不需要 ISP 迭代。
    """
    borrowed = borrow_neighbors(single, n_tri_rows, n_tri_cols, edge_mode="mirror")
    return borrowed.mean(axis=2)  # (R+G+B)/3


# ============================================================
#  2. 三角网格边缘检测
# ============================================================

def detect_edges(single, n_tri_rows, n_tri_cols,
                 low_threshold=20, high_threshold=50):
    """
    三角网格原生边缘检测 — 替代 Canny。

    算法：
    1. 估计每个三角的亮度
    2. 对每个三角，计算沿 3 条边方向的梯度
    3. 梯度幅值 = max(3个方向梯度)
    4. 梯度方向 = 最大梯度对应的边法线方向
    5. 非极大值抑制：沿梯度方向，只保留局部极大值
    6. 双阈值 + 连通性追踪

    返回:
        edge_map: [n_tri_rows, n_tri_cols] bool — True = 边缘三角
        gradient: [n_tri_rows, n_tri_cols] float — 梯度幅值
        direction: [n_tri_rows, n_tri_cols] int — 梯度方向 (0/1/2 = 三条边)
    """
    # Step 1: 亮度
    lum = estimate_luminance(single, n_tri_rows, n_tri_cols)

    # Step 2: 计算每个三角沿3条边的梯度
    grad = np.zeros((n_tri_rows, n_tri_cols), dtype=np.float32)
    grad_dir = np.zeros((n_tri_rows, n_tri_cols), dtype=np.int32)

    for r in range(n_tri_rows):
        for c in range(n_tri_cols):
            I_c = lum[r, c]
            max_g = 0.0
            max_d = 0

            for idx, (nr, nc) in enumerate(neighbors_of(r, c)):
                if 0 <= nr < n_tri_rows and 0 <= nc < n_tri_cols:
                    g = abs(I_c - lum[nr, nc])
                    if g > max_g:
                        max_g = g
                        max_d = idx

            grad[r, c] = max_g
            grad_dir[r, c] = max_d

    # Step 3: 非极大值抑制
    # 沿梯度方向检查：如果当前三角的梯度小于它的邻居（在梯度方向上的邻居），则抑制
    suppressed = np.zeros((n_tri_rows, n_tri_cols), dtype=bool)

    for r in range(n_tri_rows):
        for c in range(n_tri_cols):
            g = grad[r, c]
            d = grad_dir[r, c]

            # 梯度方向对应的邻居
            nbrs = neighbors_of(r, c)
            nr, nc = nbrs[d]

            # 检查正方向
            if 0 <= nr < n_tri_rows and 0 <= nc < n_tri_cols:
                if grad[nr, nc] > g:
                    suppressed[r, c] = True
                    continue

            # 检查反方向（通过邻居的反方向邻居）
            # 反方向 = 当前三角相对于邻居的方向
            rev_nbrs = neighbors_of(nr, nc) if (0 <= nr < n_tri_rows and 0 <= nc < n_tri_cols) else []
            for rev_idx, (rr, rc) in enumerate(rev_nbrs):
                if rr == r and rc == c:
                    # 邻居的反方向邻居的梯度更大 → 抑制当前
                    if 0 <= nr < n_tri_rows and 0 <= nc < n_tri_cols:
                        rev_nbrs2 = neighbors_of(nr, nc)
                        # 检查邻居的另两个方向
                        for other_d, (onr, onc) in enumerate(rev_nbrs2):
                            if other_d != rev_idx:
                                if 0 <= onr < n_tri_rows and 0 <= onc < n_tri_cols:
                                    if grad[onr, onc] > g:
                                        suppressed[r, c] = True
                    break

    # Step 4: 双阈值
    strong = grad >= high_threshold
    weak = (grad >= low_threshold) & ~strong
    weak[ suppressed] = False
    strong[suppressed] = False

    # Step 5: 连通性追踪 — 弱边缘如果与强边缘相邻则保留
    edge_map = strong.copy()
    changed = True
    while changed:
        changed = False
        for r in range(n_tri_rows):
            for c in range(n_tri_cols):
                if not weak[r, c]:
                    continue
                for nr, nc in neighbors_of(r, c):
                    if 0 <= nr < n_tri_rows and 0 <= nc < n_tri_cols:
                        if edge_map[nr, nc]:
                            edge_map[r, c] = True
                            weak[r, c] = False
                            changed = True
                            break

    return edge_map, grad, grad_dir


# ============================================================
#  3. Sierpinski 多尺度边缘
# ============================================================

def detect_edges_multiscale(single, n_tri_rows, n_tri_cols,
                            S, h, num_scales=3,
                            low_threshold=15, high_threshold=40):
    """
    多尺度边缘检测 — 利用 Sierpinski 金字塔。

    粗尺度 → 检测大结构边缘（抗噪声）
    细尺度 → 检测精细边缘
    融合：细尺度边缘如果与粗尺度边缘空间一致则保留

    这替代了传统的高斯金字塔 + 梯度金字塔流程。
    """
    all_edges = []
    all_grads = []
    all_scales = []

    current_single = single
    current_rows = n_tri_rows
    current_cols = n_tri_cols
    current_S = S
    current_h = h

    for level in range(num_scales):
        # 在当前尺度检测边缘
        edges, grad, _ = detect_edges(
            current_single, current_rows, current_cols,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        all_edges.append(edges)
        all_grads.append(grad)
        all_scales.append((current_rows, current_cols, current_S, current_h))

        if level == num_scales - 1:
            break

        # 降采样到更粗的尺度 (2×2 → 1)
        new_rows = max(2, current_rows // 2)
        new_cols = max(2, current_cols // 2)
        new_S = current_S * 2.0
        new_h = current_h * 2.0

        coarse = np.zeros((new_rows, new_cols), dtype=np.float32)
        for r in range(new_rows):
            for c in range(new_cols):
                block = current_single[
                    r * 2 : min((r + 1) * 2, current_rows),
                    c * 2 : min((c + 1) * 2, current_cols),
                ]
                coarse[r, c] = np.mean(block) if block.size > 0 else 0

        current_single = coarse
        current_rows = new_rows
        current_cols = new_cols
        current_S = new_S
        current_h = new_h

    # 融合：从最粗到最细传播
    fused = all_edges[-1]  # 最粗尺度

    for level in range(len(all_edges) - 2, -1, -1):
        fine_edges = all_edges[level]
        fine_rows, fine_cols = fine_edges.shape
        coarse_rows, coarse_cols = fused.shape

        # 上采样粗尺度边缘到细尺度
        upsampled = np.zeros((fine_rows, fine_cols), dtype=bool)
        for fr in range(fine_rows):
            for fc in range(fine_cols):
                cr = min(fr // 2, coarse_rows - 1)
                cc = min(fc // 2, coarse_cols - 1)
                upsampled[fr, fc] = fused[cr, cc]

        # 融合：细尺度边缘如果在粗尺度边缘的邻域内则保留，否则抑制
        dilated = upsampled.copy()
        for r in range(fine_rows):
            for c in range(fine_cols):
                if upsampled[r, c]:
                    for nr, nc in neighbors_of(r, c):
                        if 0 <= nr < fine_rows and 0 <= nc < fine_cols:
                            dilated[nr, nc] = True

        fused = fine_edges & dilated

    return fused, all_edges, all_grads


# ============================================================
#  可视化
# ============================================================

def render_edges_overlay(base_image, S, h, n_tri_rows, n_tri_cols,
                         edge_map, edge_color=(255, 0, 0), line_width=2):
    """
    在原图上叠加检测到的边缘三角。
    边缘三角用红色边框高亮显示。
    """
    result = base_image.copy().convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for r in range(n_tri_rows):
        for c in range(n_tri_cols):
            if edge_map[r, c]:
                verts = tri_vertices(r, c, S, h)
                pts = [(v[0], v[1]) for v in verts]
                color = edge_color + (180,)
                draw.polygon(pts, outline=color, fill=None)

    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def render_gradient_heatmap(grad, S, h, n_tri_rows, n_tri_cols, W, H):
    """
    将梯度幅值渲染为热力图（三角形填充）。
    红色=强梯度，蓝色=弱梯度。
    """
    if grad.max() <= 0:
        grad_norm = grad
    else:
        grad_norm = grad / grad.max()

    rgb = np.zeros((n_tri_rows, n_tri_cols, 3), dtype=np.uint8)
    for r in range(n_tri_rows):
        for c in range(n_tri_cols):
            v = grad_norm[r, c]
            # 蓝→绿→红 热力图
            if v < 0.5:
                rgb[r, c] = [0, int(v * 2 * 255), int((1 - v * 2) * 255)]
            else:
                rgb[r, c] = [int((v - 0.5) * 2 * 255), int((1 - v) * 2 * 255), 0]

    return render_triangles(rgb, S, h, n_tri_rows, n_tri_cols, W, H)


# ============================================================
#  CLI 测试
# ============================================================

if __name__ == "__main__":
    import sys, time

    if len(sys.argv) < 2:
        print("用法: python triangle_cv.py <image>")
        sys.exit(1)

    img = Image.open(sys.argv[1]).convert("RGB")
    W, H = img.size
    pixels = np.array(img).astype(np.float32)

    S = 16.0
    h = S * math.sqrt(3) / 2.0
    n_cols = int(W / (S / 2.0)) + 3
    n_rows = int(H / h) + 2

    print(f"三角网格: {n_rows}×{n_cols}")

    # 采样
    single = sample_single_channels(pixels, S, h, n_rows, n_cols, sample_radius=1.0)

    # 亮度
    lum = estimate_luminance(single, n_rows, n_cols)
    lum_rgb = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            v = np.clip(lum[r, c], 0, 255)
            lum_rgb[r, c] = [int(v), int(v), int(v)]
    render_triangles(lum_rgb, S, h, n_rows, n_cols, W, H).save("cv_luminance.png")

    # 边缘检测（单尺度）
    t0 = time.time()
    edge_map, grad, _ = detect_edges(single, n_rows, n_cols,
                                     low_threshold=15, high_threshold=35)
    print(f"边缘检测: {edge_map.sum()} 个边缘三角, {time.time()-t0:.2f}s")

    render_edges_overlay(img, S, h, n_rows, n_cols, edge_map).save("cv_edges.png")
    render_gradient_heatmap(grad, S, h, n_rows, n_cols, W, H).save("cv_gradient.png")

    # 多尺度
    t0 = time.time()
    fused, all_e, all_g = detect_edges_multiscale(
        single, n_rows, n_cols, S, h, num_scales=3)
    print(f"多尺度边缘: {fused.sum()} 个边缘三角, {time.time()-t0:.2f}s")
    render_edges_overlay(img, S, h, n_rows, n_cols, fused).save("cv_edges_ms.png")

    print("输出: cv_luminance.png, cv_edges.png, cv_gradient.png, cv_edges_ms.png")
