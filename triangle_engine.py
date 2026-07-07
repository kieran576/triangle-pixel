#!/usr/bin/env python3
"""
三角形像素引擎 v2 — 改进版

Layer 1: RAW — 单通道输出 (马赛克 + 六边形降采样)
Layer 2: BORROW — 邻居借用 (边缘镜像填充)
Layer 3: CORRECT — 空间偏移校正 (双边滤波保边缘 + 自适应迭代)
Layer 4: SIERPINSKI — 见 triangle_sierpinski.py

改进:
  - 区域采样：平均三角中心周围像素，减少混叠
  - 双边比值平滑：保边缘的同时做空间校正
  - 边缘镜像填充：边界三角形不再黑
  - RAW 降采样：真实六边形→矩形像素
"""

import math
import numpy as np
from PIL import Image, ImageDraw

# ---- numba 加速回退 ----
_has_numba = False
try:
    from triangle_engine_fast import (
        _sample_fast, _borrow_fast, _isp_fast,
        sample_fast, borrow_fast, isp_fast,
    )
    _has_numba = True
except ImportError:
    pass


# ============================================================
#  全局几何与排列规则
# ============================================================

EVEN_ROW = [0, 1, 2, 2, 1, 0]  # R G B B G R
ODD_ROW = [2, 1, 0, 0, 1, 2]  # B G R R G B


def assigned_channel(r, c):
    return EVEN_ROW[c % 6] if r % 2 == 0 else ODD_ROW[c % 6]


def is_upward(r, c):
    return (r + c) % 2 == 0


def tri_center(r, c, S, h):
    x = c * S / 2.0
    y = r * h + (2.0 * h / 3.0 if is_upward(r, c) else h / 3.0)
    return x, y


def tri_vertices(r, c, S, h):
    x, _ = tri_center(r, c, S, h)
    if is_upward(r, c):
        return [
            (x, r * h),
            (x - S / 2.0, (r + 1) * h),
            (x + S / 2.0, (r + 1) * h),
        ]
    else:
        return [
            (x - S / 2.0, r * h),
            (x + S / 2.0, r * h),
            (x, (r + 1) * h),
        ]


def neighbors_of(r, c):
    if is_upward(r, c):
        return [(r, c - 1), (r, c + 1), (r + 1, c)]
    else:
        return [(r, c - 1), (r, c + 1), (r - 1, c)]


# ============================================================
#  改进 1：区域采样（加权邻域平均）
# ============================================================

def _sample_patch(pixels, cx, cy, S, radius=1.5):
    """
    从中心 (cx,cy) 周围采样，返回平均 RGB。
    radius: 采样半径（像素），1.5 = ~7 像素的圆形邻域
    """
    H, W = pixels.shape[:2]
    r_int = int(np.ceil(radius))
    x0 = max(0, int(cx) - r_int)
    x1 = min(W, int(cx) + r_int + 1)
    y0 = max(0, int(cy) - r_int)
    y1 = min(H, int(cy) + r_int + 1)

    if x1 <= x0 or y1 <= y0:
        ix = int(np.clip(cx, 0, W - 1))
        iy = int(np.clip(cy, 0, H - 1))
        return pixels[iy, ix].astype(np.float32)

    patch = pixels[y0:y1, x0:x1].astype(np.float32)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    # 高斯权重，sigma = radius/2
    sigma = radius / 2.0
    weights = np.exp(-dist2 / (2 * sigma * sigma))
    weights = weights[:, :, np.newaxis]

    avg = np.sum(patch * weights, axis=(0, 1)) / max(np.sum(weights), 1e-6)
    return avg


def sample_single_channels(pixels, S, h, n_tri_rows, n_tri_cols,
                           sample_radius=1.5):
    """
    从原图采样每个三角形的单通道值。
    sample_radius: 0 = 仅中心像素, >0 = 邻域加权平均
    """
    H, W = pixels.shape[:2]
    single = np.zeros((n_tri_rows, n_tri_cols), dtype=np.float32)

    for r in range(n_tri_rows):
        for c in range(n_tri_cols):
            cx, cy = tri_center(r, c, S, h)
            ch = assigned_channel(r, c)

            if sample_radius <= 0:
                ix = int(np.clip(cx, 0, W - 1))
                iy = int(np.clip(cy, 0, H - 1))
                single[r, c] = pixels[iy, ix, ch]
            else:
                # 自适应半径：大三角用大半径，小三角用小半径
                radius = min(sample_radius, S / 4.0)
                avg_rgb = _sample_patch(pixels, cx, cy, S, radius)
                single[r, c] = avg_rgb[ch]

    return single


# ============================================================
#  改进 2：边缘镜像填充
# ============================================================

def _mirror_coord(val, max_val):
    """镜像边界坐标"""
    if val < 0:
        return -val - 1
    elif val >= max_val:
        return 2 * max_val - val - 1
    return val


def borrow_neighbors(single, n_tri_rows, n_tri_cols, edge_mode="mirror"):
    """
    每个三角形从三个邻居各借一个通道。

    edge_mode: "zero" (默认, 缺失=0) | "mirror" (镜像填充)
    """
    rgb = np.zeros((n_tri_rows, n_tri_cols, 3), dtype=np.float32)

    for r in range(n_tri_rows):
        for c in range(n_tri_cols):
            nbrs = neighbors_of(r, c)
            color = np.zeros(3, dtype=np.float32)

            for nr, nc in nbrs:
                if 0 <= nr < n_tri_rows and 0 <= nc < n_tri_cols:
                    nbr_ch = assigned_channel(nr, nc)
                    color[nbr_ch] = single[nr, nc]
                elif edge_mode == "mirror":
                    mnr = _mirror_coord(nr, n_tri_rows)
                    mnc = _mirror_coord(nc, n_tri_cols)
                    nbr_ch = assigned_channel(mnr, mnc)
                    color[nbr_ch] = single[mnr, mnc]
                # else: keep 0

            rgb[r, c] = color

    return rgb


# ============================================================
#  Layer 1 改进：RAW 六边形降采样
# ============================================================

def build_raw_mosaic(single, S, h, n_tri_rows, n_tri_cols, W, H):
    """RAW 马赛克：每个三角形填充其单通道纯色"""
    out = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(out)

    ch_colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}

    for r in range(n_tri_rows):
        for c in range(n_tri_cols):
            verts = tri_vertices(r, c, S, h)
            pts = [(v[0], v[1]) for v in verts]
            ch = assigned_channel(r, c)
            intensity = int(np.clip(single[r, c], 0, 255))
            base = ch_colors[ch]
            color = (
                int(base[0] * intensity / 255),
                int(base[1] * intensity / 255),
                int(base[2] * intensity / 255),
            )
            draw.polygon(pts, fill=color, outline=None)

    return out


def build_raw_downscale(single, n_tri_rows, n_tri_cols, scale=2):
    """
    六边形降采样：每 2×3 个三角 (6个=2R2G2B) → 1 个 RGB 像素。
    scale: 行方向分组数 (1=每2行1个输出行, 2=每4行...)

    输出: PIL Image, 尺寸约为 (n_cols//3 * scale, n_rows//2 * scale)
    """
    # 分组：每 2 行 × 3 列 = 6 三角
    out_rows = max(1, (n_tri_rows - 1) // 2)
    out_cols = max(1, (n_tri_cols - 1) // 3)
    out = np.zeros((out_rows, out_cols, 3), dtype=np.float32)
    counts = np.zeros((out_rows, out_cols, 3), dtype=np.int32)

    for r in range(n_tri_rows):
        for c in range(n_tri_cols):
            or_ = r // 2
            oc = c // 3
            if or_ >= out_rows or oc >= out_cols:
                continue
            ch = assigned_channel(r, c)
            out[or_, oc, ch] += single[r, c]
            counts[or_, oc, ch] += 1

    # 平均
    for ch in range(3):
        mask = counts[:, :, ch] > 0
        out[:, :, ch][mask] /= counts[:, :, ch][mask]

    out = np.clip(out, 0, 255).astype(np.uint8)

    # 上采样回原比例（用 nearest-neighbor 使六边形像素可见）
    if scale > 1:
        out = np.repeat(np.repeat(out, scale, axis=0), scale, axis=1)

    return Image.fromarray(out, "RGB")


# ============================================================
#  改进 3a：ISP 风格去伪色校正（色差空间 + 边缘检测 + 中值滤波）
# ============================================================

def correct_triangular_isp(single, borrowed_rgb, n_tri_rows, n_tri_cols,
                           iterations=3, edge_sensitivity=25):
    """
    三角网格原生去伪色。

    核心：每个三角的测量通道是地面真值，绝不漂移。
    缺失通道通过色差中值滤波 + 直接测量值加权平均来估计。

    与 borrow 的关键区别：
    - borrow: 缺失通道 = 唯一来源邻居的测量值（跨边缘就有伪色）
    - ISP: 缺失通道 = 所有邻居的加权色差反推 + 中值滤波去离群
    """
    corrected = borrowed_rgb.copy()
    eps = 1e-6

    for it in range(iterations):
        lum = corrected.mean(axis=2)

        diff_rg = corrected[:, :, 0] - corrected[:, :, 1]
        diff_rb = corrected[:, :, 0] - corrected[:, :, 2]
        diff_gb = corrected[:, :, 1] - corrected[:, :, 2]

        # 3-邻域中值滤波去 zipper
        def median3(arr):
            result = arr.copy()
            for r in range(n_tri_rows):
                for c in range(n_tri_cols):
                    vals = [arr[r, c]]
                    for nr, nc in neighbors_of(r, c):
                        if 0 <= nr < n_tri_rows and 0 <= nc < n_tri_cols:
                            vals.append(arr[nr, nc])
                    result[r, c] = np.median(vals)
            return result

        diff_rg = median3(diff_rg)
        diff_rb = median3(diff_rb)
        diff_gb = median3(diff_gb)

        new_rgb = np.zeros_like(corrected)

        for r in range(n_tri_rows):
            for c in range(n_tri_cols):
                own_ch = assigned_channel(r, c)
                known_val = single[r, c]
                I_c = lum[r, c]
                nbrs = neighbors_of(r, c)

                # 分离：直接来源邻居(提供测量真值) vs 其他邻居(提供色差参考)
                direct_src = {}  # ch -> (value, weight)
                diff_weights = []
                diff_rg_vals = []
                diff_rb_vals = []
                diff_gb_vals = []

                for nr, nc in nbrs:
                    if not (0 <= nr < n_tri_rows and 0 <= nc < n_tri_cols):
                        continue
                    nbr_ch = assigned_channel(nr, nc)
                    nbr_measured = single[nr, nc]  # 邻居的测量真值
                    I_n = lum[nr, nc]
                    w = np.exp(-(I_c - I_n) ** 2 / (2 * edge_sensitivity ** 2))
                    w = max(w, 1e-8)  # 防止权重全零

                    # 该邻居提供其拥有通道的测量真值
                    direct_src[nbr_ch] = (nbr_measured, w)

                    # 该邻居也提供色差参考
                    diff_rg_vals.append(diff_rg[nr, nc])
                    diff_rb_vals.append(diff_rb[nr, nc])
                    diff_gb_vals.append(diff_gb[nr, nc])
                    diff_weights.append(w)

                if not diff_weights:
                    new_rgb[r, c] = corrected[r, c]
                    continue

                diff_weights = np.array(diff_weights)
                w_sum = max(diff_weights.sum(), eps)

                est_rg = np.average(diff_rg_vals, weights=diff_weights)
                est_rb = np.average(diff_rb_vals, weights=diff_weights)
                est_gb = np.average(diff_gb_vals, weights=diff_weights)

                # 重建：已知通道固定，缺失通道 = 混合(直接测量, 色差反推)
                # 直接测量权重 = edge trust towards that neighbor
                # 色差反推权重 = 1 - edge trust (跨边缘时更依赖色差)

                if own_ch == 0:  # R 已知
                    Rv = known_val
                    # G: 混合直接来源和色差
                    g_direct, g_w = direct_src.get(1, (0, 0))
                    g_diff = Rv - est_rg
                    Gv = g_w * g_direct + (1 - g_w) * g_diff if g_w > 0 else g_diff
                    # B
                    b_direct, b_w = direct_src.get(2, (0, 0))
                    b_diff = Rv - est_rb
                    Bv = b_w * b_direct + (1 - b_w) * b_diff if b_w > 0 else b_diff
                elif own_ch == 1:  # G 已知
                    Gv = known_val
                    r_direct, r_w = direct_src.get(0, (0, 0))
                    r_diff = Gv + est_rg
                    Rv = r_w * r_direct + (1 - r_w) * r_diff if r_w > 0 else r_diff
                    b_direct, b_w = direct_src.get(2, (0, 0))
                    b_diff = Gv - est_gb
                    Bv = b_w * b_direct + (1 - b_w) * b_diff if b_w > 0 else b_diff
                else:  # B 已知
                    Bv = known_val
                    r_direct, r_w = direct_src.get(0, (0, 0))
                    r_diff = Bv + est_rb
                    Rv = r_w * r_direct + (1 - r_w) * r_diff if r_w > 0 else r_diff
                    g_direct, g_w = direct_src.get(1, (0, 0))
                    g_diff = Bv + est_gb
                    Gv = g_w * g_direct + (1 - g_w) * g_diff if g_w > 0 else g_diff

                new_rgb[r, c] = [
                    np.clip(Rv, 0, 255),
                    np.clip(Gv, 0, 255),
                    np.clip(Bv, 0, 255),
                ]

        corrected = new_rgb

    return corrected


# ============================================================
#  改进 3：双边滤波比值平滑（保留作为备选）
# ============================================================

def correct_spatial_offset(single, borrowed_rgb, n_tri_rows, n_tri_cols,
                           iterations=3, smooth_radius=1,
                           bilateral_sigma=0.15):
    """
    基于颜色比值局部恒定性，校正空间偏移。

    改进：使用双边滤波而非简单均值，在平滑比值的同时保边缘。

    bilateral_sigma: 双边滤波的强度参数 (在 log-ratio 空间)
                     越小越保边缘，越大越平滑
    """
    corrected = borrowed_rgb.copy()
    eps = 1e-6

    for it in range(iterations):
        # ---- 计算比值图 (在 log 空间更稳定) ----
        # log(R/G), log(R/B), log(G/B)
        log_rg = np.zeros((n_tri_rows, n_tri_cols), dtype=np.float32)
        log_rb = np.zeros((n_tri_rows, n_tri_cols), dtype=np.float32)
        log_gb = np.zeros((n_tri_rows, n_tri_cols), dtype=np.float32)

        for r in range(n_tri_rows):
            for c in range(n_tri_cols):
                R = max(corrected[r, c, 0], eps)
                G = max(corrected[r, c, 1], eps)
                B = max(corrected[r, c, 2], eps)
                log_rg[r, c] = np.log(R / G)
                log_rb[r, c] = np.log(R / B)
                log_gb[r, c] = np.log(G / B)

        # ---- 双边滤波平滑 ----
        def bilateral_smooth(log_map):
            smoothed = log_map.copy()
            for r in range(n_tri_rows):
                for c in range(n_tri_cols):
                    pts = [(r, c)] + neighbors_of(r, c)
                    center_val = log_map[r, c]
                    weights = []
                    vals = []
                    for nr, nc in pts:
                        if 0 <= nr < n_tri_rows and 0 <= nc < n_tri_cols:
                            diff = abs(log_map[nr, nc] - center_val)
                            w = np.exp(-diff * diff / (2 * bilateral_sigma * bilateral_sigma))
                            vals.append(log_map[nr, nc] * w)
                            weights.append(w)
                    if weights:
                        smoothed[r, c] = sum(vals) / max(sum(weights), eps)
            return smoothed

        for _ in range(smooth_radius):
            log_rg = bilateral_smooth(log_rg)
            log_rb = bilateral_smooth(log_rb)
            log_gb = bilateral_smooth(log_gb)

        # 转回线性空间
        ratio_rg = np.exp(log_rg)
        ratio_rb = np.exp(log_rb)
        ratio_gb = np.exp(log_gb)

        # ---- 校正 ----
        new_rgb = np.zeros_like(corrected)

        for r in range(n_tri_rows):
            for c in range(n_tri_cols):
                own_ch = assigned_channel(r, c)
                measured = single[r, c]

                if own_ch == 0:  # R
                    R_val = measured
                    G_val = R_val / max(ratio_rg[r, c], eps)
                    B_val = R_val / max(ratio_rb[r, c], eps)
                elif own_ch == 1:  # G
                    G_val = measured
                    R_val = G_val * ratio_rg[r, c]
                    B_val = G_val * ratio_gb[r, c]
                else:  # B
                    B_val = measured
                    R_val = B_val * ratio_rb[r, c]
                    G_val = B_val * ratio_gb[r, c]

                new_rgb[r, c] = [
                    np.clip(R_val, 0, 255),
                    np.clip(G_val, 0, 255),
                    np.clip(B_val, 0, 255),
                ]

        corrected = new_rgb

    return corrected


# ============================================================
#  渲染
# ============================================================

def render_triangles(rgb, S, h, n_tri_rows, n_tri_cols, W, H):
    """将 RGB 三角形数组渲染为 PIL Image"""
    out = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(out)

    for r in range(n_tri_rows):
        for c in range(n_tri_cols):
            verts = tri_vertices(r, c, S, h)
            pts = [(v[0], v[1]) for v in verts]
            color = tuple(int(np.clip(v, 0, 255)) for v in rgb[r, c])
            draw.polygon(pts, fill=color, outline=None)

    return out


# ============================================================
#  完整管线入口
# ============================================================

def process_pipeline(pil_image, triangle_side, mode="borrow",
                     correct_iterations=3, correct_sigma=0.15,
                     sample_radius=1.5, edge_mode="mirror",
                     raw_downscale_factor=2,
                     progress_callback=None):
    """
    三角形像素处理管线。

    Args:
        pil_image: PIL Image (RGB)
        triangle_side: 三角形边长 (px)
        mode: "raw" | "raw_downscale" | "borrow" | "correct"
        correct_iterations: 校正迭代次数
        correct_sigma: 双边滤波强度 (log-ratio 空间)
        sample_radius: 采样半径 (0=中心, >0=邻域)
        edge_mode: "mirror" | "zero"
        raw_downscale_factor: RAW 降采样输出倍率

    Returns:
        PIL Image
    """
    img = pil_image.convert("RGB")
    W, H = img.size
    pixels = np.array(img).astype(np.float32)

    S = float(triangle_side)
    h = S * math.sqrt(3) / 2.0
    n_tri_cols = int(W / (S / 2.0)) + 3
    n_tri_rows = int(H / h) + 2

    def prog(pct):
        if progress_callback:
            progress_callback(pct)

    # Step A: 采样
    prog(5)
    single = sample_single_channels(pixels, S, h, n_tri_rows, n_tri_cols,
                                    sample_radius=sample_radius)

    if mode == "raw":
        prog(50)
        result = build_raw_mosaic(single, S, h, n_tri_rows, n_tri_cols, W, H)
        prog(100)
        return result

    if mode == "raw_downscale":
        prog(50)
        result = build_raw_downscale(single, n_tri_rows, n_tri_cols,
                                     scale=raw_downscale_factor)
        prog(100)
        return result

    # Step B: 邻居借用
    prog(30)
    borrowed = borrow_neighbors(single, n_tri_rows, n_tri_cols, edge_mode=edge_mode)

    if mode == "borrow":
        prog(60)
        result = render_triangles(borrowed, S, h, n_tri_rows, n_tri_cols, W, H)
        prog(100)
        return result

    # Step C: 校正
    if mode == "correct":
        prog(50)
        corrected = correct_spatial_offset(
            single, borrowed, n_tri_rows, n_tri_cols,
            iterations=correct_iterations,
            bilateral_sigma=correct_sigma,
        )
        prog(80)
        result = render_triangles(corrected, S, h, n_tri_rows, n_tri_cols, W, H)
        prog(100)
        return result

    if mode == "correct_isp":
        prog(50)
        corrected = correct_triangular_isp(
            single, borrowed, n_tri_rows, n_tri_cols,
            iterations=correct_iterations,
            edge_sensitivity=correct_sigma * 200,  # remap sigma→sensitivity
        )
        prog(80)
        result = render_triangles(corrected, S, h, n_tri_rows, n_tri_cols, W, H)
        prog(100)
        return result

    raise ValueError(f"未知模式: {mode}")
