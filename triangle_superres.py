#!/usr/bin/env python3
"""
谢尔宾斯基超分引擎 — Layer 4

利用三角网格的多尺度自相似性做超分辨率：
1. 从原图构建多尺度单通道金字塔
2. 上采样到目标分辨率（双线性 + 可选自相似细化）
3. 在目标分辨率执行 L2 (借用) + L3 (ISP校正) → 高精度 RGB
4. 可递归：zoom × zoom × ... → 理论上无限放大
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from triangle_engine import (
    assigned_channel, is_upward, tri_center, tri_vertices,
    neighbors_of, sample_single_channels, borrow_neighbors,
    correct_triangular_isp, render_triangles,
)


# ============================================================
#  金字塔构建
# ============================================================

class TriPyramid:
    """三角网格多尺度金字塔"""

    def __init__(self, pixels, base_side):
        """
        pixels: 原图 numpy array [H, W, 3]
        base_side: 最高分辨率层的三角形边长
        """
        self.pixels = pixels
        self.H, self.W = pixels.shape[:2]
        self.levels = []  # [(side, h, n_rows, n_cols, single_data)]

        S = float(base_side)
        h = S * math.sqrt(3) / 2.0
        n_cols = int(self.W / (S / 2.0)) + 3
        n_rows = int(self.H / h) + 2
        single = sample_single_channels(pixels, S, h, n_rows, n_cols, sample_radius=1.0)
        self.levels.append((S, h, n_rows, n_cols, single))

    def add_coarser_level(self):
        """在金字塔顶部添加一层更粗的（降采样 2×）"""
        prev_S, prev_h, prev_rows, prev_cols, prev_single = self.levels[0]

        S = prev_S * 2.0
        h = prev_h * 2.0
        n_rows = max(2, prev_rows // 2)
        n_cols = max(2, prev_cols // 2)

        # 降采样：2×2 块平均
        coarse = np.zeros((n_rows, n_cols), dtype=np.float32)
        for r in range(n_rows):
            for c in range(n_cols):
                block = prev_single[
                    r * 2 : min((r + 1) * 2, prev_rows),
                    c * 2 : min((c + 1) * 2, prev_cols),
                ]
                coarse[r, c] = np.mean(block) if block.size > 0 else 0

        self.levels.insert(0, (S, h, n_rows, n_cols, coarse))
        return len(self.levels)

    def build_pyramid(self, num_levels):
        """构建 num_levels 层金字塔（0=最粗, N-1=原图分辨率）"""
        while len(self.levels) < num_levels:
            self.add_coarser_level()

    def get_level(self, level_idx):
        """获取第 level_idx 层 (0=最粗)"""
        return self.levels[level_idx]


# ============================================================
#  三角网格上采样
# ============================================================

def upsample_tri_grid(coarse_single, coarse_S, coarse_h,
                      n_fine_rows, n_fine_cols, fine_S, fine_h,
                      method="bilinear"):
    """
    将粗网格单通道数据上采样到细网格。

    Args:
        coarse_single: [n_coarse_rows, n_coarse_cols]
        coarse_S, coarse_h: 粗网格几何
        n_fine_rows, n_fine_cols: 目标细网格尺寸
        fine_S, fine_h: 细网格几何
        method: "nearest" | "bilinear"

    Returns:
        fine_single: [n_fine_rows, n_fine_cols]
    """
    n_coarse_rows, n_coarse_cols = coarse_single.shape
    fine = np.zeros((n_fine_rows, n_fine_cols), dtype=np.float32)

    for fr in range(n_fine_rows):
        for fc in range(n_fine_cols):
            # 细三角中心坐标
            cx, cy = tri_center(fr, fc, fine_S, fine_h)

            # 映射到粗网格浮点坐标
            if is_upward(fr, fc):
                cr_float = cy / coarse_h - 2.0 / 3.0
            else:
                cr_float = cy / coarse_h - 1.0 / 3.0
            cc_float = 2.0 * cx / coarse_S

            # clamp 到粗网格范围内
            cr_float = np.clip(cr_float, 0, n_coarse_rows - 1.001)
            cc_float = np.clip(cc_float, 0, n_coarse_cols - 1.001)

            if method == "nearest":
                cr = int(np.clip(round(cr_float), 0, n_coarse_rows - 1))
                cc = int(np.clip(round(cc_float), 0, n_coarse_cols - 1))
                fine[fr, fc] = coarse_single[cr, cc]

            elif method == "bilinear":
                cr0 = int(np.floor(cr_float))
                cc0 = int(np.floor(cc_float))
                cr1 = min(cr0 + 1, n_coarse_rows - 1)
                cc1 = min(cc0 + 1, n_coarse_cols - 1)
                cr0 = max(cr0, 0)
                cc0 = max(cc0, 0)

                wr = cr_float - cr0
                wc = cc_float - cc0

                fine[fr, fc] = (
                    (1 - wr) * (1 - wc) * coarse_single[cr0, cc0]
                    + wr * (1 - wc) * coarse_single[cr1, cc0]
                    + (1 - wr) * wc * coarse_single[cr0, cc1]
                    + wr * wc * coarse_single[cr1, cc1]
                )

    return fine


# ============================================================
#  自相似细化（非局部自相似纹理迁移）
# ============================================================

def self_similar_refine(fine_single, coarse_single, coarse_S, coarse_h,
                        fine_S, fine_h, n_fine_rows, n_fine_cols,
                        search_radius=8, k_neighbors=3, strength=0.3):
    """
    利用三角网格的自相似性细化上采样结果。

    对于每个细三角，在粗网格中搜索最相似的三角，
    将其细粒度子三角的模式迁移过来。

    原理：Sierpinski 剖分的自相似性保证了粗三角和其子三角
    之间的关系在图像中重复出现。找到相似的粗三角，
    就找到了子三角的模板。

    Args:
        fine_single: 初始上采样结果 [n_fine_rows, n_fine_cols]
        coarse_single: 粗网格单通道数据
        search_radius: 搜索窗口半径（粗网格单位）
        k_neighbors: 用多少个相似邻居
        strength: 细化强度 (0=不改, 1=完全替换)

    Returns:
        refined: 细化后的 fine_single
    """
    n_coarse_rows, n_coarse_cols = coarse_single.shape
    refined = fine_single.copy()

    # 对每个粗三角，预计算其4个子三角的"相对值模式"
    # 相对值 = 子三角值 - 父三角值（去均值，保留纹理）

    # 构建粗网格特征：每个粗三角的特征 = [自身值, 3个邻居值]
    features = np.zeros((n_coarse_rows, n_coarse_cols, 4), dtype=np.float32)
    for r in range(n_coarse_rows):
        for c in range(n_coarse_cols):
            features[r, c, 0] = coarse_single[r, c]
            for i, (nr, nc) in enumerate(neighbors_of(r, c)):
                if 0 <= nr < n_coarse_rows and 0 <= nc < n_coarse_cols:
                    features[r, c, i + 1] = coarse_single[nr, nc]
                else:
                    features[r, c, i + 1] = coarse_single[r, c]

    # 对每个细三角，找到对应的粗父三角，在搜索窗口找相似三角
    for fr in range(n_fine_rows):
        for fc in range(n_fine_cols):
            # 父三角在粗网格的坐标
            cr = fr // 2
            cc = fc // 2
            if cr >= n_coarse_rows or cc >= n_coarse_cols:
                continue

            parent_val = coarse_single[cr, cc]
            parent_feat = features[cr, cc]

            # 搜索窗口
            r_min = max(0, cr - search_radius)
            r_max = min(n_coarse_rows, cr + search_radius + 1)
            c_min = max(0, cc - search_radius)
            c_max = min(n_coarse_cols, cc + search_radius + 1)

            # 收集窗口内最相似的K个粗三角
            best_dists = []
            for sr in range(r_min, r_max):
                for sc in range(c_min, c_max):
                    if sr == cr and sc == cc:
                        continue
                    # 特征相似度
                    diff = features[sr, sc] - parent_feat
                    dist = np.sum(diff * diff)
                    best_dists.append((dist, sr, sc))

            best_dists.sort(key=lambda x: x[0])
            best_dists = best_dists[:k_neighbors]

            if not best_dists:
                continue

            # 从相似粗三角的子三角模式中合成细三角值
            # 相似粗三角 (sr, sc) 对应的细三角 (2*sr+dr, 2*sc+dc)
            # 其中 (dr, dc) 与 (fr%2, fc%2) 同余
            dr = fr % 2
            dc = fc % 2

            sum_vals = 0.0
            sum_weights = 0.0

            for dist, sr, sc in best_dists:
                sfr = 2 * sr + dr
                sfc = 2 * sc + dc

                if 0 <= sfr < n_fine_rows and 0 <= sfc < n_fine_cols:
                    # 相似粗三角的子三角值（从初始上采样中取）
                    child_val = fine_single[sfr, sfc]
                    # 去父三角均值，加回当前父三角
                    adjusted = child_val - coarse_single[sr, sc] + parent_val
                    w = 1.0 / max(dist, 1e-6)
                    sum_vals += adjusted * w
                    sum_weights += w

            if sum_weights > 1e-6:
                sim_val = sum_vals / sum_weights
                refined[fr, fc] = (
                    (1 - strength) * fine_single[fr, fc] + strength * sim_val
                )

    return refined


# ============================================================
#  完整超分管线 (修正版)
# ============================================================

def super_resolve(pil_image, triangle_side, zoom=2,
                  correct_iterations=3, edge_sensitivity=0.20,
                  use_self_sim=False, progress_callback=None):
    """
    Sierpinski 超分辨率。

    方法：三角网格天然分辨率无关。
    1. 在粗网格完成完整管线 → coarse RGB（每个三角一个颜色）
    2. 在 zoom× 输出分辨率下，将粗三角网格渲染为图像（三角被放大）
    3. 在放大的三角内部做剖分 + 双线性插值子像素颜色
    4. 得到平滑的高分辨率输出

    这利用了三角表示的核心优势：三角颜色是连续的，
    渲染分辨率可以任意选择。
    """
    img = pil_image.convert("RGB")
    W, H = img.size
    pixels = np.array(img).astype(np.float32)

    S = float(triangle_side)
    h = S * math.sqrt(3) / 2.0
    n_cols = int(W / (S / 2.0)) + 3
    n_rows = int(H / h) + 2

    def prog(pct):
        if progress_callback:
            progress_callback(pct)

    # Step 1: 粗网格完整管线
    prog(5)
    coarse_single = sample_single_channels(
        pixels, S, h, n_rows, n_cols, sample_radius=1.0
    )
    prog(20)
    coarse_borrowed = borrow_neighbors(coarse_single, n_rows, n_cols, edge_mode="mirror")
    prog(35)
    coarse_rgb = correct_triangular_isp(
        coarse_single, coarse_borrowed, n_rows, n_cols,
        iterations=correct_iterations,
        edge_sensitivity=edge_sensitivity * 200,
    )

    # Step 2: 在输出分辨率构建细网格，用双线性插值赋色
    prog(50)
    out_W = int(W * zoom)
    out_H = int(H * zoom)
    S_fine = S / zoom         # 细三角更小
    h_fine = h / zoom
    n_fine_cols = int(out_W / (S_fine / 2.0)) + 3
    n_fine_rows = int(out_H / h_fine) + 2

    # 细网格每个三角的颜色 = 双线性插值粗三角颜色
    fine_rgb = np.zeros((n_fine_rows, n_fine_cols, 3), dtype=np.float32)

    for fr in range(n_fine_rows):
        for fc in range(n_fine_cols):
            # 细三角中心 → 粗网格浮点坐标
            cx, cy = tri_center(fr, fc, S_fine, h_fine)
            if is_upward(fr, fc):
                cr_float = cy / h - 2.0 / 3.0
            else:
                cr_float = cy / h - 1.0 / 3.0
            cc_float = 2.0 * cx / S

            # clamp
            cr_float = np.clip(cr_float, 0, n_rows - 1.001)
            cc_float = np.clip(cc_float, 0, n_cols - 1.001)

            # 双线性插值
            cr0 = int(np.floor(cr_float))
            cc0 = int(np.floor(cc_float))
            cr1 = min(cr0 + 1, n_rows - 1)
            cc1 = min(cc0 + 1, n_cols - 1)
            wr = cr_float - cr0
            wc = cc_float - cc0

            fine_rgb[fr, fc] = (
                (1 - wr) * (1 - wc) * coarse_rgb[cr0, cc0]
                + wr * (1 - wc) * coarse_rgb[cr1, cc0]
                + (1 - wr) * wc * coarse_rgb[cr0, cc1]
                + wr * wc * coarse_rgb[cr1, cc1]
            )

    # Step 3: 可选自相似细化
    if use_self_sim:
        prog(65)
        fine_rgb = _edge_aware_sharpen(fine_rgb, coarse_rgb, n_fine_rows,
                                       n_fine_cols, n_rows, n_cols, h, S)

    # Step 4: 渲染
    prog(80)
    result = render_triangles(fine_rgb, S_fine, h_fine,
                              n_fine_rows, n_fine_cols, out_W, out_H)

    prog(100)
    return result


def _edge_aware_sharpen(fine_rgb, coarse_rgb, n_fine_rows, n_fine_cols,
                        n_coarse_rows, n_coarse_cols, coarse_h, coarse_S):
    """对双线性插值结果做边缘感知锐化，恢复被模糊的边界"""
    from triangle_engine import tri_center, is_upward
    import math

    sharpened = fine_rgb.copy()

    for fr in range(n_fine_rows):
        for fc in range(n_fine_cols):
            # 找到最近的两个粗三角（按距离）
            cx = fc * coarse_S / (2 * (n_fine_cols / n_coarse_cols))  # approx
            # Actually, use the center-to-coarse mapping
            # Simplified: get the two nearest coarse triangles and blend by edge similarity

            cr_float = fr * n_coarse_rows / n_fine_rows
            cc_float = fc * n_coarse_cols / n_fine_cols
            cr_float = np.clip(cr_float, 0, n_coarse_rows - 1.001)
            cc_float = np.clip(cc_float, 0, n_coarse_cols - 1.001)

            cr0 = int(np.floor(cr_float))
            cc0 = int(np.floor(cc_float))
            cr1 = min(cr0 + 1, n_coarse_rows - 1)
            cc1 = min(cc0 + 1, n_coarse_cols - 1)

            # 四个角的粗三角颜色
            colors = [
                coarse_rgb[cr0, cc0],
                coarse_rgb[cr1, cc0],
                coarse_rgb[cr0, cc1],
                coarse_rgb[cr1, cc1],
            ]
            coords = [(cr0, cc0), (cr1, cc0), (cr0, cc1), (cr1, cc1)]

            # 边缘检测：颜色差异最大的方向 → 边缘方向
            my_color = fine_rgb[fr, fc].astype(np.float32)
            diffs = [np.sqrt(np.sum((my_color - c.astype(np.float32)) ** 2)) for c in colors]
            max_diff = max(diffs) if diffs else 255

            if max_diff > 40:  # 跨边缘
                # 偏向最近的粗三角颜色（保留边缘）
                nearest_idx = np.argmin([abs(cr_float - cr) + abs(cc_float - cc)
                                          for cr, cc in coords])
                sharpened[fr, fc] = 0.7 * colors[nearest_idx] + 0.3 * my_color

    return sharpened


def recursive_super_resolve(pil_image, triangle_side, zoom_per_step=2,
                            steps=1, **kwargs):
    """
    递归超分：每次在上次输出基础上再放大。

    zoom_per_step: 每次放大倍数
    steps: 递归次数
    总放大 = zoom_per_step ** steps
    """
    result = pil_image
    current_side = triangle_side

    for i in range(steps):
        result = super_resolve(
            result, current_side, zoom=zoom_per_step, **kwargs
        )
        current_side = current_side / zoom_per_step

    return result


# ============================================================
#  CLI 测试
# ============================================================

if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("用法: python triangle_superres.py <image> [zoom]")
        sys.exit(1)

    img = Image.open(sys.argv[1])
    zoom = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    print(f"输入: {img.size}, 三角边长=16, zoom={zoom}×")
    t0 = time.time()
    result = super_resolve(img, triangle_side=16, zoom=zoom,
                           use_self_sim=True)
    elapsed = time.time() - t0
    out_path = f"superres_{zoom}x.png"
    result.save(out_path)
    print(f"输出: {result.size}, 耗时 {elapsed:.1f}s, 已保存 {out_path}")
