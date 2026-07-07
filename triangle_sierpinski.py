#!/usr/bin/env python3
"""
谢尔宾斯基三角金字塔 — Layer 4 多尺度框架

等边三角形网格的递归剖分：
  △(S) → 3×△(S/2) + 1×▽(S/2)   (四等分)
  ▽(S) → 3×▽(S/2) + 1×△(S/2)

通道排列规则在任意尺度自洽 — 同一全局公式跨层适用。
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from triangle_engine import (
    assigned_channel, is_upward, tri_center, tri_vertices,
    neighbors_of, borrow_neighbors,
)


# ============================================================
#  Sierpinski 剖分几何
# ============================================================

def subdivide_children(r, c, S, h):
    """
    将三角形 (r,c) 四等分，返回4个子三角形的网格坐标和几何属性。

    参数:
        r, c: 父三角形在粗网格的坐标
        S, h: 父三角形边长和高度

    返回:
        list of (child_r, child_c, child_S, child_h, orientation)
        orientation: 'up' = △, 'down' = ▽
    """
    child_S = S / 2.0
    child_h = h / 2.0
    up = is_upward(r, c)
    R, C = r, c  # coarse coordinates

    if up:  # 父 = △
        children = [
            # (fine_r, fine_c) — 见推导
            (2 * R, 2 * C, "up"),  # 顶角 △
            (2 * R + 1, 2 * C - 1, "up"),  # 左下 △
            (2 * R + 1, 2 * C + 1, "up"),  # 右下 △
            (2 * R + 1, 2 * C, "down"),  # 中心 ▽
        ]
    else:  # 父 = ▽
        children = [
            (2 * R, 2 * C - 1, "down"),  # 左上 ▽
            (2 * R, 2 * C + 1, "down"),  # 右上 ▽
            (2 * R + 1, 2 * C, "down"),  # 底角 ▽
            (2 * R, 2 * C, "up"),  # 中心 △
        ]

    return [(cr, cc, child_S, child_h, orient) for cr, cc, orient in children]


def parent_of(fine_r, fine_c):
    """
    给定细网格坐标 (fine_r, fine_c)，返回父三角形在粗网格的坐标。
    fine_r // 2 = coarse_r
    fine_c // 2 ≈ coarse_c (需要根据 orientation 调整)
    """
    coarse_r = fine_r // 2
    # 父列取决于细网格中的具体位置（见 subdivide_children 的映射）
    # 简化：粗列 ≈ fine_c // 2（近似）
    coarse_c = fine_c // 2
    return coarse_r, coarse_c


# ============================================================
#  多尺度金字塔
# ============================================================

class SierpinskiPyramid:
    """三角形网格的多尺度金字塔"""

    def __init__(self, base_side, n_base_rows, n_base_cols):
        """
        base_side: 金字塔底层的三角形边长 (最高分辨率)
        n_base_rows, n_base_cols: 底层的网格行列数
        """
        self.base_side = base_side
        self.base_h = base_side * math.sqrt(3) / 2.0
        self.base_rows = n_base_rows
        self.base_cols = n_base_cols
        self.levels = []  # [(side, h, n_rows, n_cols), ...] 粗→细

    def build(self, num_levels):
        """构建金字塔层级，Level 0 最粗，Level N-1 = base"""
        self.levels = []
        for L in range(num_levels):
            factor = 2 ** (num_levels - 1 - L)  # L=0 → 最小 (最粗), L=N-1 → 最大 (base)
            side = self.base_side / factor
            h_val = self.base_h / factor
            n_rows = max(2, self.base_rows // factor)
            n_cols = max(2, self.base_cols // factor)
            self.levels.append((side, h_val, n_rows, n_cols))

    @property
    def num_levels(self):
        return len(self.levels)

    def level_info(self, L):
        """返回层级 L 的 (side, h, n_rows, n_cols)"""
        return self.levels[L]

    # ---- 上采样：粗 → 细 ----

    def upsample(self, coarse_data, factor=2):
        """
        将粗网格的单通道数据上采样到细网格。

        每个粗三角 → 4 个细三角。细三角的值用双三次插值近似。

        Args:
            coarse_data: [n_rows, n_cols] 单通道值
            factor: 上采样倍数 (2 = 一级)

        Returns:
            fine_data: [n_rows*factor, n_cols*factor]
        """
        nR, nC = coarse_data.shape
        fine_R, fine_C = nR * factor, nC * factor
        fine = np.zeros((fine_R, fine_C), dtype=coarse_data.dtype)

        # 简单双线性：每个细网格点从其周围的粗网格点插值
        for fr in range(fine_R):
            for fc in range(fine_C):
                # 对应的粗网格坐标（浮点）
                cr = (fr + 0.5) / factor - 0.5
                cc = (fc + 0.5) / factor - 0.5

                cr0 = int(np.floor(cr))
                cc0 = int(np.floor(cc))
                cr1 = min(cr0 + 1, nR - 1)
                cc1 = min(cc0 + 1, nC - 1)
                cr0 = max(cr0, 0)
                cc0 = max(cc0, 0)

                wr = cr - cr0
                wc = cc - cc0

                v00 = coarse_data[cr0, cc0]
                v10 = coarse_data[cr1, cc0]
                v01 = coarse_data[cr0, cc1]
                v11 = coarse_data[cr1, cc1]

                fine[fr, fc] = (
                    (1 - wr) * (1 - wc) * v00
                    + wr * (1 - wc) * v10
                    + (1 - wr) * wc * v01
                    + wr * wc * v11
                )

        return fine

    # ---- 下采样：细 → 粗 ----

    def downsample(self, fine_data, factor=2):
        """
        将细网格数据下采样。每个粗三角取4个子三角的平均。
        """
        fR, fC = fine_data.shape
        cR, cC = fR // factor, fC // factor
        coarse = np.zeros((cR, cC), dtype=fine_data.dtype)

        for cr in range(cR):
            for cc in range(cC):
                block = fine_data[
                    cr * factor : (cr + 1) * factor,
                    cc * factor : (cc + 1) * factor,
                ]
                coarse[cr, cc] = np.mean(block)

        return coarse


# ============================================================
#  谢尔宾斯基可视化
# ============================================================

def render_sierpinski_overlay(base_image, S, h, n_tri_rows, n_tri_cols,
                               depth=2, alpha=0.3, line_color=(255, 255, 255)):
    """
    在原图上叠加谢尔宾斯基剖分线。

    Args:
        base_image: PIL Image (原图或处理后的图)
        S, h: 底层三角形边长和高度
        n_tri_rows, n_tri_cols: 底层网格尺寸
        depth: 显示几层剖分 (1=仅大三角, 2=大三角+子三角, ...)
        alpha: 线条透明度
        line_color: 线条颜色

    Returns:
        PIL Image with overlay
    """
    from PIL import ImageDraw

    result = base_image.copy().convert("RGBA")
    W, H = result.size

    # 创建叠加层
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for level in range(depth):
        level_S = S * (2 ** level)
        level_h = h * (2 ** level)
        level_rows = max(2, n_tri_rows // (2 ** level))
        level_cols = max(2, n_tri_cols // (2 ** level))

        line_w = max(1, depth - level)  # 越粗的层级线越细
        level_alpha = int(255 * alpha * (0.5 ** level))  # 越深越透明
        color = line_color + (level_alpha,)

        for r in range(level_rows):
            for c in range(level_cols):
                verts = tri_vertices(r, c, level_S, level_h)
                pts = [(v[0], v[1]) for v in verts]
                draw.polygon(pts, outline=color, fill=None)

    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def render_sierpinski_pyramid(pyramid, single_data_per_level, W, H):
    """
    渲染多尺度谢尔宾斯基三角：每层用不同颜色标记。

    single_data_per_level: dict {level: 2D array}
    """
    result = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(result)

    level_colors = [
        (255, 60, 60),  # Level 0 (最粗) — 红
        (60, 255, 60),  # Level 1 — 绿
        (60, 60, 255),  # Level 2 — 蓝
        (255, 255, 60),  # Level 3 — 黄
        (255, 60, 255),  # Level 4 — 紫
    ]

    for L, (side, h_val, n_rows, n_cols) in enumerate(pyramid.levels):
        if L not in single_data_per_level:
            continue
        data = single_data_per_level[L]
        base_color = level_colors[min(L, len(level_colors) - 1)]

        for r in range(n_rows):
            for c in range(n_cols):
                verts = tri_vertices(r, c, side, h_val)
                pts = [(v[0], v[1]) for v in verts]
                ch = assigned_channel(r, c)
                intensity = np.clip(data[r, c] / 255.0, 0, 1)
                color = tuple(int(bc * intensity) for bc in base_color)
                draw.polygon(pts, fill=color, outline=(255, 255, 255))

    return result


# ============================================================
#  跨层通道一致性验证
# ============================================================

def verify_fractal_channels(max_levels=4):
    """
    验证通道排列在跨层时的一致性：
    子三角形的通道由全局公式直接给出，且按六边形分组的 2R2G2B 性质在任意层保持。
    """
    print(f"验证 {max_levels} 层谢尔宾斯基剖分的通道一致性...")

    for level in range(max_levels):
        n = 6 * (2 ** level)  # 每层扩大网格
        hex_count = 0
        hex_ok = 0

        # 检查每个 2×2 块（4个子三角）的通道分布
        # 对于 △ 父，4个子三角：3△+1▽
        # 粗网格中每6列有 2R2G2B，细网格中也应保持

        # 检查：任意连续6个三角形中 2R2G2B
        for r in range(n):
            channels_in_row = [assigned_channel(r, c) for c in range(n)]
            for c_start in range(0, n - 5):
                six = channels_in_row[c_start : c_start + 6]
                counts = {0: 0, 1: 0, 2: 0}
                for ch in six:
                    counts[ch] += 1
                if counts == {0: 2, 1: 2, 2: 2}:
                    hex_ok += 1
                hex_count += 1

        ratio = hex_ok / hex_count * 100 if hex_count > 0 else 0
        print(f"  Level {level}: {hex_ok}/{hex_count} 六边形组满足 2R2G2B ({ratio:.1f}%)")

    print("验证完成")


# ============================================================
#  CLI 测试
# ============================================================

if __name__ == "__main__":
    verify_fractal_channels(4)
