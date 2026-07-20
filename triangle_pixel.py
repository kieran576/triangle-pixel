#!/usr/bin/env python3
"""
三角形像素图像重采样器

将输入图片按等边三角形网格重新采样：
1. 每个三角形从原图提取 RGB
2. 按排列规则只保留单一通道（R/G/B 之一）
3. 每个三角形把自己的单通道送给三个邻居
4. 每个三角形从三个邻居收到的恰好是 R、G、B 各一个
5. 合成新颜色，渲染输出
"""

import math
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw

# mode 列表: borrow 在本文件 process() 内置;
#             raw/raw_downscale/correct/correct_isp 委托给 triangle_engine.process_pipeline.
try:
    from triangle_engine import process_pipeline
except ImportError:
    process_pipeline = None  # 仅用 borrow 模式时可缺


def process(input_path, output_path, triangle_side=20, debug_channels=False,
            mode="borrow"):
    """
    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        triangle_side: 等边三角形边长（像素）
        debug_channels: True 则用纯色标记每个三角形的通道（R=红, G=绿, B=蓝）
        mode: 处理模式
              "borrow" (默认) — 邻居捐赠, 本函数内置
              "raw" / "raw_downscale" / "correct" / "correct_isp"
                          — 委托给 triangle_engine.process_pipeline
                          (需要 NumPy + (可选) Numba 加速)
    """
    if mode != "borrow":
        if process_pipeline is None:
            raise RuntimeError(
                f"mode={mode!r} 需要 triangle_engine.process_pipeline,"
                "但导入失败. 请确认 triangle_engine.py 在同目录下."
            )
        img = Image.open(input_path).convert("RGB")
        out = process_pipeline(img, triangle_side=triangle_side, mode=mode)
        out.save(output_path)
        print(f"已保存到 {output_path} (mode={mode}, 三角形边长 {triangle_side}px)")
        return

    # 加载原图
    img = Image.open(input_path).convert("RGB")
    W, H = img.size
    pixels = np.array(img).astype(np.float32)

    S = float(triangle_side)
    h = S * math.sqrt(3) / 2.0  # 等边三角形高度

    # 三角形网格尺寸
    # 列间距 = S/2（每个列索引对应一个三角形，△/▽ 交替）
    col_spacing = S / 2.0
    n_tri_cols = int(W / col_spacing) + 3
    n_tri_rows = int(H / h) + 2

    # ---------- 通道分配 ----------
    # 0=R, 1=G, 2=B
    EVEN_ROW = [0, 1, 2, 2, 1, 0]  # 偶数行: R G B B G R
    ODD_ROW = [2, 1, 0, 0, 1, 2]  # 奇数行: B G R R G B

    def assigned_channel(r, c):
        """返回三角形 (r,c) 被分配的通道编号"""
        if r % 2 == 0:
            return EVEN_ROW[c % 6]
        else:
            return ODD_ROW[c % 6]

    def is_upward(r, c):
        """△ 朝上 (r+c 偶数) / ▽ 朝下 (r+c 奇数)"""
        return (r + c) % 2 == 0

    def tri_center(r, c):
        """三角形 (r,c) 的中心像素坐标"""
        x = c * col_spacing
        # 同一行内所有三角形处于相同垂直区间 [r*h, (r+1)*h]
        # △ 朝上: 底边在下，重心在 h/3 处（距底边）
        # ▽ 朝下: 底边在上，重心在 h/3 处（距底边）
        if is_upward(r, c):
            y = r * h + 2.0 * h / 3.0  # △ 重心偏下
        else:
            y = r * h + h / 3.0  # ▽ 重心偏上
        return x, y

    def tri_vertices(r, c):
        """三角形 (r,c) 的三个顶点坐标 [(x,y), ...]"""
        x, y = tri_center(r, c)
        if is_upward(r, c):
            # △：顶点在上，底边在下
            return [
                (x, r * h),  # 顶点
                (x - S / 2.0, (r + 1) * h),  # 左下
                (x + S / 2.0, (r + 1) * h),  # 右下
            ]
        else:
            # ▽：底边在上，顶点在下
            return [
                (x - S / 2.0, r * h),  # 左上
                (x + S / 2.0, r * h),  # 右上
                (x, (r + 1) * h),  # 顶点
            ]

    # ---------- 第1步：采样单通道 ----------
    print(f"三角形网格: {n_tri_rows} 行 × {n_tri_cols} 列 = {n_tri_rows * n_tri_cols} 个三角形")
    single = np.zeros((n_tri_rows, n_tri_cols), dtype=np.float32)

    for r in range(n_tri_rows):
        for c in range(n_tri_cols):
            cx, cy = tri_center(r, c)
            ix = int(np.clip(cx, 0, W - 1))
            iy = int(np.clip(cy, 0, H - 1))
            ch = assigned_channel(r, c)
            single[r, c] = pixels[iy, ix, ch]

    # ---------- 第2步：从邻居重建 RGB ----------
    new_rgb = np.zeros((n_tri_rows, n_tri_cols, 3), dtype=np.uint8)

    for r in range(n_tri_rows):
        for c in range(n_tri_cols):
            up = is_upward(r, c)

            if up:  # △：左/右/下 邻居均为 ▽
                neighbors = [(r, c - 1), (r, c + 1), (r + 1, c)]
            else:  # ▽：左/右/上 邻居均为 △
                neighbors = [(r, c - 1), (r, c + 1), (r - 1, c)]

            rgb = np.zeros(3, dtype=np.float32)
            for nr, nc in neighbors:
                if 0 <= nr < n_tri_rows and 0 <= nc < n_tri_cols:
                    nbr_ch = assigned_channel(nr, nc)
                    rgb[nbr_ch] = single[nr, nc]
                # 边界三角形：缺失的邻居通道保持0，该通道为黑色

            new_rgb[r, c] = np.clip(rgb, 0, 255).astype(np.uint8)

    # ---------- 第3步：渲染 ----------
    out_img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(out_img)

    for r in range(n_tri_rows):
        for c in range(n_tri_cols):
            verts = tri_vertices(r, c)
            pts = [(v[0], v[1]) for v in verts]

            if debug_channels:
                ch = assigned_channel(r, c)
                # 用纯色标记通道：R=红, G=绿, B=蓝（灰度表示强度）
                ch_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                color = ch_colors[ch]
            else:
                color = tuple(int(v) for v in new_rgb[r, c])

            draw.polygon(pts, fill=color, outline=None)

    out_img.save(output_path)
    print(f"已保存到 {output_path} (尺寸 {W}×{H}, 三角形边长 {S}px)")


def main():
    parser = argparse.ArgumentParser(
        description="三角形像素图像重采样器 — 基于等边三角形网格的通道置换图像处理"
    )
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("output", help="输出图片路径")
    parser.add_argument(
        "-s",
        "--side",
        "--triangle-side",
        type=float,
        default=20,
        dest="triangle_side",
        help="三角形边长（像素），默认 20",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["borrow", "raw", "raw_downscale", "correct", "correct_isp"],
        default="borrow",
        help=(
            "处理模式: borrow (默认, 邻居捐赠) / "
            "raw (单通道 CFA) / raw_downscale / "
            "correct (空间校正) / correct_isp (含 ISP 校正, 最慢)"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式: 用纯色显示每个三角形的通道分配 (R=红 G=绿 B=蓝)",
    )
    args = parser.parse_args()

    process(args.input, args.output,
            triangle_side=args.triangle_side,
            debug_channels=args.debug,
            mode=args.mode)


if __name__ == "__main__":
    main()
