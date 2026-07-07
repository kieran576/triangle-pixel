#!/usr/bin/env python3
"""
三角引擎性能优化 — numba JIT 加速版
"""

import math
import time
import numpy as np
from numba import njit

from triangle_engine import render_triangles, sample_single_channels as _sample_slow
from triangle_engine import borrow_neighbors as _borrow_slow
from triangle_engine import correct_triangular_isp as _isp_slow


# ============================================================
#  内联工具函数 (numba 兼容)
# ============================================================

@njit(inline='always')
def _ch(r, c):
    """通道: 0=R 1=G 2=B"""
    i = c % 6
    if r % 2 == 0:  # R G B B G R
        if i == 0: return 0
        if i == 1: return 1
        if i == 2: return 2
        if i == 3: return 2
        if i == 4: return 1
        return 0
    else:  # B G R R G B
        if i == 0: return 2
        if i == 1: return 1
        if i == 2: return 0
        if i == 3: return 0
        if i == 4: return 1
        return 2


@njit(inline='always')
def _up(r, c):
    return (r + c) % 2 == 0


@njit(inline='always')
def _mir(v, mx):
    if v < 0: v = -v - 1
    elif v >= mx: v = 2 * mx - v - 1
    if v < 0: v = 0
    if v >= mx: v = mx - 1
    return v


# ============================================================
#  加速 1: 采样
# ============================================================

@njit(cache=True)
def _sample_fast(pixels, S, h, nr, nc):
    """numba 加速采样"""
    H, W = pixels.shape[:2]
    out = np.zeros((nr, nc), dtype=np.float32)
    for r in range(nr):
        for c in range(nc):
            x = c * S / 2.0
            y = r * h + (2.0 * h / 3.0 if _up(r, c) else h / 3.0)
            ix = int(min(max(x, 0), W - 1))
            iy = int(min(max(y, 0), H - 1))
            out[r, c] = pixels[iy, ix, _ch(r, c)]
    return out


def sample_fast(pixels, S, h, nr, nc, sample_radius=1.5):
    return _sample_fast(pixels.astype(np.float32), S, h, nr, nc)


# ============================================================
#  加速 2: 邻居借用
# ============================================================

@njit(cache=True)
def _borrow_fast(single, nr, nc):
    """numba 加速借用"""
    rgb = np.zeros((nr, nc, 3), dtype=np.float32)
    for r in range(nr):
        for c in range(nc):
            col = np.zeros(3, dtype=np.float32)
            nbrs = [(r, c - 1), (r, c + 1), (r + 1, c)] if _up(r, c) else \
                   [(r, c - 1), (r, c + 1), (r - 1, c)]
            for nr_, nc_ in nbrs:
                mnr = _mir(nr_, nr)
                mnc = _mir(nc_, nc)
                col[_ch(mnr, mnc)] = single[mnr, mnc]
            rgb[r, c] = col
    return rgb


def borrow_fast(single, nr, nc, edge_mode="mirror"):
    return _borrow_fast(single, nr, nc)


# ============================================================
#  加速 3: ISP 校正
# ============================================================

@njit(cache=True)
def _median3(arr, nr, nc):
    out = arr.copy()
    for r in range(nr):
        for c in range(nc):
            vals = [arr[r, c]]
            nbrs = [(r, c - 1), (r, c + 1), (r + 1, c)] if _up(r, c) else \
                   [(r, c - 1), (r, c + 1), (r - 1, c)]
            for nr_, nc_ in nbrs:
                if 0 <= nr_ < nr and 0 <= nc_ < nc:
                    vals.append(arr[nr_, nc_])
            a = np.array(vals)
            out[r, c] = np.median(a)
    return out


@njit(cache=True)
def _isp_fast(single, borrowed, nr, nc, iters, esens):
    """numba 加速 ISP 校正"""
    corr = borrowed.copy()
    eps = 1e-6
    for it in range(iters):
        lum = (corr[:, :, 0] + corr[:, :, 1] + corr[:, :, 2]) / 3.0
        drg = corr[:, :, 0] - corr[:, :, 1]
        drb = corr[:, :, 0] - corr[:, :, 2]
        dgb = corr[:, :, 1] - corr[:, :, 2]
        drg = _median3(drg, nr, nc)
        drb = _median3(drb, nr, nc)
        dgb = _median3(dgb, nr, nc)

        new_rgb = np.zeros_like(corr)
        for r in range(nr):
            for c in range(nc):
                own = _ch(r, c)
                kv = single[r, c]
                Ic = lum[r, c]
                nbrs = [(r, c - 1), (r, c + 1), (r + 1, c)] if _up(r, c) else \
                       [(r, c - 1), (r, c + 1), (r - 1, c)]

                ws = np.zeros(3, dtype=np.float32)
                vrg = np.zeros(3, dtype=np.float32)
                vrb = np.zeros(3, dtype=np.float32)
                vgb = np.zeros(3, dtype=np.float32)
                dsrc = np.zeros(3, dtype=np.float32)
                dsw = np.zeros(3, dtype=np.float32)

                for j in range(3):
                    nr_, nc_ = nbrs[j]
                    if 0 <= nr_ < nr and 0 <= nc_ < nc:
                        In = lum[nr_, nc_]
                        w = np.exp(-(Ic - In) * (Ic - In) / (2 * esens * esens))
                        w = max(w, 1e-8)
                        ws[j] = w
                        vrg[j] = drg[nr_, nc_]
                        vrb[j] = drb[nr_, nc_]
                        vgb[j] = dgb[nr_, nc_]
                        nc_ch = _ch(nr_, nc_)
                        dsrc[nc_ch] = single[nr_, nc_]
                        dsw[nc_ch] = w
                    else:
                        ws[j] = 1e-8

                sw = max(ws.sum(), eps)
                erg = (vrg * ws).sum() / sw
                erb = (vrb * ws).sum() / sw
                egb = (vgb * ws).sum() / sw

                if own == 0:
                    Rv = kv
                    gw = dsw[1]; gd = dsrc[1]
                    Gv = gw * gd + (1 - gw) * (Rv - erg) if gw > 0 else (Rv - erg)
                    bw = dsw[2]; bd = dsrc[2]
                    Bv = bw * bd + (1 - bw) * (Rv - erb) if bw > 0 else (Rv - erb)
                elif own == 1:
                    Gv = kv
                    rw = dsw[0]; rd = dsrc[0]
                    Rv = rw * rd + (1 - rw) * (Gv + erg) if rw > 0 else (Gv + erg)
                    bw = dsw[2]; bd = dsrc[2]
                    Bv = bw * bd + (1 - bw) * (Gv - egb) if bw > 0 else (Gv - egb)
                else:
                    Bv = kv
                    rw = dsw[0]; rd = dsrc[0]
                    Rv = rw * rd + (1 - rw) * (Bv + erb) if rw > 0 else (Bv + erb)
                    gw = dsw[1]; gd = dsrc[1]
                    Gv = gw * gd + (1 - gw) * (Bv + egb) if gw > 0 else (Bv + egb)

                new_rgb[r, c, 0] = min(max(Rv, 0.0), 255.0)
                new_rgb[r, c, 1] = min(max(Gv, 0.0), 255.0)
                new_rgb[r, c, 2] = min(max(Bv, 0.0), 255.0)

        corr = new_rgb
    return corr


def isp_fast(single, borrowed, nr, nc, iterations=3, edge_sensitivity=40):
    return _isp_fast(single.astype(np.float32), borrowed.astype(np.float32),
                     nr, nc, iterations, edge_sensitivity)


# ============================================================
#  加速 4: 快速渲染 (numpy 逐像素填充, 替代 PIL polygon)
# ============================================================

def render_fast(rgb, S, h, nr, nc, W, H):
    """快速三角形渲染 — numpy 逐像素查找所属三角并填充"""
    out = np.zeros((H, W, 3), dtype=np.uint8)

    for py in range(H):
        # 确定像素所在的行区间
        tri_row = int(py / h)
        if tri_row >= nr:
            tri_row = nr - 1

        row_start_y = tri_row * h
        row_end_y = (tri_row + 1) * h

        for px in range(W):
            # 确定列
            tri_col = int(px / (S / 2.0))
            if tri_col >= nc:
                tri_col = nc - 1

            # 确定像素在 △ 还是 ▽
            x_in_cell = px - tri_col * S / 2.0
            y_in_cell = py - row_start_y

            # 对角线划分：△在左上-右下对角线上方
            # 方程: x/S + y/h > 1 则属于 △ (朝上), 否则属于 ▽ (朝下)
            if _up(tri_row, tri_col):
                # △: x/S + y/h > 1 属于该三角
                if x_in_cell / (S / 2.0) + y_in_cell / h > 1.0:
                    r, c = tri_row, tri_col
                else:
                    r, c = tri_row, max(0, tri_col - 1)
            else:
                # ▽: x/S + y/h < S/2... 
                # 简化：用最近三角
                r, c = tri_row, min(tri_col, nc - 1)

            r = min(max(r, 0), nr - 1)
            c = min(max(c, 0), nc - 1)
            out[py, px] = np.clip(rgb[r, c], 0, 255).astype(np.uint8)

    return out


# ============================================================
#  完整快速管线
# ============================================================

def process_fast(pixels, S, h, nr, nc, W, H, correct_iters=3, edge_sens=40):
    """端到端快速管线: 采样→借用→ISP→渲染"""
    single = _sample_fast(pixels, S, h, nr, nc)
    borrowed = _borrow_fast(single, nr, nc)
    corrected = _isp_fast(single, borrowed, nr, nc, correct_iters, edge_sens)
    from PIL import Image
    result = render_triangles(corrected, S, h, nr, nc, W, H)
    return result


# ============================================================
#  性能测试
# ============================================================

if __name__ == "__main__":
    from PIL import Image

    img = Image.open("test_edge.png").convert("RGB")
    W, H = img.size
    pixels = np.array(img).astype(np.float32)
    S = 16.0; h = S * math.sqrt(3) / 2
    nc = int(W / (S / 2)) + 3
    nr = int(H / h) + 2

    print(f"Grid: {nr}x{nc}, triangles: {nr*nc}")
    print()

    # 预热
    _sample_fast(pixels, S, h, nr, nc)
    s1 = _sample_fast(pixels, S, h, nr, nc)
    _borrow_fast(s1, nr, nc)
    b1 = _borrow_fast(s1, nr, nc)
    _isp_fast(s1, b1, nr, nc, 1, 40)

    print("=== 加速对比 (5次平均) ===")
    for name, fn_slow, fn_fast, args_fast in [
        ("sample", lambda: _sample_slow(pixels, S, h, nr, nc, 1.0),
         lambda: _sample_fast(pixels, S, h, nr, nc), ()),
        ("borrow", lambda: _borrow_slow(s1, nr, nc, "mirror"),
         lambda: _borrow_fast(s1, nr, nc), ()),
        ("ISP x1", lambda: _isp_slow(s1, b1, nr, nc, 1, 40),
         lambda: _isp_fast(s1, b1, nr, nc, 1, 40), ()),
        ("ISP x3", lambda: _isp_slow(s1, b1, nr, nc, 3, 40),
         lambda: _isp_fast(s1, b1, nr, nc, 3, 40), ()),
    ]:
        # slow
        N = 10 if "ISP" not in name else 3
        t0 = time.time()
        for _ in range(N):
            fn_slow()
        t_slow = (time.time() - t0) / N

        # fast
        N2 = N * 3
        t0 = time.time()
        for _ in range(N2):
            fn_fast()
        t_fast = (time.time() - t0) / N2

        sp = t_slow / t_fast if t_fast > 0 else 0
        print(f"  {name:10s}: {t_slow*1000:6.1f}ms -> {t_fast*1000:6.1f}ms  ({sp:.1f}x)")

    # 端到端
    print()
    print("=== 端到端管线 ===")
    t0 = time.time()
    r1 = process_fast(pixels, S, h, nr, nc, W, H, 3, 40)
    t1 = time.time() - t0
    print(f"  Fast pipeline: {t1:.2f}s")
    r1.save("test_fast.png")
    print(f"  Saved: test_fast.png ({r1.size})")
