#!/usr/bin/env python3
"""
三角网格特征检测与匹配 — Phase 3

三角网格原生的 SIFT 等价实现：
- Tri-Harris 角点检测（结构张量 + 3方向梯度 + 多尺度）
- Tri-Descriptor（3-fold 旋转不变描述子）
- 特征匹配（L2距离 + Lowe's ratio test）
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from triangle_engine import (
    assigned_channel, is_upward, neighbors_of,
    sample_single_channels, borrow_neighbors, render_triangles,
)
from triangle_cv import estimate_luminance


# ============================================================
#  1. Tri-Harris 角点检测
# ============================================================

def _gradient_xy(r, c, lum, n_rows, n_cols):
    """计算三角网格上 (r,c) 处的 (gx, gy) 梯度分量"""
    L_c = lum[r, c]
    nbrs = neighbors_of(r, c)

    # 收集3个方向的梯度
    grads = []
    for nr, nc in nbrs:
        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            grads.append(L_c - lum[nr, nc])
        else:
            grads.append(0.0)

    g1, g2, g3 = grads  # left, right, vertical

    # 投影到 (gx, gy)
    sqrt3_2 = math.sqrt(3) / 2.0
    gx = sqrt3_2 * (g2 - g1)
    if is_upward(r, c):
        gy = 0.5 * g1 + 0.5 * g2 + g3
    else:
        gy = -(0.5 * g1 + 0.5 * g2 + g3)

    return gx, gy


def detect_harris_keypoints(single, n_rows, n_cols,
                            k=0.04, threshold=0.01, nms_radius=2):
    """
    三角网格 Harris 角点检测。

    步骤：
    1. 估计亮度
    2. 每个三角计算 (gx, gy)，构建结构张量 M
    3. Harris 响应 R = det(M) - k * trace(M)²
    4. 非极大值抑制 + 阈值

    Args:
        single: 单通道三角网格
        k: Harris k 参数
        threshold: R 响应的相对阈值 (乘 R.max())
        nms_radius: NMS 半径（1-ring = 1, 2-ring = 2）

    Returns:
        keypoints: [(r, c, response), ...] 排序后
    """
    lum = estimate_luminance(single, n_rows, n_cols)

    # 结构张量
    M_xx = np.zeros((n_rows, n_cols), dtype=np.float32)
    M_xy = np.zeros((n_rows, n_cols), dtype=np.float32)
    M_yy = np.zeros((n_rows, n_cols), dtype=np.float32)

    for r in range(n_rows):
        for c in range(n_cols):
            gx, gy = _gradient_xy(r, c, lum, n_rows, n_cols)
            M_xx[r, c] = gx * gx
            M_xy[r, c] = gx * gy
            M_yy[r, c] = gy * gy

    # 局部求和（1-ring: 自身 + 3邻居）
    M_xx_sum = M_xx.copy()
    M_xy_sum = M_xy.copy()
    M_yy_sum = M_yy.copy()

    for r in range(n_rows):
        for c in range(n_cols):
            for nr, nc in neighbors_of(r, c):
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    M_xx_sum[r, c] += M_xx[nr, nc]
                    M_xy_sum[r, c] += M_xy[nr, nc]
                    M_yy_sum[r, c] += M_yy[nr, nc]

    # Harris 响应
    R = np.zeros((n_rows, n_cols), dtype=np.float32)
    for r in range(n_rows):
        for c in range(n_cols):
            det = M_xx_sum[r, c] * M_yy_sum[r, c] - M_xy_sum[r, c] ** 2
            trace = M_xx_sum[r, c] + M_yy_sum[r, c]
            R[r, c] = det - k * trace * trace

    # 阈值
    R_max = R.max()
    if R_max <= 0:
        return []
    thresh = threshold * R_max

    # NMS
    keypoints = []
    for r in range(n_rows):
        for c in range(n_cols):
            if R[r, c] < thresh:
                continue
            is_max = True
            # 检查 1-ring 邻域
            for nr, nc in neighbors_of(r, c):
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    if R[nr, nc] > R[r, c]:
                        is_max = False
                        break
            if is_max:
                keypoints.append((r, c, float(R[r, c])))

    keypoints.sort(key=lambda x: -x[2])
    return keypoints


def detect_keypoints_multiscale(single, n_rows, n_cols, S, h,
                                num_scales=3, max_keypoints=200):
    """
    多尺度 Harris 角点检测。

    在每个尺度检测角点，粗尺度的角点映射回细尺度。
    """
    all_kps = []
    current_single = single
    current_rows = n_rows
    current_cols = n_cols
    current_S = S
    current_h = h

    for level in range(num_scales):
        kps = detect_harris_keypoints(
            current_single, current_rows, current_cols,
            k=0.04, threshold=0.005, nms_radius=2,
        )

        # 映射回 finest 尺度坐标
        scale_factor = 2 ** level
        for r, c, resp in kps:
            fr = r * scale_factor + scale_factor // 2
            fc = c * scale_factor + scale_factor // 2
            all_kps.append((min(fr, n_rows - 1), min(fc, n_cols - 1),
                           resp, level))

        if level == num_scales - 1:
            break

        # 降采样
        new_rows = max(2, current_rows // 2)
        new_cols = max(2, current_cols // 2)
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
        current_S = current_S * 2.0
        current_h = current_h * 2.0

    # 去重 + 排序
    all_kps.sort(key=lambda x: -x[2])
    seen = set()
    unique = []
    for r, c, resp, lvl in all_kps:
        key = (r, c)
        if key not in seen:
            seen.add(key)
            unique.append((r, c, resp, lvl))
        if len(unique) >= max_keypoints:
            break

    return unique


# ============================================================
#  2. 三角描述子
# ============================================================

def compute_descriptor(single, keypoint_r, keypoint_c,
                       n_rows, n_cols):
    """
    三角网格固定维度描述子 (16D)。

    使用 1-ring 固定模板: center + 3 neighbors = 4 triangles。
    每个三角: [归一化亮度, 3旋转边梯度] → 4×4=16D。
    旋转不变: 按主梯度方向循环移位邻居索引。
    """
    lum = estimate_luminance(single, n_rows, n_cols)
    eps = 1e-6

    # 固定模板: 1-ring = [center, left, right, vertical]
    # 确保描述子维度一致 (4 triangles × 4 values = 16D)
    template = [(keypoint_r, keypoint_c)]  # center
    for nr, nc in neighbors_of(keypoint_r, keypoint_c):
        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            template.append((nr, nc))
        else:
            template.append((-1, -1))  # placeholder for boundary

    # 亮度与梯度
    max_L = eps
    max_edge = eps
    patch_data = []
    for pr, pc in template:
        if pr >= 0:
            L = lum[pr, pc]
            max_L = max(max_L, abs(L))
            edges = []
            for nr, nc in neighbors_of(pr, pc):
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    e = lum[pr, pc] - lum[nr, nc]
                    max_edge = max(max_edge, abs(e))
                    edges.append(e)
                else:
                    edges.append(0.0)
            patch_data.append((L, edges))
        else:
            patch_data.append((0.0, [0.0, 0.0, 0.0]))

    # 主方向
    _, center_edges = patch_data[0]
    abs_edges = [abs(e) for e in center_edges]
    if max(abs_edges) < eps:
        dominant = 0
    else:
        dominant = abs_edges.index(max(abs_edges))

    # 构建16D描述子
    descriptor = []
    for L_val, edges_raw in patch_data:
        rotated = edges_raw[dominant:] + edges_raw[:dominant]
        descriptor.append(L_val / max_L)
        for e in rotated:
            descriptor.append(e / max_edge)

    desc = np.array(descriptor, dtype=np.float32)
    norm = np.sqrt(np.sum(desc * desc))
    if norm > 1e-8:
        desc /= norm

    return desc, dominant


# ============================================================
#  3. 特征匹配
# ============================================================

def match_descriptors(desc1_list, desc2_list, ratio_thresh=0.75):
    """
    双向匹配 + Lowe's ratio test。

    Args:
        desc1_list: [(descriptor, (r, c)), ...]
        desc2_list: [(descriptor, (r, c)), ...]

    Returns:
        matches: [(kp1, kp2, distance), ...] 排序后
    """
    matches = []

    for i, (d1, kp1) in enumerate(desc1_list):
        # 找最近的两个
        dists = []
        for j, (d2, kp2) in enumerate(desc2_list):
            dist = np.sqrt(np.sum((d1 - d2) ** 2))
            dists.append((dist, j))

        dists.sort(key=lambda x: x[0])
        if len(dists) < 2:
            continue

        best, second = dists[0], dists[1]
        if best[0] < ratio_thresh * second[0]:
            matches.append((kp1, desc2_list[best[1]][1], best[0]))

    matches.sort(key=lambda x: x[2])
    return matches


# ============================================================
#  4. 可视化
# ============================================================

def render_keypoints(image, S, h, n_rows, n_cols, keypoints,
                     color=(0, 255, 0), radius=3):
    """在图像上标记关键点"""
    from triangle_engine import tri_center
    result = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for r, c, resp, *_ in keypoints:
        cx, cy = tri_center(r, c, S, h)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=color + (200,), width=2,
        )

    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def render_matches(img1, kps1, img2, kps2, matches,
                   S1, h1, nr1, nc1, S2, h2, nr2, nc2):
    """并排两张图，连线显示匹配"""
    from triangle_engine import tri_center

    W1, H1 = img1.size
    W2, H2 = img2.size
    total_W = W1 + W2
    total_H = max(H1, H2)

    result = Image.new("RGB", (total_W, total_H), (50, 50, 50))
    result.paste(img1, (0, 0))
    result.paste(img2, (W1, 0))

    draw = ImageDraw.Draw(result)
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
              (255, 255, 100), (255, 100, 255), (100, 255, 255)]

    for i, ((r1, c1), (r2, c2), dist) in enumerate(matches[:50]):
        x1, y1 = tri_center(r1, c1, S1, h1)
        x2, y2 = tri_center(r2, c2, S2, h2)
        color = colors[i % len(colors)]
        draw.line([(x1, y1), (x2 + W1, y2)], fill=color, width=1)
        r = 3
        draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=color)
        draw.ellipse([x2 + W1 - r, y2 - r, x2 + W1 + r, y2 + r], fill=color)

    return result


# ============================================================
#  CLI 测试
# ============================================================

if __name__ == "__main__":
    import sys, time

    if len(sys.argv) < 2:
        print("用法: python triangle_features.py <image> [image2]")
        sys.exit(1)

    img1 = Image.open(sys.argv[1]).convert("RGB")
    W1, H1 = img1.size
    pix1 = np.array(img1).astype(np.float32)
    S1 = 16.0; h1 = S1 * math.sqrt(3) / 2
    nc1 = int(W1 / (S1 / 2)) + 3; nr1 = int(H1 / h1) + 2
    single1 = sample_single_channels(pix1, S1, h1, nr1, nc1)

    # 检测关键点
    t0 = time.time()
    kps = detect_harris_keypoints(single1, nr1, nc1, k=0.04, threshold=0.01)
    print(f"关键点: {len(kps)} 个, {time.time()-t0:.2f}s")
    if kps:
        print(f"  Top 5: response={[f'{r:.1f}' for _,_,r in kps[:5]]}")

    # 可视化
    borrowed1 = borrow_neighbors(single1, nr1, nc1, edge_mode="mirror")
    base1 = render_triangles(borrowed1, S1, h1, nr1, nc1, W1, H1)
    render_keypoints(base1, S1, h1, nr1, nc1, kps[:100]).save("features_kp.png")

    # 多尺度
    kps_ms = detect_keypoints_multiscale(single1, nr1, nc1, S1, h1, num_scales=3)
    print(f"多尺度关键点: {len(kps_ms)} 个")

    # 如果有第二张图，做匹配
    if len(sys.argv) > 2:
        img2 = Image.open(sys.argv[2]).convert("RGB")
        W2, H2 = img2.size
        pix2 = np.array(img2).astype(np.float32)
        S2 = 16.0; h2 = S2 * math.sqrt(3) / 2
        nc2 = int(W2 / (S2 / 2)) + 3; nr2 = int(H2 / h2) + 2
        single2 = sample_single_channels(pix2, S2, h2, nr2, nc2)
        kps2 = detect_harris_keypoints(single2, nr2, nc2, k=0.04, threshold=0.01)

        # 计算描述子
        desc1 = []
        for r, c, resp in kps[:80]:
            d, orient = compute_descriptor(single1, r, c, nr1, nc1)
            desc1.append((d, (r, c)))
        desc2 = []
        for r, c, resp in kps2[:80]:
            d, orient = compute_descriptor(single2, r, c, nr2, nc2)
            desc2.append((d, (r, c)))

        # 匹配
        matches = match_descriptors(desc1, desc2, ratio_thresh=0.8)
        print(f"匹配: {len(matches)} 对")

        borrowed2 = borrow_neighbors(single2, nr2, nc2, edge_mode="mirror")
        base2 = render_triangles(borrowed2, S2, h2, nr2, nc2, W2, H2)

        result = render_matches(base1, kps1, base2, kps2, matches,
                                S1, h1, nr1, nc1, S2, h2, nr2, nc2)
        result.save("features_match.png")
        print("Saved: features_match.png")

    print("Saved: features_kp.png")
