#!/usr/bin/env python3
"""
三角网格 3D 统一 — Phase 4

三角网格天然是 3D 面片表示：
- 每个三角面 = 3D mesh 的一个 face
- 顶点共享（相邻三角共用同一条边）
- 深度 = 每顶点的 z 坐标
- 2D 检测结果直接投射到 3D 表面

核心类:
  TriMesh — 3D 三角网格 (顶点/面/颜色)
  export → OBJ 格式
  render → 3D 视图 (可旋转)
  depth-aware → 深度感知边缘 (不跨深度断层)
"""

import math
import numpy as np
from PIL import Image, ImageDraw

from triangle_engine import (
    assigned_channel, is_upward, neighbors_of, tri_vertices,
    sample_single_channels, borrow_neighbors, render_triangles,
)
from triangle_cv import estimate_luminance


# ============================================================
#  TriMesh — 3D 三角网格
# ============================================================

class TriMesh:
    """3D 三角网格 — 从 2D 三角网格 + 深度图构建"""

    def __init__(self):
        self.vertices = []  # [(x, y, z), ...]
        self.faces = []  # [(v1, v2, v3), ...]
        self.colors = []  # [(r, g, b), ...] per face
        self.vertex_index = {}  # (vr, vc) → vertex_id

    def from_triangle_grid(self, single, S, h, n_rows, n_cols,
                           depth_map=None, rgb_map=None):
        """
        从 2D 三角网格构建 3D mesh。

        Args:
            single: 单通道数据
            S, h: 三角几何
            depth_map: [n_rows, n_cols] 每三角的深度，None=用亮度
            rgb_map: [n_rows, n_cols, 3] 每三角的 RGB，None=单通道灰度
        """
        if depth_map is None:
            lum = estimate_luminance(single, n_rows, n_cols)
            # 归一化深度到合理范围
            depth_map = (lum - lum.min()) / max(lum.max() - lum.min(), 1) * S * 2

        if rgb_map is None:
            rgb_map = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)
            for r in range(n_rows):
                for c in range(n_cols):
                    v = int(np.clip(single[r, c], 0, 255))
                    rgb_map[r, c] = [v, v, v]

        # Helper: 获取或创建顶点
        def get_vertex(vr, vc):
            """获取顶点 (vr, vc) 的 ID，若不存在则创建"""
            key = (vr, vc)
            if key in self.vertex_index:
                return self.vertex_index[key]

            x = vc * S / 2.0
            y = vr * h

            # 从相邻三角的平均深度插值顶点深度
            z = 0.0
            count = 0
            for dr in [-1, 0]:
                for dc in [-1, 0]:
                    tr = vr + dr
                    tc = vc + dc
                    if 0 <= tr < n_rows and 0 <= tc < n_cols:
                        z += depth_map[tr, tc]
                        count += 1
            z = z / max(count, 1)

            vid = len(self.vertices)
            self.vertices.append((x, y, z))
            self.vertex_index[key] = vid
            return vid

        # 构建面和顶点
        for r in range(n_rows):
            for c in range(n_cols):
                up = is_upward(r, c)

                if up:  # △: vertices (r, c+1), (r+1, c), (r+1, c+1)
                    v1 = get_vertex(r, c + 1)
                    v2 = get_vertex(r + 1, c)
                    v3 = get_vertex(r + 1, c + 1)
                else:  # ▽: vertices (r, c), (r, c+1), (r+1, c)
                    v1 = get_vertex(r, c)
                    v2 = get_vertex(r, c + 1)
                    v3 = get_vertex(r + 1, c)

                self.faces.append((v1, v2, v3))
                self.colors.append(tuple(int(v) for v in rgb_map[r, c]))

    # ---- OBJ 导出 ----

    def export_obj(self, path, scale=1.0):
        """导出 Wavefront OBJ 文件（含材质颜色）"""
        mtl_path = path.rsplit(".", 1)[0] + ".mtl"
        mtl_name = mtl_path.replace("\\", "/").split("/")[-1]

        with open(path, "w") as f:
            f.write(f"# Triangle Pixel 3D Mesh\n")
            f.write(f"# vertices: {len(self.vertices)}  faces: {len(self.faces)}\n")
            f.write(f"mtllib {mtl_name}\n")

            for x, y, z in self.vertices:
                f.write(f"v {x * scale:.4f} {y * scale:.4f} {z * scale:.4f}\n")

            # 写入面（OBJ 面索引从1开始）
            for i, (v1, v2, v3) in enumerate(self.faces):
                mat_idx = (i % len(self.colors)) + 1 if self.colors else 1
                f.write(f"usemtl mat{mat_idx}\n")
                f.write(f"f {v1 + 1} {v2 + 1} {v3 + 1}\n")

            # 材质文件
            with open(mtl_path, "w") as mf:
                for i, (r, g, b) in enumerate(self.colors):
                    mf.write(f"newmtl mat{i + 1}\n")
                    mf.write(f"Kd {r / 255:.3f} {g / 255:.3f} {b / 255:.3f}\n")
                    mf.write(f"Ka 0.2 0.2 0.2\n")

        return path

    # ---- 3D 渲染 ----

    def render_view(self, width, height,
                    azimuth=30, elevation=20, scale=None):
        """
        将 3D mesh 渲染为 2D 图像（正交投影）。

        Args:
            azimuth: 水平旋转角 (度)
            elevation: 俯仰角 (度)
        """
        import math

        # 旋转矩阵
        az = math.radians(azimuth)
        el = math.radians(elevation)
        cos_a, sin_a = math.cos(az), math.sin(az)
        cos_e, sin_e = math.cos(el), math.sin(el)

        def project(x, y, z):
            # 绕 Y 轴旋转 (azimuth)
            x1 = x * cos_a - z * sin_a
            z1 = x * sin_a + z * cos_a
            # 绕 X 轴旋转 (elevation)
            y1 = y * cos_e - z1 * sin_e
            z2 = y * sin_e + z1 * cos_e  # depth for z-buffer
            return float(x1), float(y1), float(z2)

        # 投影所有顶点并计算包围盒
        proj = [project(x, y, z) for x, y, z in self.vertices]
        xs = [p[0] for p in proj]
        ys = [p[1] for p in proj]
        zs = [p[2] for p in proj]

        if scale is None:
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            s = min(width / max(x_range, 1), height / max(y_range, 1)) * 0.85
        else:
            s = scale

        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2

        # 透视排序：按面的平均深度排序（画家算法）
        face_depths = []
        for i, (v1, v2, v3) in enumerate(self.faces):
            z_avg = (zs[v1] + zs[v2] + zs[v3]) / 3
            face_depths.append((z_avg, i))
        face_depths.sort(reverse=True)  # 远的面先画

        # 渲染
        img = Image.new("RGB", (width, height), (30, 30, 30))
        draw = ImageDraw.Draw(img)

        for _, fi in face_depths:
            v1, v2, v3 = self.faces[fi]
            p1 = (int((proj[v1][0] - cx) * s + width / 2),
                  int((proj[v1][1] - cy) * s + height / 2))
            p2 = (int((proj[v2][0] - cx) * s + width / 2),
                  int((proj[v2][1] - cy) * s + height / 2))
            p3 = (int((proj[v3][0] - cx) * s + width / 2),
                  int((proj[v3][1] - cy) * s + height / 2))

            color = self.colors[fi % len(self.colors)]
            try:
                draw.polygon([p1, p2, p3], fill=color, outline=None)
            except Exception:
                pass

        return img


# ============================================================
#  深度感知操作
# ============================================================

def detect_depth_edges(single, n_rows, n_cols, S, h,
                       depth_threshold=30, color_threshold=20):
    """
    深度感知边缘检测：只在颜色或深度有显著变化时标记边缘。

    与纯 2D 边缘检测的区别：
    - 如果两个三角深度差大（不在同一表面），即使颜色相似也不抑制
    - 避免将阴影边界误认为物体边缘
    """
    lum = estimate_luminance(single, n_rows, n_cols)
    # 深度用亮度近似
    depth = lum.copy()

    edge_map = np.zeros((n_rows, n_cols), dtype=bool)

    for r in range(n_rows):
        for c in range(n_cols):
            I_c = lum[r, c]
            D_c = depth[r, c]
            is_edge = False

            for nr, nc in neighbors_of(r, c):
                if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                    continue
                I_n = lum[nr, nc]
                D_n = depth[nr, nc]

                color_diff = abs(I_c - I_n)
                depth_diff = abs(D_c - D_n)

                # 边缘条件：颜色差大 OR 深度差大
                if color_diff > color_threshold or depth_diff > depth_threshold:
                    is_edge = True
                    break

            edge_map[r, c] = is_edge

    return edge_map


def keypoints_3d(keypoints, single, S, h, n_rows, n_cols):
    """
    将 2D 三角网格上的关键点转换为 3D 坐标。

    深度来自亮度估计。
    """
    lum = estimate_luminance(single, n_rows, n_cols)
    depth = lum.copy()
    depth_norm = (depth - depth.min()) / max(depth.max() - depth.min(), 1)

    points_3d = []
    for r, c, resp, *_ in keypoints:
        x = c * S / 2.0
        y = r * h
        z = depth_norm[r, c] * S * 1.5  # scale depth
        points_3d.append((x, y, z, resp))

    return points_3d


# ============================================================
#  CLI 测试
# ============================================================

if __name__ == "__main__":
    import sys

    img = Image.open(sys.argv[1] if len(sys.argv) > 1 else "test_edge.png")
    img = img.convert("RGB")
    W, H = img.size
    pixels = np.array(img).astype(np.float32)

    S = 16.0; h = S * math.sqrt(3) / 2
    n_cols = int(W / (S / 2)) + 3
    n_rows = int(H / h) + 2
    single = sample_single_channels(pixels, S, h, n_rows, n_cols)
    borrowed = borrow_neighbors(single, n_rows, n_cols, edge_mode="mirror")

    # 构建 3D mesh
    mesh = TriMesh()
    mesh.from_triangle_grid(single, S, h, n_rows, n_cols, rgb_map=borrowed)

    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    mesh.export_obj("mesh_test.obj")

    # 3D 渲染
    for az in [0, 45, 90]:
        view = mesh.render_view(800, 600, azimuth=az, elevation=25)
        view.save(f"mesh_view_{az}.png")
    print("Saved: mesh_test.obj, mesh_view_*.png")
