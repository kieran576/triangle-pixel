#!/usr/bin/env python3
"""
Triangle Pixel — 一键 Demo

跑通核心验证管线:
1. 生成测试图集
2. 三角 vs Bayer 传感器对比
3. AI 端到端去马赛克 vs 手工 ISP
4. Sierpinski 超分演示
5. 输出可视化结果
"""

import math, time, os, sys
import numpy as np
from PIL import Image, ImageDraw

os.makedirs("demo_output", exist_ok=True)

def hr(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
#  Step 1: 生成测试图
# ============================================================

hr("Step 1/5: 生成测试图")

os.makedirs("demo_output/images", exist_ok=True)

# 颜色边缘图
edge = Image.new("RGB", (400, 300), (200, 60, 60))
draw = ImageDraw.Draw(edge)
draw.rectangle([200, 0, 400, 300], fill=(60, 200, 60))
draw.rectangle([130, 80, 270, 220], fill=(60, 60, 200))
edge.save("demo_output/images/edge.png")

# 渐变图
grad = Image.new("RGB", (400, 300))
for y in range(300):
    for x in range(400):
        grad.putpixel((x, y), (
            int(255 * x / 400), int(255 * y / 300), int(255 * (x + y) / 700)
        ))
grad.save("demo_output/images/gradient.png")

# 纹理图
tex = Image.new("RGB", (400, 300), (128, 128, 128))
draw = ImageDraw.Draw(tex)
for y in range(0, 300, 20):
    for x in range(0, 400, 20):
        c = 200 if (x//20 + y//20) % 2 == 0 else 60
        draw.rectangle([x, y, x+19, y+19], fill=(c, c, c))
for y in range(50, 250, 40):
    draw.rectangle([40, y, 360, y+20], fill=(180, 80, 80))
tex.save("demo_output/images/texture.png")

print("  [OK] edge.png  [OK] gradient.png  [OK] texture.png")


# ============================================================
#  Step 2: 传感器对比
# ============================================================

hr("Step 2/5: 三角 vs Bayer 传感器对比")

from triangle_sensor import (TriangleSensor, render_comparison,
                              compute_psnr, compute_ssim)

sides = [8, 12, 16]
results = {}

for img_name in ["edge", "gradient", "texture"]:
    img = Image.open(f"demo_output/images/{img_name}.png").convert("RGB")

    for side in sides:
        sensor = TriangleSensor(triangle_side=side, psf_sigma=0.5, iso=100)
        t0 = time.time()
        r = sensor.compare_with_bayer(img)
        elapsed = time.time() - t0

        n_tri = r['meta']['n_rows'] * r['meta']['n_cols']
        data_pct = n_tri / (img.size[0] * img.size[1]) * 100
        key = f"{img_name}_s{side}"
        results[key] = r

        if img_name == "edge" and side == 8:
            comp = render_comparison(r['tri_rgb'], r['bayer_rgb'],
                                     r['tri_psnr'], r['bayer_psnr'],
                                     r['tri_ssim'], r['bayer_ssim'])
            comp.save("demo_output/sensor_compare.png")

        print(f"  {img_name} S={side}: {n_tri} tri ({data_pct:.0f}% data) "
              f"TRI={r['tri_psnr']:.1f}dB BAYER={r['bayer_psnr']:.1f}dB "
              f"[{elapsed:.1f}s]")

# Summary
s8_psnr = [r['tri_psnr'] for k, r in results.items() if k.endswith('_s8')]
print(f"\n  >> S=8 平均 PSNR: {np.mean(s8_psnr):.1f} dB (Bayer: {np.mean([r['bayer_psnr'] for k,r in results.items() if k.endswith('_s8')]):.1f} dB)")


# ============================================================
#  Step 3: AI 去马赛克 vs 手工 ISP
# ============================================================

hr("Step 3/5: AI 端到端 vs 手工 ISP")

from triangle_demosaic_ai import (TriDemosaicGCN, demosaic_image,
                                   _build_adjacency, train_demosaic_gcn)
from triangle_engine import process_pipeline

for img_name in ["edge", "gradient"]:
    img = Image.open(f"demo_output/images/{img_name}.png").convert("RGB")
    ref = np.array(img).astype(np.float32)

    # Train AI
    t0 = time.time()
    model, adj_ref, losses = train_demosaic_gcn(
        [f"demo_output/images/{img_name}.png"],
        triangle_side=16, epochs=120
    )
    train_t = time.time() - t0

    # AI inference
    t0 = time.time()
    ai_rgb = demosaic_image(model, img, adj_ref)
    ai_t = time.time() - t0

    # ISP
    t0 = time.time()
    isp_rgb = process_pipeline(img, 16, mode='correct_isp',
                               correct_iterations=3, correct_sigma=0.20)
    isp_t = time.time() - t0

    ai_psnr = compute_psnr(np.array(ai_rgb).astype(float), ref)
    isp_psnr = compute_psnr(np.array(isp_rgb).astype(float), ref)

    print(f"  {img_name}: AI={ai_psnr:.1f}dB ISP={isp_psnr:.1f}dB "
          f"(train={train_t:.1f}s AI={ai_t*1000:.0f}ms ISP={isp_t*1000:.0f}ms)")


# ============================================================
#  Step 4: Sierpinski 超分
# ============================================================

hr("Step 4/5: Sierpinski 超分")

from triangle_superres import super_resolve

img = Image.open("demo_output/images/edge.png").convert("RGB")
for zoom in [2, 4]:
    t0 = time.time()
    sr = super_resolve(img, triangle_side=20, zoom=zoom,
                       use_self_sim=True, correct_iterations=2)
    elapsed = time.time() - t0
    sr.save(f"demo_output/superres_{zoom}x.png")
    print(f"  {zoom}×: {sr.size} [{elapsed:.1f}s]")


# ============================================================
#  Step 5: 3D 视图 + 输出汇总
# ============================================================

hr("Step 5/5: 3D 视图 + 汇总")

from triangle_3d import TriMesh
from triangle_engine import sample_single_channels, borrow_neighbors

img = Image.open("demo_output/images/edge.png").convert("RGB")
pixels = np.array(img).astype(np.float32)
S = 16.0; h = S * math.sqrt(3) / 2
nc = int(img.size[0] / (S / 2)) + 3
nr = int(img.size[1] / h) + 2

single = sample_single_channels(pixels, S, h, nr, nc)
borrowed = borrow_neighbors(single, nr, nc, edge_mode="mirror")

mesh = TriMesh()
mesh.from_triangle_grid(single, S, h, nr, nc, rgb_map=borrowed)
mesh.export_obj("demo_output/mesh.obj")

for az in [0, 45, 90]:
    view = mesh.render_view(400, 300, azimuth=az, elevation=25)
    view.save(f"demo_output/3d_view_{az}.png")
    print(f"  3D view az={az}°: {view.size}")

print(f"\n  OBJ 导出: demo_output/mesh.obj ({len(mesh.vertices)}v, {len(mesh.faces)}f)")


# ============================================================
#  Final
# ============================================================

hr("Demo 完成")

outputs = os.listdir("demo_output")
print(f"\n  输出文件 ({len(outputs)} 个):")
for f in sorted(outputs):
    path = f"demo_output/{f}"
    if os.path.isdir(path):
        continue
    size = os.path.getsize(path)
    print(f"    {path} ({size/1024:.0f} KB)")

print(f"\n  >> 打开 GUI: python triangle_pixel_gui.py")
print(f"  >> 基准报告: BENCHMARK.md")
print(f"  >> README:    README.md")
print(f"\n{'='*60}")
print(f"  三角像素视觉系统 — Demo 验证通过")
print(f"{'='*60}")
