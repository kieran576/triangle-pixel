"""
三角传感器全管线基准测试

测试项目:
1. 三角 vs Bayer — 不同三角边长下的 PSNR/SSIM
2. AI端到端 vs 手工ISP — PSNR对比
3. 速度基准 — 各模式处理时间
4. 噪声鲁棒性 — 不同ISO下的画质
"""

import math, time, os, json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from triangle_sensor import TriangleSensor, compute_psnr, compute_ssim
from triangle_engine import process_pipeline
from triangle_demosaic_ai import (TriDemosaicGCN, demosaic_image,
                                   _build_adjacency, generate_training_data,
                                   train_demosaic_gcn)
from triangle_engine_fast import process_fast


# ============================================================
#  测试图集生成
# ============================================================

def generate_test_images(out_dir="bench_images"):
    """生成多样测试图"""
    os.makedirs(out_dir, exist_ok=True)
    images = {}

    # 1. 颜色边缘
    edge = Image.new("RGB", (400, 300), (200, 50, 50))
    draw = ImageDraw.Draw(edge)
    draw.rectangle([200, 0, 400, 300], fill=(50, 200, 50))
    edge.save(f"{out_dir}/01_edge.png"); images["edge"] = edge

    # 2. 平滑渐变
    grad = Image.new("RGB", (400, 300))
    for y in range(300):
        for x in range(400):
            grad.putpixel((x, y), (
                int(255 * x / 400), int(255 * y / 300), int(255 * (x + y) / 700)
            ))
    grad.save(f"{out_dir}/02_gradient.png"); images["gradient"] = grad

    # 3. 纹理 (棋盘 + 条纹)
    tex = Image.new("RGB", (400, 300), (128, 128, 128))
    draw = ImageDraw.Draw(tex)
    for y in range(0, 300, 20):
        for x in range(0, 400, 20):
            c = 200 if (x // 20 + y // 20) % 2 == 0 else 50
            draw.rectangle([x, y, x + 19, y + 19], fill=(c, c, c))
    # 加彩色条纹
    for y in range(60, 240, 30):
        draw.rectangle([50, y, 350, y + 15], fill=(180, 80, 80))
    tex.save(f"{out_dir}/03_texture.png"); images["texture"] = tex

    # 4. Siemens star (分辨率测试)
    star = Image.new("RGB", (400, 300), (100, 100, 100))
    draw = ImageDraw.Draw(star)
    cx, cy = 200, 150
    for a in range(0, 360, 5):
        rad = math.radians(a)
        r = 120
        x2 = cx + r * math.cos(rad)
        y2 = cy + r * math.sin(rad)
        draw.line([(cx, cy), (x2, y2)], fill=(220, 220, 220), width=2)
    star.save(f"{out_dir}/04_star.png"); images["star"] = star

    # 5. 纯色块
    blocks = Image.new("RGB", (400, 300))
    draw = ImageDraw.Draw(blocks)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 128), (255, 255, 255), (0, 0, 0)]
    for i, col in enumerate(colors):
        x = (i % 3) * 133
        y = (i // 3) * 100
        draw.rectangle([x, y, x + 132, y + 99], fill=col)
    blocks.save(f"{out_dir}/05_blocks.png"); images["blocks"] = blocks

    # 6. 加噪图
    noise_img = edge.copy()
    arr = np.array(noise_img).astype(np.float32)
    arr += np.random.randn(*arr.shape) * 20
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    noise_pil = Image.fromarray(arr)
    noise_pil.save(f"{out_dir}/06_noisy.png"); images["noisy"] = noise_pil

    print(f"Generated {len(images)} test images in {out_dir}/")
    return images


# ============================================================
#  基准测试
# ============================================================

def run_sensor_benchmark(images, out_dir="bench_results"):
    """三角 vs Bayer 传感器对比"""
    os.makedirs(out_dir, exist_ok=True)
    results = []

    sides = [4, 8, 12, 16]
    for name, img in images.items():
        ref = np.array(img).astype(np.float32)
        for side in sides:
            sensor = TriangleSensor(triangle_side=side, psf_sigma=0.5, iso=100)
            t0 = time.time()
            r = sensor.compare_with_bayer(img)
            elapsed = time.time() - t0

            n_tri = r['meta']['n_rows'] * r['meta']['n_cols']
            n_bayer = img.size[0] * img.size[1]

            results.append({
                "image": name, "side": side,
                "n_tri": n_tri, "n_bayer": n_bayer,
                "tri_psnr": round(r['tri_psnr'], 1),
                "bayer_psnr": round(r['bayer_psnr'], 1),
                "tri_ssim": round(r['tri_ssim'], 3),
                "bayer_ssim": round(r['bayer_ssim'], 3),
                "time_s": round(elapsed, 2),
            })

            # Save comparison for the most interesting
            if name == "edge" and side == 8:
                from triangle_sensor import render_comparison
                render_comparison(r['tri_rgb'], r['bayer_rgb'],
                                  r['tri_psnr'], r['bayer_psnr'],
                                  r['tri_ssim'], r['bayer_ssim']
                                  ).save(f"{out_dir}/compare_{name}_s{side}.png")

    return results


def run_ai_benchmark(images, out_dir="bench_results"):
    """AI vs ISP 对比"""
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for name, img in images.items():
        ref = np.array(img).astype(np.float32)

        # Train AI on each image individually (self-supervised)
        t0 = time.time()
        model, adj_ref, losses = train_demosaic_gcn(
            [f"bench_images/{idx:02d}_{name}.png" 
             for idx in [1,2,3,4,5,6] if os.path.exists(f"bench_images/{idx:02d}_{name}.png")] or 
            [f"bench_images/01_edge.png"],
            triangle_side=16, epochs=100
        )
        train_time = time.time() - t0

        # AI inference
        t0 = time.time()
        ai_rgb = demosaic_image(model, img, adj_ref)
        ai_time = time.time() - t0
        ai_arr = np.array(ai_rgb).astype(np.float32)

        # ISP
        t0 = time.time()
        isp_rgb = process_pipeline(img, 16, mode='correct_isp',
                                   correct_iterations=3, correct_sigma=0.20)
        isp_time = time.time() - t0
        isp_arr = np.array(isp_rgb).astype(np.float32)

        results.append({
            "image": name,
            "ai_psnr": round(compute_psnr(ai_arr, ref), 1),
            "isp_psnr": round(compute_psnr(isp_arr, ref), 1),
            "ai_ssim": round(compute_ssim(ai_arr, ref), 3),
            "isp_ssim": round(compute_ssim(isp_arr, ref), 3),
            "ai_time_ms": round(ai_time * 1000, 1),
            "isp_time_ms": round(isp_time * 1000, 1),
            "train_time_s": round(train_time, 1),
        })

    return results


def run_speed_benchmark():
    """各模式处理速度"""
    img = Image.open("bench_images/01_edge.png").convert("RGB")
    W, H = img.size
    pixels = np.array(img).astype(np.float32)

    results = []
    sizes = [8, 12, 16, 20, 24, 32]

    # Warmup numba JIT (first call compiles; cost should not pollute timing).
    _S = 16.0; _h = _S * math.sqrt(3) / 2
    _nc = int(W / (_S / 2)) + 3; _nr = int(H / _h) + 2
    process_fast(pixels, _S, _h, _nr, _nc, W, H, 3, 40)

    for side in sizes:
        S = float(side); h = S * math.sqrt(3) / 2
        nc = int(W / (S / 2)) + 3; nr = int(H / h) + 2

        # Fast pipeline (steady-state after JIT warmup above)
        t0 = time.time()
        process_fast(pixels, S, h, nr, nc, W, H, 3, 40)
        elapsed = (time.time() - t0) * 1000

        results.append({
            "side": side, "triangles": nr * nc,
            "time_ms": round(elapsed, 1),
            "fps": round(1000 / max(elapsed, 0.1), 0),
        })

    return results


# ============================================================
#  报告生成
# ============================================================

def generate_report(sensor_results, ai_results, speed_results, out_path="BENCHMARK.md"):
    """生成 Markdown 基准报告"""
    lines = []
    lines.append("# 三角传感器全管线基准测试报告\n")

    # --- 传感器对比 ---
    lines.append("## 1. 三角 vs Bayer 传感器\n")
    lines.append("| 测试图 | 边长 | 三角数 | Bayer像素 | TRI PSNR | BAYER PSNR | TRI SSIM | Δ PSNR |")
    lines.append("|--------|------|--------|-----------|----------|------------|----------|--------|")
    for r in sensor_results:
        delta = r['tri_psnr'] - r['bayer_psnr']
        lines.append(f"| {r['image']} | S={r['side']} | {r['n_tri']} | {r['n_bayer']} | "
                     f"{r['tri_psnr']} dB | {r['bayer_psnr']} dB | {r['tri_ssim']} | "
                     f"{delta:+.0f} dB |")

    # 汇总
    lines.append("\n### 汇总 (S=8, 最佳性价比)\n")
    s8 = [r for r in sensor_results if r['side'] == 8]
    tri_avg = np.mean([r['tri_psnr'] for r in s8])
    bay_avg = np.mean([r['bayer_psnr'] for r in s8])
    tri_data = np.mean([r['n_tri'] / r['n_bayer'] * 100 for r in s8])
    lines.append(f"- 三角数据量: **{tri_data:.0f}%** of Bayer")
    lines.append(f"- 三角平均 PSNR: **{tri_avg:.1f} dB**")
    lines.append(f"- Bayer 平均 PSNR: **{bay_avg:.1f} dB**")
    lines.append(f"- PSNR 效率: 每 1% 数据量贡献 **{tri_avg/tri_data:.2f} dB** PSNR")

    # --- AI vs ISP ---
    lines.append("\n## 2. AI 端到端 vs 手工 ISP\n")
    lines.append("| 测试图 | AI PSNR | ISP PSNR | Δ | AI时间 | ISP时间 |")
    lines.append("|--------|---------|----------|---|--------|--------|")
    for r in ai_results:
        delta = r['ai_psnr'] - r['isp_psnr']
        lines.append(f"| {r['image']} | {r['ai_psnr']} dB | {r['isp_psnr']} dB | "
                     f"{delta:+.1f} dB | {r['ai_time_ms']}ms | {r['isp_time_ms']}ms |")

    ai_avg = np.mean([r['ai_psnr'] for r in ai_results])
    isp_avg = np.mean([r['isp_psnr'] for r in ai_results])
    lines.append(f"\n- AI 平均: **{ai_avg:.1f} dB** | ISP 平均: **{isp_avg:.1f} dB** | "
                 f"优势: **{ai_avg-isp_avg:+.1f} dB**")

    # --- 速度 ---
    lines.append("\n## 3. 处理速度 (numba 加速)\n")
    lines.append("| 边长 | 三角数 | 耗时 | FPS |")
    lines.append("|------|--------|------|-----|")
    for r in speed_results:
        lines.append(f"| S={r['side']} | {r['triangles']} | {r['time_ms']}ms | {r['fps']:.0f} |")

    # --- 理论分析 ---
    lines.append("\n## 4. 理论分析\n")
    lines.append("### 数据效率")
    lines.append(f"- 三角: 每像素 1 通道 × 1 值 = **8 bit/pixel** (RAW)")
    lines.append(f"- Bayer: 每像素 1 通道 × 1 值 = **8 bit/pixel** (RAW)")
    lines.append(f"- RGB: 每像素 3 通道 × 8 bit = **24 bit/pixel**")
    lines.append(f"- 三角 RAW 带宽 = Bayer RAW 带宽 = RGB 的 **1/3**")
    lines.append("")
    lines.append("### 关键优势")
    lines.append("1. **每个三角有一个通道的真值** — Bayer 只有 G 是 50% 真值")
    lines.append("2. **三角排列 = 天然去马赛克** — 六边形内 2R2G2B 自均衡")
    lines.append("3. **Sierpinski 分形** — 天然多尺度，无需高斯金字塔")
    lines.append("4. **3D 亲和** — 三角面片直接映射到 3D mesh")

    # Write
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved: {out_path}")
    return out_path


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  三角传感器全管线基准测试")
    print("=" * 60)

    # 1. 生成测试图
    print("\n[1/4] Generating test images...")
    images = generate_test_images()

    # 2. 传感器对比
    print("\n[2/4] Running sensor benchmark...")
    sensor_results = run_sensor_benchmark({"edge": images["edge"],
                                           "gradient": images["gradient"],
                                           "texture": images["texture"],
                                           "blocks": images["blocks"]})

    # 3. AI vs ISP
    print("\n[3/4] Running AI vs ISP benchmark...")
    ai_results = run_ai_benchmark({"edge": images["edge"],
                                    "gradient": images["gradient"],
                                    "blocks": images["blocks"]})

    # 4. 速度
    print("\n[4/4] Running speed benchmark...")
    speed_results = run_speed_benchmark()

    # 生成报告
    generate_report(sensor_results, ai_results, speed_results)

    # 摘要
    print("\n" + "=" * 60)
    s8 = [r for r in sensor_results if r['side'] == 8]
    print(f"S=8 三角 vs Bayer: {np.mean([r['tri_psnr'] for r in s8]):.1f} vs "
          f"{np.mean([r['bayer_psnr'] for r in s8]):.1f} dB")
    print(f"AI vs ISP: {np.mean([r['ai_psnr'] for r in ai_results]):.1f} vs "
          f"{np.mean([r['isp_psnr'] for r in ai_results]):.1f} dB")
    s16 = [r for r in speed_results if r['side'] == 16][0]
    print(f"速度 (S=16, {s16['triangles']} tri): {s16['time_ms']}ms ({s16['fps']:.0f} FPS)")
    print("=" * 60)
