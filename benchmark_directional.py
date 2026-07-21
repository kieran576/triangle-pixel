"""
补充基准: 定向边缘 8 测试图集
恢复论文 Table 1 旧版的 8 行测试集, 验证各向异性声明.
"""
import math, time, os, json
import numpy as np
from PIL import Image, ImageDraw

from triangle_sensor import TriangleSensor, compute_psnr, compute_ssim


def gen_edge(angle_deg, W=400, H=300, color_a=(220, 60, 60), color_b=(60, 100, 220)):
    """生成特定角度的两色边界"""
    img = Image.new("RGB", (W, H), color_a)
    draw = ImageDraw.Draw(img)
    rad = math.radians(angle_deg)
    # 边缘经过画布中心, 法线方向 (cos, sin), 长度 > diag
    cx, cy = W/2, H/2
    diag = math.sqrt(W*W + H*H) / 2
    # 边缘上两点
    x1 = cx - math.cos(rad + math.pi/2) * diag
    y1 = cy - math.sin(rad + math.pi/2) * diag
    x2 = cx + math.cos(rad + math.pi/2) * diag
    y2 = cy + math.sin(rad + math.pi/2) * diag
    # 一侧填充 color_b (沿法线方向位移 epsilon)
    nx, ny = math.cos(rad), math.sin(rad)
    # 把屏幕像素按与边缘的有符号距离分类
    arr = np.array(img).copy()
    Y, X = np.indices((H, W))
    signed = (X - cx) * nx + (Y - cy) * ny
    mask = signed > 0
    arr[mask] = color_b
    return Image.fromarray(arr)


def gen_color_boundary(W=400, H=300):
    """R -> B 水平渐变"""
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for x in range(W):
        t = x / (W - 1)
        arr[:, x] = (int(220*(1-t)), int(40+30*t), int(40+200*t))
    return Image.fromarray(arr)


def gen_gray_ramp(W=400, H=300):
    """线性灰阶"""
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    levels = np.linspace(20, 235, W).astype(np.uint8)
    arr[:] = levels[np.newaxis, :, np.newaxis]
    return Image.fromarray(arr)


def gen_siemens_star(W=400, H=300, n_spokes=36):
    """西门子星 (高空间频率)"""
    img = Image.new("RGB", (W, H), (110, 110, 110))
    draw = ImageDraw.Draw(img)
    cx, cy = W/2, H/2
    r_max = min(W, H) / 2 - 2
    for i in range(n_spokes):
        a1 = 2 * math.pi * i / n_spokes
        a2 = a1 + math.pi / n_spokes
        for r in range(2, int(r_max)):
            x1 = cx + r * math.cos(a1); y1 = cy + r * math.sin(a1)
            x2 = cx + (r+1) * math.cos(a1); y2 = cy + (r+1) * math.sin(a1)
            draw.line([(x1, y1), (x2, y2)], fill=(235, 235, 235), width=1)
            x1 = cx + r * math.cos(a2); y1 = cy + r * math.sin(a2)
            x2 = cx + (r+1) * math.cos(a2); y2 = cy + (r+1) * math.sin(a2)
            draw.line([(x1, y1), (x2, y2)], fill=(15, 15, 15), width=1)
    return img


def gen_real_photo_proxy(W=400, H=300):
    """无真实照片, 用复杂度代理图 (颜色块 + 渐变 + 圆 + 文字)"""
    img = Image.new("RGB", (W, H), (40, 80, 120))
    draw = ImageDraw.Draw(img)
    # 渐变背景
    arr = np.array(img).astype(np.float32)
    Y, X = np.indices((H, W))
    arr[..., 0] = np.clip(40 + 120 * (X / W), 0, 255)
    arr[..., 1] = np.clip(80 + 80 * (Y / H), 0, 255)
    arr[..., 2] = np.clip(120 - 60 * (X / W), 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    # 叠加结构
    draw.ellipse([60, 40, 200, 180], fill=(220, 200, 60))
    draw.rectangle([220, 60, 360, 220], fill=(80, 200, 100))
    draw.polygon([(120, 220), (260, 250), (300, 290)], fill=(180, 80, 80))
    return img


def generate_set(out_dir="bench_directional"):
    os.makedirs(out_dir, exist_ok=True)
    images = {}
    for ang in [0, 45, 90, 135]:
        name = f"edge_{ang}"
        im = gen_edge(ang)
        im.save(f"{out_dir}/{name}.png")
        images[name] = im
    im = gen_color_boundary(); im.save(f"{out_dir}/color_rb.png"); images["color_rb"] = im
    im = gen_gray_ramp(); im.save(f"{out_dir}/gray_ramp.png"); images["gray_ramp"] = im
    im = gen_siemens_star(); im.save(f"{out_dir}/siemens_star.png"); images["siemens_star"] = im
    im = gen_real_photo_proxy(); im.save(f"{out_dir}/real_proxy.png"); images["real_proxy"] = im
    print(f"Generated {len(images)} directional test images in {out_dir}/")
    return images


def run_directional_benchmark(images, sides=(8, 12), out_dir="bench_directional"):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for name, img in images.items():
        ref = np.array(img).astype(np.float32)
        for side in sides:
            sensor = TriangleSensor(triangle_side=side, psf_sigma=0.5, iso=100)
            t0 = time.time()
            r = sensor.compare_with_bayer(img)
            elapsed = time.time() - t0
            results.append({
                "image": name, "side": side,
                "tri_psnr": round(r['tri_psnr'], 2),
                "tri_ssim": round(r['tri_ssim'], 4),
                "bayer_psnr": round(r['bayer_psnr'], 2),
                "bayer_ssim": round(r['bayer_ssim'], 4),
                "delta_psnr": round(r['bayer_psnr'] - r['tri_psnr'], 2),
                "time_s": round(elapsed, 2),
            })
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  定向边缘 8 测试图集: 各向异性验证")
    print("=" * 60)

    images = generate_set()

    print("\n[1/2] Running S=8 benchmark (4% data)...")
    res_s8 = run_directional_benchmark(images, sides=(8,))
    print("\n[2/2] Running S=12 benchmark (2% data)...")
    res_s12 = run_directional_benchmark(images, sides=(12,))

    out = {"s8": res_s8, "s12": res_s12}
    with open("directional_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  方向各向异性 (S=8)")
    print("=" * 60)
    edge_psnrs = [r['tri_psnr'] for r in res_s8 if r['image'].startswith('edge_')]
    if edge_psnrs:
        anisotropy = max(edge_psnrs) - min(edge_psnrs)
        print(f"  Edge 0/45/90/135 PSNR: {edge_psnrs}")
        print(f"  Max - Min = {anisotropy:.2f} dB")
        print(f"  Claim: 2.1 dB -> measured: {anisotropy:.2f} dB", "OK" if abs(anisotropy - 2.1) < 1.0 else "DIFF")

    print("\n" + "=" * 60)
    print("  Full S=8 table")
    print("=" * 60)
    print(f"{'Image':18s} {'TRI PSNR':>10s} {'TRI SSIM':>10s} {'Bayer PSNR':>10s} {'Bayer SSIM':>10s}")
    for r in res_s8:
        print(f"{r['image']:18s} {r['tri_psnr']:>10.2f} {r['tri_ssim']:>10.4f} {r['bayer_psnr']:>10.2f} {r['bayer_ssim']:>10.4f}")