#!/usr/bin/env python3
"""
三角传感器模拟器

模拟三角 CMOS 传感器的完整物理捕获过程，
并与传统 Bayer 传感器对比。

管线:
  高分辨率场景 → 光学模糊 → 三角采样 → 噪声 → 量化 → TRI RAW
                → 光学模糊 → Bayer采样 → 噪声 → 量化 → BAYER RAW

重建:
  TRI RAW → borrow + ISP → TRI RGB
  BAYER RAW → bilinear demosaic → BAYER RGB

对比:
  PSNR / SSIM / 差异图
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from triangle_engine import (
    sample_single_channels, borrow_neighbors,
    correct_triangular_isp, render_triangles,
    assigned_channel,
)


# ============================================================
#  光学模型
# ============================================================

def optical_blur(image, psf_sigma=0.7):
    """
    模拟镜头光学模糊。

    psf_sigma: PSF 高斯核标准差 (像素)。
               0.5-0.7 = 锐利镜头，1.0-1.5 = 柔和镜头。
    """
    if psf_sigma <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=psf_sigma))


# ============================================================
#  噪声模型
# ============================================================

def add_sensor_noise(raw, base_iso=100, read_noise_e=3.0, dark_current_e=0.5):
    """
    模拟 CMOS 传感器噪声。

    Args:
        raw: 单通道 RAW 数据 [0-255]
        base_iso: 基础 ISO
        read_noise_e: 读出噪声 (电子数)
        dark_current_e: 暗电流 (电子数/秒)

    噪声模型:
    - 光子散粒噪声: Poisson，方差 ∝ 信号
    - 读出噪声: Gaussian，固定方差
    - 暗电流: 固定偏移 + Poisson 噪声

    Returns:
        noisy: 带噪声的 RAW [0-255]
    """
    # ISO=0: 无噪声, 直接返回
    if base_iso <= 0:
        return raw.copy()

    # 转换为电子数 (假设满阱容量 ~5000e- at ISO 100，线性映射)
    full_well = 5000.0 * (100.0 / base_iso)
    electrons = raw / 255.0 * full_well

    # 光子散粒噪声
    electrons = np.maximum(electrons, 0)
    electrons = np.random.poisson(np.maximum(electrons, 0))

    # 暗电流
    dark = np.random.poisson(dark_current_e, size=electrons.shape).astype(np.float32)
    electrons = electrons.astype(np.float32) + dark

    # 读出噪声
    electrons += np.random.randn(*electrons.shape) * read_noise_e

    # 转回 [0-255]
    electrons = np.maximum(electrons, 0)
    raw_noisy = electrons / full_well * 255.0
    return np.clip(raw_noisy, 0, 255)


# ============================================================
#  Bayer 传感器
# ============================================================

def sample_bayer(pixels, pattern_offset=0):
    """
    从 RGB 图生成 Bayer RAW。

    Bayer 模式 (pattern_offset=0):
      R  G  R  G ...
      G  B  G  B ...
      R  G  R  G ...

    Returns:
        bayer: [H, W] 单通道 RAW (每个像素一个值)
        bayer_mask: [H, W, 3] 颜色掩码 (1=该像素有此通道)
    """
    H, W = pixels.shape[:2]
    bayer = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W, 3), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            if (y + pattern_offset) % 2 == 0:  # even row: R, G, R, G
                if (x + pattern_offset) % 2 == 0:
                    bayer[y, x] = pixels[y, x, 0]  # R
                    mask[y, x, 0] = 1
                else:
                    bayer[y, x] = pixels[y, x, 1]  # G
                    mask[y, x, 1] = 1
            else:  # odd row: G, B, G, B
                if (x + pattern_offset) % 2 == 0:
                    bayer[y, x] = pixels[y, x, 1]  # G
                    mask[y, x, 1] = 1
                else:
                    bayer[y, x] = pixels[y, x, 2]  # B
                    mask[y, x, 2] = 1

    return bayer, mask


def demosaic_bayer_bilinear(bayer, mask):
    """
    Bayer RAW → RGB (双线性 demosaic)。

    对每个缺失通道，用最近邻的 2-4 个已知像素的均值填充。
    """
    H, W = bayer.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    for ch in range(3):
        # 对每个通道，已知位置保持不变，缺失位置插值
        known = mask[:, :, ch] > 0
        rgb[:, :, ch][known] = bayer[known]

        # 双线性: 对缺失位置，取上下左右最近已知像素的均值
        # 简化：用 3×3 窗口内的已知像素均值
        for y in range(H):
            for x in range(W):
                if not known[y, x]:
                    y0 = max(0, y - 1)
                    y1 = min(H, y + 2)
                    x0 = max(0, x - 1)
                    x1 = min(W, x + 2)
                    window = mask[y0:y1, x0:x1, ch]
                    if window.sum() > 0:
                        rgb[y, x, ch] = (bayer[y0:y1, x0:x1][window > 0]).mean()
                    # else: keep 0 (edge case)

    return np.clip(rgb, 0, 255).astype(np.uint8)


# ============================================================
#  三角传感器主类
# ============================================================

class TriangleSensor:
    """三角 CMOS 传感器模拟器"""

    def __init__(self, triangle_side=12, psf_sigma=0.6,
                 iso=100, read_noise=3.0, dark_current=0.5):
        """
        Args:
            triangle_side: 三角边长 (像素)
            psf_sigma: 光学模糊强度
            iso: ISO 感光度
            read_noise: 读出噪声 (电子)
            dark_current: 暗电流 (电子)
        """
        self.triangle_side = triangle_side
        self.psf_sigma = psf_sigma
        self.iso = iso
        self.read_noise = read_noise
        self.dark_current = dark_current

    def capture(self, scene, progress_callback=None):
        """
        模拟完整捕获过程。

        Args:
            scene: PIL Image — "连续"场景 (高分辨率)

        Returns:
            tri_raw: [n_rows, n_cols] 三角 RAW
            metadata: {S, h, n_rows, n_cols, W, H}
        """
        W, H = scene.size

        def prog(pct, msg=""):
            if progress_callback:
                progress_callback(pct)

        # 1. 光学模糊
        prog(5, "光学模糊...")
        blurred = optical_blur(scene, self.psf_sigma)
        pixels = np.array(blurred).astype(np.float32)

        # 2. 三角采样
        prog(15, "三角采样...")
        S = float(self.triangle_side)
        h = S * math.sqrt(3) / 2.0
        n_cols = int(W / (S / 2.0)) + 3
        n_rows = int(H / h) + 2

        raw = sample_single_channels(pixels, S, h, n_rows, n_cols, sample_radius=1.0)

        # 3. 噪声
        prog(40, "噪声模拟...")
        raw = add_sensor_noise(raw, self.iso, self.read_noise, self.dark_current)

        # 4. 量化 (8-bit) — 已隐含在 [0-255] 范围中
        raw = np.clip(raw, 0, 255)

        prog(100)
        meta = {"S": S, "h": h, "n_rows": n_rows, "n_cols": n_cols, "W": W, "H": H}
        return raw, meta

    def reconstruct(self, tri_raw, meta, isp_iters=3):
        """
        从三角 RAW 重建 RGB。

        Returns:
            PIL Image (RGB)
        """
        nr, nc = meta["n_rows"], meta["n_cols"]
        S, h = meta["S"], meta["h"]
        W, H = meta["W"], meta["H"]

        borrowed = borrow_neighbors(tri_raw, nr, nc, edge_mode="mirror")
        corrected = correct_triangular_isp(
            tri_raw, borrowed, nr, nc,
            iterations=isp_iters, edge_sensitivity=40,
        )
        return render_triangles(corrected, S, h, nr, nc, W, H)

    def compare_with_bayer(self, scene, progress_callback=None):
        """
        完整对比: 三角 vs Bayer。

        Returns:
            dict with keys:
                tri_rgb, bayer_rgb: PIL Images
                tri_psnr, bayer_psnr: float
                tri_ssim, bayer_ssim: float
        """
        def prog(pct):
            if progress_callback:
                progress_callback(pct)

        W, H = scene.size

        # --- 三角传感器 ---
        prog(5)
        tri_raw, meta = self.capture(scene)
        prog(35)
        tri_rgb = self.reconstruct(tri_raw, meta)
        prog(50)

        # --- Bayer 传感器 ---
        blurred = optical_blur(scene, self.psf_sigma)
        pixels = np.array(blurred).astype(np.float32)

        bayer, mask = sample_bayer(pixels)
        bayer = add_sensor_noise(bayer, self.iso, self.read_noise, self.dark_current)
        prog(70)
        bayer_rgb = demosaic_bayer_bilinear(bayer, mask)
        prog(85)

        # --- 质量对比 ---
        ref = np.array(blurred).astype(np.float32)  # 参考 = 模糊后的场景

        tri_arr = np.array(tri_rgb).astype(np.float32)
        bayer_arr = bayer_rgb.astype(np.float32)

        tri_psnr = compute_psnr(tri_arr, ref)
        bayer_psnr = compute_psnr(bayer_arr, ref)

        tri_ssim = compute_ssim(tri_arr, ref)
        bayer_ssim = compute_ssim(bayer_arr, ref)

        prog(100)

        return {
            "tri_rgb": tri_rgb,
            "bayer_rgb": Image.fromarray(bayer_rgb),
            "tri_psnr": tri_psnr,
            "bayer_psnr": bayer_psnr,
            "tri_ssim": tri_ssim,
            "bayer_ssim": bayer_ssim,
            "meta": meta,
        }


# ============================================================
#  质量指标
# ============================================================

def compute_psnr(img, ref):
    """PSNR (Peak Signal-to-Noise Ratio)"""
    mse = np.mean((img - ref) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(img, ref, k1=0.01, k2=0.03, win_size=11):
    """SSIM (Structural Similarity) — Wang et al. 2004 局部窗口版.

    用 11×11 高斯加权滑动窗口 (sigma=1.5) 计算每像素 SSIM,
    然后取整图均值. 支持单通道 (H,W) 或三通道 (H,W,3) 输入.

    与旧"全局亮度/方差"近似版的区别:
      - 旧版只看整图均值/方差, 不区分空间结构
      - 新版对每个像素独立评估局部结构相似度, 更接近人眼感知

    依赖: skimage (skimage.metrics.structural_similarity).
    若 skimage 不可用, 自动降级到 8×8 均匀窗口 (仍非全局版).

    返回: 标量 float, 范围 [-1, 1], 越高越好.
    """
    try:
        from skimage.metrics import structural_similarity as _ssim
        channel_axis = 2 if img.ndim == 3 else None
        val = _ssim(
            ref.astype(np.float64), img.astype(np.float64),
            data_range=255.0,
            win_size=win_size,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            channel_axis=channel_axis,
        )
        return float(val)
    except ImportError:
        # 降级: 用 8×8 均匀窗口手算 (避免与旧版完全相同)
        from scipy.ndimage import uniform_filter
        a = img.astype(np.float64)
        b = ref.astype(np.float64)
        c1 = (k1 * 255) ** 2
        c2 = (k2 * 255) ** 2
        mu_a = uniform_filter(a, size=8)
        mu_b = uniform_filter(b, size=8)
        mu_a2 = mu_a * mu_a
        mu_b2 = mu_b * mu_b
        mu_ab = mu_a * mu_b
        sigma_a2 = uniform_filter(a * a, size=8) - mu_a2
        sigma_b2 = uniform_filter(b * b, size=8) - mu_b2
        sigma_ab = uniform_filter(a * b, size=8) - mu_ab
        ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / \
                   ((mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2))
        return float(ssim_map.mean())


# ============================================================
#  对比可视化
# ============================================================

def render_comparison(tri_rgb, bayer_rgb, tri_psnr, bayer_psnr,
                      tri_ssim, bayer_ssim):
    """生成三角 vs Bayer 对比图"""
    W = tri_rgb.size[0] * 2 + 40
    H = tri_rgb.size[1] + 60

    result = Image.new("RGB", (W, H), (30, 30, 30))
    result.paste(tri_rgb, (10, 50))
    result.paste(bayer_rgb, (tri_rgb.size[0] + 30, 50))

    draw = ImageDraw.Draw(result)
    draw.text((20, 10),
              f"TRI Sensor — PSNR={tri_psnr:.1f}dB  SSIM={tri_ssim:.3f}",
              fill=(100, 255, 100))
    draw.text((tri_rgb.size[0] + 40, 10),
              f"BAYER Sensor — PSNR={bayer_psnr:.1f}dB  SSIM={bayer_ssim:.3f}",
              fill=(255, 255, 100))

    # 胜出标记
    if tri_psnr > bayer_psnr:
        draw.text((W // 2 - 40, 30), "WINNER", fill=(0, 255, 0))
    else:
        draw.text((W // 2 - 40, 30), "WINNER", fill=(255, 255, 0))

    return result


# ============================================================
#  CLI 测试
# ============================================================

if __name__ == "__main__":
    import sys, time

    scene_path = sys.argv[1] if len(sys.argv) > 1 else "test_edge.png"
    scene = Image.open(scene_path).convert("RGB")
    print(f"Scene: {scene.size}")

    sensor = TriangleSensor(triangle_side=16, psf_sigma=0.6, iso=100)
    print(f"Sensor: side={sensor.triangle_side} PSF={sensor.psf_sigma} ISO={sensor.iso}")

    t0 = time.time()
    result = sensor.compare_with_bayer(scene)
    elapsed = time.time() - t0

    print(f"\n=== 对比结果 ({elapsed:.1f}s) ===")
    print(f"  TRI:   PSNR={result['tri_psnr']:.1f}dB  SSIM={result['tri_ssim']:.3f}")
    print(f"  BAYER: PSNR={result['bayer_psnr']:.1f}dB  SSIM={result['bayer_ssim']:.3f}")
    delta = result['tri_psnr'] - result['bayer_psnr']
    print(f"  Delta: {delta:+.1f}dB  {'TRI wins!' if delta > 0 else 'BAYER wins'}")

    comparison = render_comparison(
        result['tri_rgb'], result['bayer_rgb'],
        result['tri_psnr'], result['bayer_psnr'],
        result['tri_ssim'], result['bayer_ssim'],
    )
    comparison.save("sensor_compare.png")
    print("\nSaved: sensor_compare.png")
