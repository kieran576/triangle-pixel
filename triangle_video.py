"""
时序一致性分析: 三角管线在视频流上的帧间稳定性

研究方向: 零延迟感知要求每帧之间没有闪烁/抖动.
- 静态场景: 帧间差应趋近 0 (只有传感器噪声)
- 动态场景: 帧间差应反映真实运动, 不引入 ISP 伪影

使用现有 TriangleSensor 模拟一个可控的视频:
- 100 帧静态场景 (测试传感器噪声下限)
- 100 帧平移场景 (测试 ISP 对运动边缘的处理)
"""
import os, math
import numpy as np
from PIL import Image

from triangle_sensor import TriangleSensor, optical_blur
from triangle_engine import process_pipeline


def make_static_scene(W=400, H=300, t=0):
    """静态彩色测试图 (颜色块 + 圆 + 文字)"""
    img = Image.new('RGB', (W, H), (40, 80, 120))
    arr = np.array(img).astype(np.float32)
    Y, X = np.indices((H, W))
    arr[..., 0] = np.clip(40 + 120 * (X / W), 0, 255)
    arr[..., 1] = np.clip(80 + 80 * (Y / H), 0, 255)
    arr[..., 2] = np.clip(120 - 60 * (X / W), 0, 255)
    arr[60:160, 30:130] = (220, 200, 60)
    arr[80:220, 180:340] = (80, 200, 100)
    return Image.fromarray(arr.astype(np.uint8))


def make_translating_scene(W=400, H=300, t=0):
    """水平匀速平移测试图"""
    base = make_static_scene(W, H)
    arr = np.array(base)
    shift = int(20 * (t % 50) / 50) % (W // 2)
    shifted = np.zeros_like(arr)
    shifted[:, :W-shift] = arr[:, shift:]
    return Image.fromarray(shifted)


def compute_frame_difference(img_a, img_b):
    """L1 + L_inf 帧间差"""
    a = np.array(img_a).astype(np.float32)
    b = np.array(img_b).astype(np.float32)
    diff = np.abs(a - b)
    return {
        'mean': float(diff.mean()),
        'max': float(diff.max()),
        'p95': float(np.percentile(diff, 95)),
    }


def evaluate_video(scenes, name, side=12, psf_sigma=0.7, iso=400, mode="correct"):
    """对一序列场景跑三角管线, 评估时序一致性"""
    sensor = TriangleSensor(triangle_side=side, psf_sigma=psf_sigma, iso=iso)
    frames_tri = []
    frames_raw = []
    diffs_tri = []
    diffs_raw = []

    prev_tri = None
    prev_raw = None
    for i, img in enumerate(scenes):
        # Apply optical blur before pipeline
        blurred = optical_blur(img, psf_sigma=psf_sigma)
        # Triangle pipeline
        tri = process_pipeline(blurred,
                                triangle_side=side, mode=mode)
        frames_tri.append(tri)
        if prev_tri is not None:
            d = compute_frame_difference(prev_tri, tri)
            diffs_tri.append(d)
        prev_tri = tri

    mean_diff = np.mean([d['mean'] for d in diffs_tri]) if diffs_tri else 0
    max_diff = np.max([d['max'] for d in diffs_tri]) if diffs_tri else 0
    p95_diff = np.mean([d['p95'] for d in diffs_tri]) if diffs_tri else 0

    return {
        'name': name, 'n_frames': len(scenes), 'side': side,
        'mode': mode, 'iso': iso, 'psf_sigma': psf_sigma,
        'frame_diff_mean': round(mean_diff, 3),
        'frame_diff_p95': round(p95_diff, 3),
        'frame_diff_max': round(max_diff, 3),
    }


def run_temporal_benchmark(out_path='temporal_benchmark.json'):
    print("=" * 60)
    print("  时序一致性基准")
    print("=" * 60)

    results = []

    # 1. 静态场景 (无运动): 帧间差应只反映传感器噪声
    print("\n[1/2] Static scene (no motion, 100 frames)...")
    static = [make_static_scene() for _ in range(100)]
    results.append(evaluate_video(static, "static", side=12, psf_sigma=0.5, iso=100))

    # 2. 平移场景 (有运动): 帧间差应反映运动边缘
    print("[2/2] Translating scene (uniform motion, 50 frames)...")
    translating = [make_translating_scene(t=i) for i in range(50)]
    results.append(evaluate_video(translating, "translating", side=12, psf_sigma=0.5, iso=100))

    import json
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"  {'Scene':12s} {'Mode':10s} {'Mean':>8s} {'P95':>8s} {'Max':>8s}")
    print("=" * 60)
    for r in results:
        print(f"  {r['name']:12s} {r['mode']:10s} "
              f"{r['frame_diff_mean']:>8.3f} "
              f"{r['frame_diff_p95']:>8.3f} "
              f"{r['frame_diff_max']:>8.3f}")

    return results


if __name__ == '__main__':
    run_temporal_benchmark()
    print('\nDone.')