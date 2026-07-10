#!/usr/bin/env python3
"""
tri_convert.py — 矩形图像/视频 → 三角 RAW 数据集批量转换器

输入: 图片目录 / 视频文件 / 抖音视频目录
输出: .tri 对 (.npy RAW + .json 元数据)

用法:
    # 单张图片
    python tri_convert.py --source image ./photos/

    # 视频文件
    python tri_convert.py --source video ./videos/dance.mp4

    # 抖音视频目录 (自动竖屏适配)
    python tri_convert.py --source douyin ./douyin_videos/

    # 实时摄像头 (按 Q 退出)
    python tri_convert.py --source webcam

参数:
    --triangle-side   三角边长, 默认 12 (适合 1080p)
    --output-dir      输出目录, 默认 ./tri_output/
    --frame-step      视频每隔 N 帧取一帧, 默认 5
    --max-frames      每个视频最多取帧数, 默认 200
    --noise-iso       ISO 噪声级别 (0=无噪声), 默认 0
    --workers         并行线程数, 默认 4
    --denoise-douyin  抖音视频轻度去压缩伪影
    --crop-watermark  裁掉右上角水印区域
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np
from PIL import Image

# 尝试导入 tqdm, 没有就用简单进度
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# 尝试导入 OpenCV
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# 导入三角引擎
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triangle_engine import sample_single_channels, tri_center
from triangle_sensor import TriangleSensor


# ============================================================
#  抖音专用预处理
# ============================================================

def denoise_douyin(frame):
    """轻度去抖音压缩伪影 (fastNlMeansDenoisingColored)."""
    if not HAS_CV2:
        return frame
    return cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)


def crop_watermark(pil_img):
    """裁掉右上角水印区域 (约占宽度 12%, 高度 8%)."""
    w, h = pil_img.size
    crop_w = int(w * 0.12)
    crop_h = int(h * 0.08)
    # 用邻域平均填充裁掉的水印区域, 避免破坏三角网格连续性
    arr = np.array(pil_img)
    # 右上角 = 用左边紧邻像素填充
    for y in range(crop_h):
        for x in range(w - crop_w, w):
            arr[y, x] = arr[y, w - crop_w - 1]
    return Image.fromarray(arr)


# ============================================================
#  核心转换
# ============================================================


def _save_preview(raw_path, tri_raw, sensor):
    """从三角 RAW 重建 PNG 预览, 存到 .npy 旁边."""
    try:
        from triangle_engine import borrow_neighbors, correct_triangular_isp, render_triangles
        nr, nc = tri_raw.shape
        S, h = sensor.triangle_side, sensor.triangle_side * 0.8660254
        borrowed = borrow_neighbors(tri_raw, nr, nc, edge_mode="mirror")
        corrected = correct_triangular_isp(tri_raw, borrowed, nr, nc, iterations=2)
        # 推算原图尺寸
        W = int(nc * S / 2.0)
        H = int(nr * h)
        result = render_triangles(corrected, S, h, nr, nc, W, H)
        png_path = raw_path.replace(".npy", "_preview.png")
        result.save(png_path)
    except Exception:
        pass  # 预览失败不影响主流程
def convert_image(img_path, sensor, output_dir, denoise=False, watermark=False):
    """转换单张图片 → .tri 对."""
    scene = Image.open(img_path).convert("RGB")

    if watermark:
        scene = crop_watermark(scene)

    # 三角传感器捕获
    tri_raw, meta = sensor.capture(scene)

    if denoise and HAS_CV2:
        # 在 RAW 域做轻度去噪 (不同于 RGB 域)
        tri_raw = cv2.fastNlMeansDenoising(
            tri_raw.astype(np.uint8), None, 5, 7, 21
        ).astype(np.float32)

    # 保存
    stem = Path(img_path).stem
    raw_path = os.path.join(output_dir, f"{stem}.npy")
    meta_path = os.path.join(output_dir, f"{stem}.json")
    meta["original"] = str(img_path)

    np.save(raw_path, tri_raw.astype(np.float32))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _save_preview(raw_path, tri_raw, sensor)
    return raw_path


def convert_video(video_path, sensor, output_dir, frame_step=5,
                  max_frames=200, denoise=False, watermark=False):
    """转换视频文件 → 帧序列 .tri 目录."""
    if not HAS_CV2:
        print("  [跳过] 需要 OpenCV (pip install opencv-python)")
        return []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # 视频输出子目录
    stem = Path(video_path).stem
    vid_dir = os.path.join(output_dir, stem)
    os.makedirs(vid_dir, exist_ok=True)

    # 保存视频元信息
    video_meta = {
        "source": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "frame_step": frame_step,
        "sensor_S": sensor.triangle_side,
    }
    with open(os.path.join(vid_dir, "_video.json"), "w", encoding="utf-8") as f:
        json.dump(video_meta, f, ensure_ascii=False, indent=2)

    saved = []
    frame_idx = 0
    saved_count = 0

    pbar = tqdm(total=min(total_frames, max_frames * frame_step),
                desc=f"  {stem}") if HAS_TQDM else None

    while saved_count < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if watermark:
                frame_rgb = cv2.cvtColor(
                    np.array(crop_watermark(
                        Image.fromarray(frame_rgb)
                    )),
                    cv2.COLOR_RGB2BGR
                )
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

            if denoise:
                frame_rgb = denoise_douyin(frame_rgb)

            scene = Image.fromarray(frame_rgb)
            tri_raw, meta = sensor.capture(scene)

            frame_name = f"frame_{saved_count:06d}"
            raw_path = os.path.join(vid_dir, f"{frame_name}.npy")
            meta_path = os.path.join(vid_dir, f"{frame_name}.json")
            meta["original"] = str(video_path)
            meta["frame_index"] = frame_idx

            np.save(raw_path, tri_raw.astype(np.float32))
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            _save_preview(raw_path, tri_raw, sensor)
            saved.append(raw_path)
            saved_count += 1

        frame_idx += 1
        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()
    cap.release()
    return saved


def convert_webcam(sensor, output_dir, denoise=False):
    """实时摄像头预览 + 按 S 保存截图."""
    if not HAS_CV2:
        print("需要 OpenCV")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    os.makedirs(output_dir, exist_ok=True)
    capture_count = 0

    print("\n  按 S 保存截图 | 按 Q 退出\n")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if denoise:
            frame_rgb = denoise_douyin(frame_rgb)

        scene = Image.fromarray(frame_rgb)
        tri_raw, meta = sensor.capture(scene)

        # 重建 RGB 用于预览
        from triangle_engine import borrow_neighbors, correct_triangular_isp, render_triangles
        nr, nc = meta["n_rows"], meta["n_cols"]
        S_val, h_val = meta["S"], meta["h"]
        borrowed = borrow_neighbors(tri_raw, nr, nc, edge_mode="mirror")
        corrected = correct_triangular_isp(tri_raw, borrowed, nr, nc, iterations=2)
        preview = render_triangles(corrected, S_val, h_val, nr, nc,
                                   frame_rgb.shape[1], frame_rgb.shape[0])
        preview_bgr = cv2.cvtColor(np.array(preview), cv2.COLOR_RGB2BGR)

        # 显示: 原始 | 三角重建
        display = np.hstack([frame_bgr, preview_bgr])
        cv2.imshow("Tri Camera - S:Save Q:Quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            name = f"capture_{capture_count:04d}"
            np.save(os.path.join(output_dir, f"{name}.npy"), tri_raw.astype(np.float32))
            with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
                m = dict(meta)
                m["capture_index"] = capture_count
                json.dump(m, f, ensure_ascii=False, indent=2)
            print(f"  [保存] {name}")
            capture_count += 1

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
#  批量处理
# ============================================================

def collect_images(source_dir):
    """收集目录下所有图片文件."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    paths = []
    for f in sorted(os.listdir(source_dir)):
        if Path(f).suffix.lower() in exts:
            paths.append(os.path.join(source_dir, f))
    return paths


def collect_videos(source_dir):
    """收集目录下所有视频文件."""
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    paths = []
    for f in sorted(os.listdir(source_dir)):
        if Path(f).suffix.lower() in exts:
            paths.append(os.path.join(source_dir, f))
    return paths


# ============================================================
#  主入口
# ============================================================


def replay_video(tri_dir):
    """Reconstruct playable .mp4 from .tri frame directory."""
    if not HAS_CV2:
        print("Need OpenCV: pip install opencv-python")
        return
    video_meta = {}
    mp = os.path.join(tri_dir, "_video.json")
    if os.path.exists(mp):
        with open(mp, "r", encoding="utf-8") as f:
            video_meta = json.load(f)
    fps = video_meta.get("fps", 30)
    frames = sorted([f for f in os.listdir(tri_dir) if f.startswith("frame_") and f.endswith(".npy")])
    if not frames:
        print(f"No frames in: {tri_dir}")
        return
    print(f"Replay: {len(frames)} frames @ {fps} FPS")
    raw0 = np.load(os.path.join(tri_dir, frames[0]))
    nr, nc = raw0.shape
    mp0 = os.path.join(tri_dir, frames[0].replace(".npy", ".json"))
    with open(mp0, "r", encoding="utf-8") as f:
        meta0 = json.load(f)
    S = meta0.get("S", 12)
    h = meta0.get("h", S * math.sqrt(3) / 2.0)
    W = meta0.get("W", int(nc * S / 2.0))
    H = meta0.get("H", int(nr * h))
    out = os.path.join(tri_dir, "reconstructed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, fps, (W, H))
    from triangle_engine import borrow_neighbors, correct_triangular_isp, render_triangles
    it = tqdm(frames, desc="Replay") if HAS_TQDM else frames
    for fn in it:
        raw = np.load(os.path.join(tri_dir, fn)).astype(np.float32)
        b = borrow_neighbors(raw, nr, nc, edge_mode="mirror")
        c = correct_triangular_isp(raw, b, nr, nc, iterations=2)
        r = render_triangles(c, S, h, nr, nc, W, H)
        writer.write(cv2.cvtColor(np.array(r), cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved: {out}")


def main():
    # Handle --replay before argparse (doesnt need --source)
    if "--replay" in sys.argv:
        idx = sys.argv.index("--replay")
        tri_dir = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "./tri_output/"
        replay_video(tri_dir)
        return
    parser = argparse.ArgumentParser(
        description="矩形图像/视频 → 三角 RAW 批量转换器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tri_convert.py --source image ./photos/ --triangle-side 12
  python tri_convert.py --source video ./dance.mp4 --frame-step 3
  python tri_convert.py --source douyin ./douyin_videos/ --denoise-douyin
  python tri_convert.py --source webcam --triangle-side 16
        """,
    )
    parser.add_argument("--source", choices=["image", "video", "douyin", "webcam"],
                        required=True, help="输入类型")
    parser.add_argument("path", nargs="?", default=None,
                        help="图片/视频路径 (留空弹出文件选择框, webcam 模式忽略)")
    parser.add_argument("--triangle-side", type=int, default=12,
                        help="三角边长 (像素), 默认 12")
    parser.add_argument("--output-dir", default="./tri_output/",
                        help="输出目录, 默认 ./tri_output/")
    parser.add_argument("--frame-step", type=int, default=5,
                        help="视频每隔 N 帧取一帧, 默认 5")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="每个视频最多取帧数, 默认 200")
    parser.add_argument("--noise-iso", type=int, default=0,
                        help="ISO 噪声级别 (0=无噪声), 默认 0")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行线程数, 默认 4")
    parser.add_argument("--denoise-douyin", action="store_true",
                        help="抖音视频轻度去压缩伪影")
    parser.add_argument("--crop-watermark", action="store_true",
                        help="Crop watermark area")
    parser.add_argument("--replay", action="store_true",
                        help="Reconstruct playable .mp4 from converted .tri frames")
    args = parser.parse_args()

    if args.replay:
        replay_video(args.path or args.output_dir)
        return

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 传感器
    sensor = TriangleSensor(
        triangle_side=args.triangle_side,
        psf_sigma=0.5,
        iso=args.noise_iso,
        read_noise=2.0 if args.noise_iso > 0 else 0,
        dark_current=0.3 if args.noise_iso > 0 else 0,
    )

    # 抖音模式: 自动竖屏适配
    douyin_mode = (args.source == "douyin")
    denoise = args.denoise_douyin or douyin_mode
    watermark = args.crop_watermark

    print(f"三角边长: {args.triangle_side}px")
    print(f"输出目录: {args.output_dir}")
    print(f"传感器噪声: ISO {args.noise_iso}")
    if douyin_mode:
        print("抖音模式: 竖屏适配 + 去压缩伪影")
    print()

    if args.source == "webcam":
        convert_webcam(sensor, args.output_dir, denoise=denoise)
        print("完成")
        return

    # 文件选择框
    src_path = args.path
    if not src_path:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if args.source == "image":
            src_path = filedialog.askdirectory(title="选择图片目录")
        else:
            src_path = filedialog.askopenfilename(
                title="选择视频文件",
                filetypes=[("视频文件", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
                           ("所有文件", "*.*")]
            )
        root.destroy()
        if not src_path:
            print("未选择文件, 退出")
            sys.exit(0)
        print(f"已选择: {src_path}")

    # 图片模式
    if args.source == "image":
        image_paths = []
        path = Path(src_path)
        if path.is_dir():
            image_paths = collect_images(args.path)
        elif path.is_file():
            image_paths = [str(path)]
        else:
            print(f"路径不存在: {args.path}")
            sys.exit(1)

        print(f"图片数: {len(image_paths)}")
        if HAS_TQDM:
            pbar = tqdm(total=len(image_paths), desc="转换图片")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(convert_image, p, sensor, args.output_dir,
                            denoise, watermark): p
                for p in image_paths
            }
            for future in as_completed(futures):
                if HAS_TQDM:
                    pbar.update(1)
        if HAS_TQDM:
            pbar.close()
        print(f"完成: {len(image_paths)} 张图片 → {args.output_dir}")

    # 视频模式
    elif args.source in ("video", "douyin"):
        video_paths = []
        path = Path(src_path)
        if path.is_dir():
            video_paths = collect_videos(args.path)
        elif path.is_file():
            video_paths = [str(path)]
        else:
            print(f"路径不存在: {args.path}")
            sys.exit(1)

        print(f"视频数: {len(video_paths)}")
        total_saved = 0
        for vp in video_paths:
            print(f"处理: {Path(vp).name}")
            saved = convert_video(
                vp, sensor, args.output_dir,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                denoise=denoise,
                watermark=watermark,
            )
            total_saved += len(saved)
            print(f"  → {len(saved)} 帧")

        print(f"\n完成: {len(video_paths)} 个视频, 共 {total_saved} 帧 → {args.output_dir}")


if __name__ == "__main__":
    main()
