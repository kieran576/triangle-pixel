"""
时序一致性可视化: 把 Section 8.4 的"零延迟/无伪影"实证成视频
- 静态场景 30 帧: 帧间差应全为 0 (肉眼无变化)
- 平移场景 30 帧: 帧间差只在运动边缘
- 旁置一个 frame-difference heatmap 实时显示
"""
import os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from triangle_engine import process_pipeline, sample_single_channels, borrow_neighbors
from triangle_sensor import optical_blur


def make_static_scene(W=400, H=300):
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    Y, X = np.indices((H, W))
    arr[..., 0] = np.clip(40 + 120 * (X / W), 0, 255).astype(np.uint8)
    arr[..., 1] = np.clip(80 + 80 * (Y / H), 0, 255).astype(np.uint8)
    arr[..., 2] = np.clip(120 - 60 * (X / W), 0, 255).astype(np.uint8)
    arr[60:160, 30:130] = (220, 200, 60)
    arr[80:220, 180:340] = (80, 200, 100)
    arr[200:280, 250:380] = (180, 80, 80)
    return Image.fromarray(arr)


def make_translating_scene(W=400, H=300, t=0):
    base = np.array(make_static_scene(W, H))
    period = 30
    shift = int(W * (t % period) / period)
    shifted = np.zeros_like(base)
    shifted[:, :W - shift] = base[:, shift:]
    return Image.fromarray(shifted)


def render_heatmap(diff, size, max_scale=20.0):
    """L1 帧差灰度热力图 (白色=无差, 红色=有差)"""
    h, w = diff.shape[:2]
    img = Image.new("RGB", (w, h), (255, 255, 255))
    arr = np.array(img).astype(np.float32)
    d = np.clip(diff, 0, max_scale) / max_scale
    arr[..., 0] = 255 - d * 255  # R 减小
    arr[..., 1] = 255 - d * 255  # G 减小
    arr[..., 2] = 255            # B 保持
    # 高亮处用红色
    red_mask = d > 0.3
    arr[red_mask, 0] = 255
    arr[red_mask, 1] = 80
    arr[red_mask, 2] = 80
    return img.resize(size)


def render_temporal_video(out_path, scene_name, n_frames=30, side=12, size=(720, 360)):
    """生成 3 栏对比视频: 原图 | ISP 输出 | 帧间差热力图"""
    print(f"Generating temporal video for {scene_name}...")

    frames_tri = []
    diffs = []
    prev = None

    for t in range(n_frames):
        if scene_name == "static":
            img = make_static_scene()
        else:
            img = make_translating_scene(t=t)

        blurred = optical_blur(img, psf_sigma=0.5)
        tri = process_pipeline(blurred, triangle_side=side, mode="correct")
        frames_tri.append(tri)

        if prev is not None:
            d = np.abs(np.array(tri).astype(np.float32) -
                       np.array(prev).astype(np.float32))
            d_max = d.mean(axis=2)  # per-pixel L1 average across channels
            diffs.append(d_max)
        prev = tri

    # 现在合成 GIF
    canvas_w, canvas_h = size
    bar_w = canvas_w // 3
    bar_h = canvas_h

    frames_out = []
    for t in range(n_frames):
        canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
        # 左: 原图
        canvas.paste(frames_tri[t-1 if t > 0 else 0].resize((bar_w, bar_h)), (0, 0))
        # 中: ISP 输出
        canvas.paste(frames_tri[t].resize((bar_w, bar_h)), (bar_w, 0))
        # 右: 帧间差热力图
        if t == 0:
            # 第一帧无 diff, 用零热力图
            heat = np.zeros((bar_h, bar_w), dtype=np.float32)
            heatmap = render_heatmap(heat, (bar_w, bar_h), max_scale=10.0)
        else:
            # 把最近 diff 上采样到 bar_w x bar_h
            d_map = diffs[t-1]
            heatmap = render_heatmap(d_map, (bar_w, bar_h), max_scale=10.0)
        canvas.paste(heatmap, (2 * bar_w, 0))

        d = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype('arial.ttf', 14)
            font_meta = ImageFont.truetype('arial.ttf', 11)
        except Exception:
            font = ImageFont.load_default()
            font_meta = font

        d.text((6, 6), 'Frame t-1',
               fill='white', font=font, stroke_width=1, stroke_fill='black')
        d.text((bar_w + 6, 6), f'Frame t ({scene_name})',
               fill='white', font=font, stroke_width=1, stroke_fill='black')
        d.text((2 * bar_w + 6, 6), '|t - (t-1)| heatmap',
               fill='white', font=font, stroke_width=1, stroke_fill='black')
        # 标签: 当前帧最大帧差
        if t > 0 and diffs:
            mean_d = float(diffs[t-1].mean())
            max_d = float(diffs[t-1].max())
            d.text((6, canvas_h - 18),
                   f'frame {t+1}/{n_frames}  mean L1={mean_d:.3f}  max={max_d:.1f}',
                   fill='white', font=font_meta, stroke_width=1, stroke_fill='black')
        else:
            d.text((6, canvas_h - 18),
                   f'frame {t+1}/{n_frames}  (reference frame)',
                   fill='white', font=font_meta, stroke_width=1, stroke_fill='black')

        frames_out.append(canvas)

    # 保存 GIF
    frames_q = [f.convert('P', palette=Image.ADAPTIVE, colors=96) for f in frames_out]
    frames_q[0].save(out_path, save_all=True, append_images=frames_q[1:],
                     duration=200, loop=0, optimize=True, disposal=2)
    print(f'  Saved {out_path}: {os.path.getsize(out_path)//1024} KB')


if __name__ == '__main__':
    print("=" * 60)
    print("  Section 8.4 temporal consistency demo video")
    print("=" * 60)
    render_temporal_video('demo_temporal_static.gif', 'static', n_frames=30)
    render_temporal_video('demo_temporal_translating.gif', 'translating', n_frames=30)
    print("Done.")