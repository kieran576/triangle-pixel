"""生成紧凑版 demo_video_small.gif 用于论文嵌入"""
import os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from triangle_engine import process_pipeline, sample_single_channels, borrow_neighbors
from triangle_3d import TriMesh


def make_frames(img_path, n_frames=24, side=8, size=(480, 360), out_dir='demo_video_frames'):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading {img_path}...")
    img = Image.open(img_path).convert('RGB').resize((400, 300))
    raw = process_pipeline(img, triangle_side=side, mode="raw")
    recon = process_pipeline(img, triangle_side=side, mode="correct")

    print("Building 3D mesh from triangular layout...")
    S = float(side); h = S * math.sqrt(3) / 2
    W, H = img.size
    n_cols = int(W / (S/2)) + 3
    n_rows = int(H / h) + 2
    pixels = np.array(img).astype(np.float32)
    single = sample_single_channels(pixels, S, h, n_rows, n_cols)
    borrowed = borrow_neighbors(single, n_rows, n_cols, edge_mode="mirror")
    mesh = TriMesh()
    mesh.from_triangle_grid(single, S, h, n_rows, n_cols, rgb_map=borrowed)

    frames = []
    for i in range(n_frames):
        ang = 2 * math.pi * i / n_frames
        elev = math.sin(2 * math.pi * i / n_frames) * 0.4
        azim = ang

        view = mesh.render_view(width=size[0]//2, height=size[1],
                                 azimuth=math.degrees(azim), elevation=math.degrees(elev))

        canvas = Image.new('RGB', size, 'white')
        canvas.paste(view, (0, 0))
        canvas.paste(img.resize((size[0]//2, size[1]//2)),
                     (size[0]//2, 0))
        canvas.paste(recon.resize((size[0]//2, size[1]//2)),
                     (size[0]//2, size[1]//2))

        d = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype('arial.ttf', 12)
            font_meta = ImageFont.truetype('arial.ttf', 10)
        except Exception:
            font = ImageFont.load_default()
            font_meta = font
        d.text((6, 6), 'Triangular 3D mesh',
               fill='white', font=font, stroke_width=1, stroke_fill='black')
        d.text((size[0]//2 + 6, 6), 'Original photo',
               fill='white', font=font, stroke_width=1, stroke_fill='black')
        d.text((size[0]//2 + 6, size[1]//2 + 6), 'Triangle ISP output',
               fill='white', font=font, stroke_width=1, stroke_fill='black')
        d.text((6, size[1] - 16),
               f'f{i+1}/{n_frames}  az={math.degrees(azim):.0f}°  el={math.degrees(elev):.0f}°',
               fill='white', font=font_meta, stroke_width=1, stroke_fill='black')

        frames.append(canvas)
    return frames


def make_gif(frames, out_path, duration_ms=140, max_colors=64):
    frames_q = [f.convert('P', palette=Image.ADAPTIVE, colors=max_colors) for f in frames]
    frames_q[0].save(out_path, save_all=True, append_images=frames_q[1:],
                     duration=duration_ms, loop=0, optimize=True, disposal=2)
    print(f'Saved {out_path}: {os.path.getsize(out_path)//1024} KB')


if __name__ == '__main__':
    src = 'bench_directional/real_proxy.png'
    frames = make_frames(src, n_frames=24, side=8, size=(480, 360))
    make_gif(frames, 'demo_video_small.gif', duration_ms=140, max_colors=64)
    print('Done.')