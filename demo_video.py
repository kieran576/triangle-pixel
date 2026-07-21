"""
生成论文 demo 视频 (GIF 格式, 因为没有 ffmpeg)
展示: 三角 mesh 3D 旋转 + 原图 vs ISP 结果对比
"""
import os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from triangle_engine import process_pipeline, sample_single_channels, borrow_neighbors
from triangle_3d import TriMesh
import math


def make_frames(img_path, n_frames=36, side=8, size=(640, 480), out_dir='demo_video_frames'):
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

    tri_img_arr = np.array(img)
    h, w = tri_img_arr.shape[:2]

    frames = []
    for i in range(n_frames):
        ang = 2 * math.pi * i / n_frames
        elev = math.sin(2 * math.pi * i / n_frames) * 0.4
        azim = ang

        # Render 3D mesh using painter's algorithm
        view = mesh.render_view(width=size[0]//2, height=size[1],
                          azimuth=math.degrees(azim), elevation=math.degrees(elev))

        # Composite: 3D mesh | original | ISP reconstruction
        canvas = Image.new('RGB', size, 'white')
        # left: 3D mesh
        canvas.paste(view, (0, 0))
        # right top: original
        canvas.paste(img.resize((size[0]//2, size[1]//2)),
                     (size[0]//2, 0))
        # right bot: ISP recon
        canvas.paste(recon.resize((size[0]//2, size[1]//2)),
                     (size[0]//2, size[1]//2))

        # labels
        d = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype('arial.ttf', 16)
            font_big = ImageFont.truetype('arial.ttf', 18)
        except Exception:
            font = ImageFont.load_default()
            font_big = font
        d.text((10, 10), f'3D mesh (frame {i+1}/{n_frames})',
               fill='white', font=font, stroke_width=1, stroke_fill='black')
        d.text((size[0]//2 + 10, 10), 'Original photo',
               fill='white', font=font, stroke_width=1, stroke_fill='black')
        d.text((size[0]//2 + 10, size[1]//2 + 10), 'Triangle ISP reconstruction',
               fill='white', font=font, stroke_width=1, stroke_fill='black')
        # meta
        d.text((10, size[1]-30),
               f'S={side}  side  azim={math.degrees(azim):.0f}°  elev={math.degrees(elev):.0f}°',
               fill='white', font=font, stroke_width=1, stroke_fill='black')

        frames.append(canvas)

    return frames


def make_gif(frames, out_path='demo_video.gif', duration_ms=100):
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    print(f'Saved {out_path}: {os.path.getsize(out_path)//1024} KB')


if __name__ == '__main__':
    src = 'bench_directional/real_proxy.png'
    if not os.path.exists(src):
        # fallback to a bench image
        src = 'bench_images/01_edge.png'
    frames = make_frames(src, n_frames=36, side=8)
    make_gif(frames, out_path='demo_video.gif', duration_ms=120)
    print('Done.')