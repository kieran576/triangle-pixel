#!/usr/bin/env python3
"""生成论文图表 — matplotlib"""
import os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Polygon

os.makedirs("paper/figs", exist_ok=True)

# ============================================================
#  Fig 1: Triangular Grid Geometry
# ============================================================

def draw_triangle_grid():
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    S = 1.0
    h = S * math.sqrt(3) / 2

    # Draw triangles
    for r in range(6):
        for c in range(10):
            x = c * S / 2
            y = r * h
            up = (r + c) % 2 == 0
            if up:
                tri = Polygon([(x, y + h/3), (x - S/2, y + h), (x + S/2, y + h)],
                              facecolor='lightgray', edgecolor='black', linewidth=0.5)
            else:
                tri = Polygon([(x - S/2, y), (x + S/2, y), (x, y + 2*h/3)],
                              facecolor='white', edgecolor='black', linewidth=0.5)
            ax.add_patch(tri)

    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Triangular Grid Geometry', fontsize=12)
    plt.tight_layout()
    plt.savefig("paper/figs/fig1_grid.pdf", dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
#  Fig 2: Channel Assignment Pattern
# ============================================================

def draw_channel_assignment():
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    S = 1.0; h = S * math.sqrt(3) / 2
    colors = {0: '#FF4444', 1: '#44FF44', 2: '#4488FF'}
    labels = {0: 'R', 1: 'G', 2: 'B'}

    EVEN = [0, 1, 2, 2, 1, 0]
    ODD = [2, 1, 0, 0, 1, 2]

    for r in range(3):
        for c in range(12):
            x = c * S / 2
            y = r * h
            up = (r + c) % 2 == 0
            ch = EVEN[c % 6] if r % 2 == 0 else ODD[c % 6]

            if up:
                tri = Polygon([(x, y), (x - S/2, y + h), (x + S/2, y + h)],
                              facecolor=colors[ch], edgecolor='white', linewidth=0.5, alpha=0.9)
            else:
                tri = Polygon([(x - S/2, y), (x + S/2, y), (x, y + h)],
                              facecolor=colors[ch], edgecolor='white', linewidth=0.5, alpha=0.9)
            ax.add_patch(tri)

    # Highlight a hexagon
    hex_center = (3 * S/2, h)
    hexagon = RegularPolygon(hex_center, 6, radius=S*0.65, orientation=0,
                             facecolor='none', edgecolor='black', linewidth=2, linestyle='--')
    ax.add_patch(hexagon)

    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.text(3, h + 0.3, 'Hexagon: 2R+2G+2B', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title('Channel Assignment Pattern (Even rows: RGBBGR, Odd: BGRRGB)', fontsize=11)
    plt.tight_layout()
    plt.savefig("paper/figs/fig2_channels.pdf", dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
#  Fig 3: Sensor Comparison Bar Chart
# ============================================================

def draw_sensor_comparison():
    data = [
        ("Edge 0°", 26.3, 40.3),
        ("Edge 45°", 27.4, 40.3),
        ("Edge 90°", 27.9, 40.4),
        ("Edge 135°", 25.8, 40.3),
        ("Color R→B", 27.3, 39.3),
        ("Real Photo", 22.3, 35.4),
        ("Gray Ramp", 36.4, 42.0),
        ("Siemens", 14.6, 22.2),
    ]

    labels = [d[0] for d in data]
    tri_vals = [d[1] for d in data]
    bay_vals = [d[2] for d in data]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width/2, tri_vals, width, label='Triangle (S=12, 2% data)',
                   color='#FF6644', edgecolor='white')
    bars2 = ax.bar(x + width/2, bay_vals, width, label='Bayer (100% data)',
                   color='#4488CC', edgecolor='white')

    ax.set_ylabel('PSNR (dB)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 50)
    ax.grid(axis='y', alpha=0.3)

    # Add PSNR labels
    for bar, val in zip(bars1, tri_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=7)
    for bar, val in zip(bars2, bay_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=7)

    ax.set_title('Triangle vs Bayer Sensor PSNR Comparison (S=12)', fontsize=12)
    plt.tight_layout()
    plt.savefig("paper/figs/fig3_sensor_bars.pdf", dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
#  Fig 4: Anisotropy Radar Chart
# ============================================================

def draw_anisotropy():
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    tri_psnr = [26.3, 27.4, 27.9, 25.8, 26.3, 27.4, 27.9, 25.8]  # mirrored
    bay_psnr = [40.3, 40.3, 40.4, 40.3, 40.3, 40.3, 40.4, 40.3]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw=dict(polar=True))
    theta = np.radians(angles)

    ax.fill(theta, tri_psnr, alpha=0.3, color='#FF6644', label='Triangle (S=12)')
    ax.plot(theta, tri_psnr, color='#FF6644', linewidth=2)
    ax.fill(theta, bay_psnr, alpha=0.15, color='#4488CC', label='Bayer')
    ax.plot(theta, bay_psnr, color='#4488CC', linewidth=2, linestyle='--')

    ax.set_xticks(theta)
    ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
    ax.set_ylim(0, 45)
    ax.set_title('Directional Anisotropy (max−min = 2.1 dB)', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig("paper/figs/fig4_anisotropy.pdf", dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
#  Fig 5: AI vs ISP + Speed
# ============================================================

def draw_ai_vs_isp():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # AI vs ISP PSNR
    methods = ['Edge Image', 'Real Photo']
    ai_vals = [24.7, 20.0]
    isp_vals = [25.1, 21.4]
    x = np.arange(len(methods)); w = 0.3
    ax1.bar(x - w/2, ai_vals, w, label='GCN (end-to-end)', color='#44BB44')
    ax1.bar(x + w/2, isp_vals, w, label='ISP (hand-crafted)', color='#FF8844')
    for i in range(2):
        ax1.text(i - w/2, ai_vals[i] + 0.3, f'{ai_vals[i]:.1f}', ha='center', fontsize=8)
        ax1.text(i + w/2, isp_vals[i] + 0.3, f'{isp_vals[i]:.1f}', ha='center', fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(methods)
    ax1.set_ylabel('PSNR (dB)'); ax1.legend(fontsize=7)
    ax1.set_title('AI (GCN) vs Hand-Crafted ISP')
    ax1.grid(axis='y', alpha=0.3)

    # Speed
    sides = [16, 20, 24, 32]
    fps = [59, 85, 121, 203]
    ax2.plot(sides, fps, 'o-', color='#6644FF', linewidth=2, markersize=8)
    ax2.fill_between(sides, fps, alpha=0.1, color='#6644FF')
    for s, f in zip(sides, fps):
        ax2.text(s, f + 10, f'{f}', ha='center', fontsize=9)
    ax2.set_xlabel('Triangle Side (pixels)'); ax2.set_ylabel('FPS')
    ax2.set_title('Processing Speed (numba JIT)')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 240)

    plt.tight_layout()
    plt.savefig("paper/figs/fig5_ai_speed.pdf", dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
#  Fig 6: System Architecture (text-based)
# ============================================================

def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

    boxes = [
        (1, 5, 3, 0.8, 'Scene', '#DDDDDD'),
        (1, 3.5, 3, 0.8, 'Triangle RAW\n(single-channel)', '#FFCCCC'),
        (1, 2, 3, 2.2, 'Reconstruction\nBorrow + ISP + GCN', '#CCEEFF'),
        (5.5, 5, 3, 0.8, 'Bayer RAW\n(comparison)', '#DDDDDD'),
        (5.5, 3.5, 3, 0.8, 'Bilinear Demosaic', '#CCEEFF'),
        (5.5, 2, 3, 0.7, 'RGB Output', '#DDFFDD'),
        (1, 0.8, 8, 0.7, 'Downstream: Edge Detection | Harris Corners | 3D Mesh | Super-Resolution', '#EEEEFF'),
    ]

    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black',
                             linewidth=1, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8)

    # Arrows
    ax.annotate('', xy=(2.5, 4.3), xytext=(7, 4.3),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    ax.text(4.75, 4.5, 'PSNR/SSIM', ha='center', fontsize=7, color='gray')

    ax.set_title('Triangle Pixel System Architecture', fontsize=12)
    plt.tight_layout()
    plt.savefig("paper/figs/fig6_arch.pdf", dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
#  Generate all
# ============================================================

print("Generating paper figures...")
draw_triangle_grid();       print("  fig1_grid.pdf")
draw_channel_assignment();  print("  fig2_channels.pdf")
draw_sensor_comparison();   print("  fig3_sensor_bars.pdf")
draw_anisotropy();          print("  fig4_anisotropy.pdf")
draw_ai_vs_isp();           print("  fig5_ai_speed.pdf")
draw_architecture();        print("  fig6_arch.pdf")
print("Done: paper/figs/")
