# Triangle Pixel — 三角网格视觉系统

**用等边三角形网格替代矩形像素网格的完整计算机视觉管线。**

从传感器物理模拟到 AI 推理，端到端的三角原生处理。
无需 Bayer 滤波器，无需传统 ISP，无需矩形 CNN。

## 核心理念

```
传统视觉:  Bayer RAW → ISP → RGB → 矩形CV → 结果
三角视觉:  三角RAW → GCN → RGB → 三角CV → 结果
                  ↘ 零计算六边形 → 人眼直出
                  ↘ Sierpinski → 无限超分
                  ↘ 三角面片 → 3D原生
```

## 快速开始

```bash
# 安装依赖
pip install pillow numpy numba

# 启动 GUI（15种模式）
python triangle_pixel_gui.py

# 或跑完整 demo
python demo.py
```

## 系统架构

```
┌────────────────────────────────────────────────────────┐
│                   triangle_pixel_gui.py                │
│                     (15 modes)                         │
├────────────────────────────────────────────────────────┤
│  L1-L3 ISP     │  Phase 2 CV    │  Phase 3-5 AI/3D    │
│  triangle_     │  triangle_cv   │  triangle_features  │
│  engine.py     │  .py           │  .py                │
│  engine_fast   │                │  triangle_3d.py     │
│  .py (numba)   │                │  triangle_ai.py     │
│                │                │  demosaic_ai.py     │
├────────────────────────────────────────────────────────┤
│  传感器/仿真                                         │
│  triangle_sensor.py   triangle_superres.py            │
│  triangle_sierpinski.py                                │
├────────────────────────────────────────────────────────┤
│  测试/基准                                            │
│  benchmark.py → BENCHMARK.md                          │
└────────────────────────────────────────────────────────┘
```

## 15 种 GUI 模式

| 模式 | 功能 | 阶段 |
|------|------|------|
| RAW 马赛克 | 单通道三角纯色 (零计算) | L1 |
| RAW 降采样 | 六边形→矩形像素 | L1 |
| 邻居借用 | 3邻域借通道→全彩 | L2 |
| 双边校正 | 比值平滑校正 | L3 |
| ISP 去伪色 | 三角原生去伪色 | L3 |
| Sierpinski | 剖分线可视化 | L4 |
| 超分 2× / 4× | Sierpinski 超分辨率 | L4 |
| CV 亮度 | 单通道→亮度估计 | P2 |
| CV 边缘 | 三角 Canny 等价 | P2 |
| CV 多尺度边缘 | Sierpinski 金字塔边缘 | P2 |
| CV 特征点 | Tri-Harris 角点 | P3 |
| 3D 视图 | 三角网格→3D 渲染 | P4 |
| AI 去噪 | GCN 去噪 | P5 |
| AI 直出RGB | 端到端 GCN 去马赛克 | P5 |
| 传感器对比 | 三角 vs Bayer 模拟 | 仿真 |

## 核心基准数据

### 传感器效率 (S=8, 仅 4% 数据量 vs Bayer)

| 测试图 | TRI PSNR | Bayer PSNR | TRI SSIM |
|--------|----------|-----------|----------|
| 渐变 | 38.0 dB | 42.0 dB | 0.999 |
| 边缘 | 29.1 dB | 40.3 dB | 0.992 |
| 色块 | 18.9 dB | 32.3 dB | 0.970 |
| **平均** | **25.1 dB** | **35.2 dB** | — |

**数据效率: 6.49 dB PSNR 每 1% 数据量**

### 处理速度 (numba 加速, CPU)

| 三角边长 | 三角数 | 耗时 | FPS |
|---------|--------|------|-----|
| S=16 | 1,219 | 17ms | 59 |
| S=20 | 817 | 12ms | 85 |
| S=24 | 576 | 8ms | 121 |
| S=32 | 336 | 5ms | 203 |

### AI vs 手工 ISP

| 测试图 | AI PSNR | ISP PSNR |
|--------|---------|----------|
| 边缘 | 24.7 dB | 25.1 dB |
| 渐变 | 20.5 dB | 36.4 dB |
| 色块 | 11.9 dB | 14.7 dB |

AI 在边缘图上接近手工 ISP（差 0.4 dB）。渐变图上 ISP 优势大，因为有强先验。

## 文件清单

| 文件 | 功能 | 行数 |
|------|------|------|
| `triangle_engine.py` | RAW/BORROW/ISP | 590 |
| `triangle_engine_fast.py` | numba 加速 (54×) | 290 |
| `triangle_cv.py` | 边缘检测/多尺度 | 320 |
| `triangle_features.py` | Harris/16D描述子/匹配 | 440 |
| `triangle_3d.py` | 3D mesh/OBJ导出 | 290 |
| `triangle_ai.py` | GCN 去噪 | 350 |
| `triangle_demosaic_ai.py` | 端到端 AI 去马赛克 | 320 |
| `triangle_superres.py` | Sierpinski 超分 | 310 |
| `triangle_sensor.py` | 传感器模拟器 | 300 |
| `triangle_sierpinski.py` | 剖分几何 | 280 |
| `benchmark.py` | 基准测试套件 | 310 |
| `demo.py` | 一键 demo | — |
| `triangle_pixel_gui.py` | GUI (15 modes) | 490 |
| **总计** | **~4800 行** | |

## 引用

如果这个项目对你的研究有帮助：

```bibtex
@misc{wang2025triangle,
  title   = {Triangle Pixel: A Triangular-Grid Vision System},
  author  = {Kieran Wang},
  year    = {2025},
  eprint  = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  note    = {arXiv:XXXX.XXXXX}
}
```

> 论文全文: [arxiv_submission/paper.tex](arxiv_submission/paper.tex)
> 编译: 上传 `arxiv_submission/` 到 [Overleaf](https://overleaf.com)

## 许可证

MIT
