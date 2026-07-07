#!/usr/bin/env python3
"""
三角形像素图像处理器 — GUI v4 (全参数可调)

四层管线 + 可调参数:
  - 采样半径 (中心/区域)
  - 边缘模式 (镜像/黑色)
  - 双边滤波强度 (保边缘)
  - RAW 降采样倍率
  - Sierpinski 剖分深度
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk

from triangle_engine import process_pipeline
from triangle_sierpinski import render_sierpinski_overlay


class TrianglePixelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Triangle Pixel — 三角形像素图像处理器 v4")
        self.root.geometry("1250x820")
        self.root.minsize(1050, 620)

        self.original_image = None
        self.processed_image = None
        self.processing = False

        self._build_menu()
        self._build_ui()
        self._show_placeholder()

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开图片...", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="保存结果...", command=self.save_result, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_result())

    def _build_ui(self):
        # ========== 预览 ==========
        preview = ttk.Frame(self.root)
        preview.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        left = ttk.LabelFrame(preview, text="原图")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.original_label = ttk.Label(left, anchor=tk.CENTER, background="#d0d0d0")
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        right = ttk.LabelFrame(preview, text="处理后")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.processed_label = ttk.Label(right, anchor=tk.CENTER, background="#d0d0d0")
        self.processed_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ========== 控制面板 ==========
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill=tk.X, padx=10, pady=(5, 10))

        # === 第1行：模式 + 边长 ===
        row1 = ttk.Frame(ctrl)
        row1.pack(fill=tk.X, pady=(0, 8))

        # 模式
        mode_frame = ttk.LabelFrame(row1, text="处理模式")
        mode_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.mode_var = tk.StringVar(value="borrow")
        modes = [
            ("RAW 马赛克", "raw"),
            ("RAW 降采样", "raw_downscale"),
            ("邻居借用", "borrow"),
            ("双边校正", "correct"),
            ("ISP 去伪色", "correct_isp"),
            ("Sierpinski", "sierpinski"),
            ("超分 2×", "superres2"),
            ("CV 亮度", "cv_lum"),
            ("CV 边缘", "cv_edge"),
            ("CV 多尺度边缘", "cv_edge_ms"),
            ("CV 特征点", "cv_features"),
            ("3D 视图", "view3d"),
            ("AI 去噪", "ai_denoise"),
            ("传感器对比", "sensor"),
            ("AI 直出RGB", "ai_demosaic"),
        ]
        for i, (text, val) in enumerate(modes):
            ttk.Radiobutton(
                mode_frame, text=text, variable=self.mode_var, value=val,
                command=self._on_mode_change,
            ).grid(row=i // 3, column=i % 3, sticky=tk.W, padx=8, pady=2)

        # 三角形边长
        size_frame = ttk.LabelFrame(row1, text="三角形边长")
        size_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.side_var = tk.IntVar(value=20)
        ttk.Scale(size_frame, from_=4, to=80, variable=self.side_var,
                  orient=tk.HORIZONTAL, length=140,
                  command=lambda v: self.side_label.configure(text=f"{int(float(v))} px")
                  ).pack(side=tk.LEFT, padx=5, pady=5)
        self.side_label = ttk.Label(size_frame, text="20 px", width=6)
        self.side_label.pack(side=tk.LEFT, padx=5)

        # 采样半径
        sample_frame = ttk.LabelFrame(row1, text="采样精度")
        sample_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.sample_var = tk.DoubleVar(value=1.5)
        ttk.Scale(sample_frame, from_=0, to=4, variable=self.sample_var,
                  orient=tk.HORIZONTAL, length=100,
                  command=lambda v: self.sample_label.configure(
                      text=f"{float(v):.1f} px" if float(v) > 0 else "中心")
                  ).pack(side=tk.LEFT, padx=5, pady=5)
        self.sample_label = ttk.Label(sample_frame, text="1.5 px", width=7)
        self.sample_label.pack(side=tk.LEFT, padx=5)

        # 边缘模式
        edge_frame = ttk.LabelFrame(row1, text="边界")
        edge_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.edge_var = tk.StringVar(value="mirror")
        ttk.Radiobutton(edge_frame, text="镜像", variable=self.edge_var,
                        value="mirror").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(edge_frame, text="黑色", variable=self.edge_var,
                        value="zero").pack(side=tk.LEFT, padx=5)

        # === 第2行：校正参数 (仅 correct 模式有效) ===
        row2 = ttk.Frame(ctrl)
        row2.pack(fill=tk.X, pady=(0, 8))

        # 校正迭代
        self.iter_frame = ttk.LabelFrame(row2, text="校正迭代")
        self.iter_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.iter_var = tk.IntVar(value=3)
        ttk.Scale(self.iter_frame, from_=1, to=10, variable=self.iter_var,
                  orient=tk.HORIZONTAL, length=120,
                  command=lambda v: self.iter_label.configure(text=f"{int(float(v))} 次")
                  ).pack(side=tk.LEFT, padx=5, pady=5)
        self.iter_label = ttk.Label(self.iter_frame, text="3 次", width=5)
        self.iter_label.pack(side=tk.LEFT, padx=5)

        # 双边滤波 σ
        self.bilateral_frame = ttk.LabelFrame(row2, text="保边缘强度 (σ)")
        self.bilateral_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.sigma_var = tk.DoubleVar(value=0.15)
        ttk.Scale(self.bilateral_frame, from_=0.03, to=0.5, variable=self.sigma_var,
                  orient=tk.HORIZONTAL, length=120,
                  command=lambda v: self.sigma_label.configure(text=f"{float(v):.2f}")
                  ).pack(side=tk.LEFT, padx=5, pady=5)
        self.sigma_label = ttk.Label(self.bilateral_frame, text="0.15", width=5)
        self.sigma_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.bilateral_frame, text="小=锐利 大=平滑", font=("", 8)
                  ).pack(side=tk.LEFT, padx=5)

        # RAW 降采样倍率
        self.raw_scale_frame = ttk.LabelFrame(row2, text="RAW 降采样倍率")
        self.raw_scale_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.raw_scale_var = tk.IntVar(value=2)
        ttk.Scale(self.raw_scale_frame, from_=1, to=6, variable=self.raw_scale_var,
                  orient=tk.HORIZONTAL, length=100,
                  command=lambda v: self.raw_scale_label.configure(text=f"×{int(float(v))}")
                  ).pack(side=tk.LEFT, padx=5, pady=5)
        self.raw_scale_label = ttk.Label(self.raw_scale_frame, text="×2", width=4)
        self.raw_scale_label.pack(side=tk.LEFT, padx=5)

        # Sierpinski 深度
        self.sier_frame = ttk.LabelFrame(row2, text="Sierpinski 剖分深度")
        self.sier_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.depth_var = tk.IntVar(value=2)
        ttk.Scale(self.sier_frame, from_=1, to=4, variable=self.depth_var,
                  orient=tk.HORIZONTAL, length=100,
                  command=lambda v: self.depth_label.configure(text=f"{int(float(v))} 层")
                  ).pack(side=tk.LEFT, padx=5, pady=5)
        self.depth_label = ttk.Label(self.sier_frame, text="2 层", width=5)
        self.depth_label.pack(side=tk.LEFT, padx=5)

        # === 第3行：按钮 + 进度 ===
        row3 = ttk.Frame(ctrl)
        row3.pack(fill=tk.X)

        btn_frame = ttk.Frame(row3)
        btn_frame.pack(side=tk.LEFT, padx=(0, 15))

        self.process_btn = ttk.Button(btn_frame, text="▶ 处理", command=self.process, width=12)
        self.process_btn.pack(pady=3)

        self.save_btn = ttk.Button(btn_frame, text="保存结果...", command=self.save_result, width=12)
        self.save_btn.pack(pady=3)

        self.progress = ttk.Progressbar(row3, mode="determinate", length=250)
        self.progress.pack(side=tk.LEFT, padx=(0, 10))

        self.status_var = tk.StringVar(value="就绪 — 文件 → 打开图片")
        ttk.Label(row3, textvariable=self.status_var).pack(side=tk.LEFT)

        self._on_mode_change()

    def _on_mode_change(self):
        mode = self.mode_var.get()
        correct_on = (mode in ("correct", "correct_isp", "superres2", "superres4", "ai_denoise", "sensor"))
        raw_on = (mode == "raw_downscale")
        sier_on = (mode == "sierpinski")
        superres_on = (mode in ("superres2", "superres4"))
        cv_on = mode.startswith("cv_")
        ai_on = (mode == "ai_denoise")

        for w in self.iter_frame.winfo_children():
            w.configure(state="normal" if correct_on else "disabled")
        for w in self.bilateral_frame.winfo_children():
            w.configure(state="normal" if correct_on else "disabled")
        for w in self.raw_scale_frame.winfo_children():
            w.configure(state="normal" if raw_on else "disabled")
        for w in self.sier_frame.winfo_children():
            w.configure(state="normal" if sier_on else "disabled")

    def _show_placeholder(self):
        for label in [self.original_label, self.processed_label]:
            label.configure(text="拖拽图片到此处\n或\n文件 → 打开图片",
                            font=("Microsoft YaHei", 11))

    def _fit_image(self, pil_image, max_w, max_h):
        w, h = pil_image.size
        ratio = min(max_w / w, max_h / h, 1.0)
        if ratio < 1.0:
            return pil_image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return pil_image.copy()

    def _update_preview(self, pil_image, label_widget):
        label_widget.update_idletasks()
        w = label_widget.winfo_width()
        h = label_widget.winfo_height()
        if w < 10:
            w, h = 400, 300
        preview = self._fit_image(pil_image, w - 10, h - 10)
        photo = ImageTk.PhotoImage(preview)
        label_widget.configure(image=photo, text="")
        label_widget.image = photo

    def open_image(self):
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp"),
                       ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            self.original_image = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图片:\n{e}")
            return
        self.processed_image = None
        self._update_preview(self.original_image, self.original_label)
        self.processed_label.configure(image="", text="点击「处理」生成结果")
        name = path.replace("\\", "/").split("/")[-1]
        self.status_var.set(f"已加载: {name} ({self.original_image.size[0]}×{self.original_image.size[1]})")

    def save_result(self):
        if self.processed_image is None:
            messagebox.showinfo("提示", "请先处理一张图片")
            return
        path = filedialog.asksaveasfilename(
            title="保存结果", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if not path:
            return
        try:
            self.processed_image.save(path)
            self.status_var.set(f"已保存: {path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败:\n{e}")

    def process(self):
        if self.original_image is None:
            messagebox.showinfo("提示", "请先打开一张图片")
            return
        if self.processing:
            return

        self.processing = True
        self.process_btn.configure(state="disabled")
        self.progress["value"] = 0

        mode = self.mode_var.get()
        side = self.side_var.get()
        iterations = self.iter_var.get()
        sigma = self.sigma_var.get()
        sample_r = self.sample_var.get()
        edge_mode = self.edge_var.get()
        raw_scale = self.raw_scale_var.get()
        depth = self.depth_var.get()

        names = {"raw": "RAW马赛克", "raw_downscale": "RAW降采样",
                 "borrow": "邻居借用", "correct": "双边校正",
                 "correct_isp": "ISP去伪色", "sierpinski": "Sierpinski",
                 "superres2": "超分2x", "superres4": "超分4x",
                 "cv_lum": "CV亮度", "cv_edge": "CV边缘",
                 "cv_edge_ms": "CV多尺度边缘",
                 "cv_features": "CV特征点", "view3d": "3D视图",
                 "ai_denoise": "AI去噪",
                 "sensor": "传感器对比",
                 "ai_demosaic": "AI直出RGB"}
        self.status_var.set(f"处理中 ({names.get(mode, mode)})...")

        def update_prog(pct):
            self.root.after(0, lambda: self.progress.configure(value=pct))

        def run():
            try:
                if mode == "sierpinski":
                    result = self._process_sierpinski(side, depth, sample_r, edge_mode, update_prog)
                elif mode in ("superres2", "superres4"):
                    result = self._process_superres(side, mode, sample_r, edge_mode, update_prog)
                elif mode.startswith("cv_"):
                    result = self._process_cv(side, mode, sample_r, edge_mode, update_prog)
                elif mode == "view3d":
                    result = self._process_3d(side, sample_r, edge_mode, update_prog)
                elif mode == "ai_denoise":
                    result = self._process_ai(side, sample_r, edge_mode, update_prog)
                elif mode == "sensor":
                    result = self._process_sensor(side, update_prog)
                elif mode == "ai_demosaic":
                    result = self._process_ai_demosaic(side, sample_r, update_prog)
                else:
                    result = process_pipeline(
                        self.original_image, side, mode=mode,
                        correct_iterations=iterations,
                        correct_sigma=sigma,
                        sample_radius=sample_r,
                        edge_mode=edge_mode,
                        raw_downscale_factor=raw_scale,
                        progress_callback=update_prog,
                    )
                self.processed_image = result
                self.root.after(0, self._on_done)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _process_cv(self, side, mode, sample_r, edge_mode, progress):
        import math
        from triangle_engine import sample_single_channels, borrow_neighbors, render_triangles
        from triangle_cv import (
            estimate_luminance, detect_edges, detect_edges_multiscale,
            render_edges_overlay,
        )
        img = self.original_image.convert("RGB")
        W, H = img.size
        pixels = np.array(img).astype(np.float32)
        S = float(side)
        h = S * math.sqrt(3) / 2.0
        n_c = int(W / (S / 2.0)) + 3
        n_r = int(H / h) + 2
        progress(10)
        single = sample_single_channels(pixels, S, h, n_r, n_c, sample_radius=sample_r)
        if mode == "cv_lum":
            progress(40)
            lum = estimate_luminance(single, n_r, n_c)
            rgb = np.zeros((n_r, n_c, 3), dtype=np.uint8)
            for r in range(n_r):
                for c in range(n_c):
                    v = np.clip(lum[r, c], 0, 255)
                    rgb[r, c] = [int(v), int(v), int(v)]
            progress(70)
            result = render_triangles(rgb, S, h, n_r, n_c, W, H)
        elif mode == "cv_edge":
            progress(30)
            edge_map, _, _ = detect_edges(single, n_r, n_c, 15, 35)
            progress(60)
            borrowed = borrow_neighbors(single, n_r, n_c, edge_mode="mirror")
            base = render_triangles(borrowed, S, h, n_r, n_c, W, H)
            progress(80)
            result = render_edges_overlay(base, S, h, n_r, n_c, edge_map)
        else:  # cv_edge_ms or cv_features
            progress(20)
            if mode == "cv_edge_ms":
                fused, _, _ = detect_edges_multiscale(single, n_r, n_c, S, h, 3, 12, 30)
                progress(60)
                borrowed = borrow_neighbors(single, n_r, n_c, edge_mode="mirror")
                base = render_triangles(borrowed, S, h, n_r, n_c, W, H)
                progress(80)
                result = render_edges_overlay(base, S, h, n_r, n_c, fused)
            else:  # cv_features
                from triangle_features import detect_harris_keypoints, render_keypoints
                progress(30)
                kps = detect_harris_keypoints(single, n_r, n_c, k=0.04, threshold=0.005)
                progress(60)
                borrowed = borrow_neighbors(single, n_r, n_c, edge_mode="mirror")
                base = render_triangles(borrowed, S, h, n_r, n_c, W, H)
                progress(80)
                result = render_keypoints(base, S, h, n_r, n_c, kps[:150])
        progress(100)
        return result

    def _process_3d(self, side, sample_r, edge_mode, progress):
        import math, numpy as np
        from triangle_engine import sample_single_channels, borrow_neighbors
        from triangle_3d import TriMesh
        img = self.original_image.convert("RGB")
        W,H=img.size
        pixels=np.array(img).astype(np.float32)
        S=float(side);h=S*math.sqrt(3)/2
        nc=int(W/(S/2))+3;nr=int(H/h)+2
        progress(10)
        single=sample_single_channels(pixels,S,h,nr,nc,sample_radius=sample_r)
        progress(30)
        borrowed=borrow_neighbors(single,nr,nc,edge_mode=edge_mode)
        progress(50)
        mesh=TriMesh()
        mesh.from_triangle_grid(single,S,h,nr,nc,rgb_map=borrowed)
        progress(70)
        return mesh.render_view(800,600,azimuth=30,elevation=25)

    def _process_ai(self, side, sample_r, edge_mode, progress):
        import os, math, numpy as np
        from triangle_ai import TriGCN, build_features, denoise_image, build_adjacency
        img = self.original_image.convert("RGB")
        W,H=img.size
        S=float(side);h_val=S*math.sqrt(3)/2
        nc=int(W/(S/2))+3;nr=int(H/h_val)+2
        adj_ref = (nr, nc, S, h_val)
        wpath = "tri_gcn_weights.npz"
        if not os.path.exists(wpath):
            progress(5)
            from triangle_ai import train_denoiser
            model, _ = train_denoiser(
                ["test_edge.png"], triangle_side=side,
                epochs=30, progress_callback=lambda e,l: None)
        else:
            w = np.load(wpath)
            model = TriGCN()
            model.W1=w["W1"]; model.b1=w["b1"]
            model.W2=w["W2"]; model.b2=w["b2"]
            model.W3=w["W3"]; model.b3=w["b3"]
        progress(50)
        return denoise_image(model, self.original_image, adj_ref)

    def _process_sensor(self, side, progress):
        from triangle_sensor import TriangleSensor, render_comparison
        sensor = TriangleSensor(triangle_side=side, psf_sigma=0.5, iso=100)
        progress(10)
        result = sensor.compare_with_bayer(self.original_image,
                                           progress_callback=progress)
        return render_comparison(
            result['tri_rgb'], result['bayer_rgb'],
            result['tri_psnr'], result['bayer_psnr'],
            result['tri_ssim'], result['bayer_ssim'])

    def _process_ai_demosaic(self, side, sample_r, progress):
        import os, math, numpy as np
        from triangle_demosaic_ai import TriDemosaicGCN, demosaic_image, _build_adjacency
        from triangle_engine import sample_single_channels
        img = self.original_image.convert("RGB")
        W,H=img.size; S=float(side);h=S*math.sqrt(3)/2
        nc=int(W/(S/2))+3;nr=int(H/h)+2
        adj_ref=(nr,nc,S,h)
        wpath="tri_demosaic_weights.npz"
        progress(10)
        if not os.path.exists(wpath):
            from triangle_demosaic_ai import train_demosaic_gcn
            model,adj_ref,_=train_demosaic_gcn(["test_edge.png"],side,epochs=50)
        else:
            w=np.load(wpath)
            model=TriDemosaicGCN(hidden=32)
            model.W1=w["W1"];model.b1=w["b1"]
            model.W2=w["W2"];model.b2=w["b2"]
            model.W3=w["W3"];model.b3=w["b3"]
        progress(50)
        return demosaic_image(model,img,adj_ref)

    def _process_superres(self, side, mode, sample_r, edge_mode, progress):
        from triangle_superres import super_resolve
        zoom = 2 if mode == "superres2" else 4
        progress(0)
        result = super_resolve(
            self.original_image, triangle_side=side, zoom=zoom,
            correct_iterations=self.iter_var.get(),
            edge_sensitivity=self.sigma_var.get(),
            use_self_sim=True,
            progress_callback=progress,
        )
        return result

    def _process_sierpinski(self, side, depth, sample_r, edge_mode, progress):
        import math
        from triangle_engine import (
            sample_single_channels, borrow_neighbors, render_triangles,
        )
        img = self.original_image.convert("RGB")
        W, H = img.size
        pixels = np.array(img).astype(np.float32)
        S = float(side)
        h = S * math.sqrt(3) / 2.0
        n_cols = int(W / (S / 2.0)) + 3
        n_rows = int(H / h) + 2

        progress(15)
        single = sample_single_channels(pixels, S, h, n_rows, n_cols,
                                        sample_radius=sample_r)
        progress(40)
        borrowed = borrow_neighbors(single, n_rows, n_cols, edge_mode=edge_mode)
        progress(60)
        base = render_triangles(borrowed, S, h, n_rows, n_cols, W, H)
        progress(75)
        result = render_sierpinski_overlay(base, S, h, n_rows, n_cols,
                                           depth=depth, alpha=0.35,
                                           line_color=(255, 255, 255))
        progress(100)
        return result

    def _on_done(self):
        self.processing = False
        self.process_btn.configure(state="normal")
        self._update_preview(self.processed_image, self.processed_label)
        self.status_var.set("完成")

    def _on_error(self, msg):
        self.processing = False
        self.process_btn.configure(state="normal")
        self.progress["value"] = 0
        messagebox.showerror("处理错误", msg)
        self.status_var.set("处理失败")


def main():
    root = tk.Tk()
    app = TrianglePixelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
