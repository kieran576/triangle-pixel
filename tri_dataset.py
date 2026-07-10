#!/usr/bin/env python3
"""
tri_dataset.py — 三角 RAW 数据集类 (PyTorch)

从 tri_convert.py 产出的 .tri 文件加载数据,
支持图片模式和视频帧序列模式,
兼容 PyTorch DataLoader。

用法:
    from tri_dataset import TriDataset

    ds = TriDataset("./tri_output/", mode="image")
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    # 视频时序模式
    ds = TriDataset("./tri_output/dance_video/", mode="video", temporal_window=8)
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# 导入三角引擎
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triangle_engine import (
    assigned_channel, borrow_neighbors, correct_triangular_isp,
    sample_single_channels, tri_center,
)


class TriDataset(Dataset):
    """
    三角 RAW 数据集。

    mode="image": 每张图片独立, __getitem__ 返回 (raw, gt_rgb, meta)
    mode="video": 按 temporal_window 取连续帧, 返回 (frames, gt_frames)

    gt_rgb 通过 borrow_neighbors + ISP 从 RAW 重建得到
    (因为是模拟数据, 没有独立的 ground truth 传感器)
    """

    def __init__(
        self,
        data_dir,
        mode="image",
        temporal_window=8,
        frame_stride=1,
        normalize=True,
        return_gt=True,
    ):
        """
        Args:
            data_dir: .tri 文件目录
            mode: "image" | "video"
            temporal_window: 视频模式下的连续帧数
            frame_stride: 视频模式下帧之间的步长
            normalize: 归一化到 [0,1]
            return_gt: 是否返回 ground truth RGB
        """
        self.data_dir = data_dir
        self.mode = mode
        self.temporal_window = temporal_window
        self.frame_stride = frame_stride
        self.normalize = normalize
        self.return_gt = return_gt

        # 收集 .npy 文件
        self.raw_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        )

        if mode == "video":
            # 过滤掉非帧文件 (不以 "frame_" 开头的)
            self.raw_files = [f for f in self.raw_files if f.startswith("frame_")]
            self._video_meta = self._load_video_meta()

        if not self.raw_files:
            raise ValueError(f"未找到 .npy 文件: {data_dir}")

        # 从第一个文件加载网格参数 (所有文件共享同一网格)
        self._load_grid_params()

    def _load_video_meta(self):
        """加载视频元信息."""
        meta_path = os.path.join(self.data_dir, "_video.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_grid_params(self):
        """从第一个 .npy 文件的 meta 加载网格参数."""
        f0 = self.raw_files[0]
        meta_path = os.path.join(self.data_dir, f0.replace(".npy", ".json"))
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.n_rows = meta.get("n_rows", 0)
            self.n_cols = meta.get("n_cols", 0)
            self.S = meta.get("S", 12)
            self.h = meta.get("h", self.S * math.sqrt(3) / 2.0)
        else:
            # 从 .npy shape 推算
            raw0 = np.load(os.path.join(self.data_dir, f0))
            self.n_rows, self.n_cols = raw0.shape
            self.S = 12  # 默认
            self.h = self.S * math.sqrt(3) / 2.0

    def __len__(self):
        if self.mode == "video":
            # 视频: 每个起始帧位置是一个样本
            stride = self.frame_stride
            window = self.temporal_window
            total = len(self.raw_files)
            return max(0, (total - window * stride) // stride + 1)
        return len(self.raw_files)

    def __getitem__(self, idx):
        if self.mode == "video":
            return self._get_video_item(idx)
        return self._get_image_item(idx)

    def _get_image_item(self, idx):
        """图片模式: 返回单个三角 RAW + GT RGB."""
        fname = self.raw_files[idx]
        raw_path = os.path.join(self.data_dir, fname)
        raw = np.load(raw_path).astype(np.float32)

        # 重建 GT RGB (borrow + ISP)
        if self.return_gt:
            gt = self._reconstruct_gt(raw)
            if self.normalize:
                gt = gt / 255.0

        if self.normalize:
            raw = raw / 255.0

        raw_tensor = torch.from_numpy(raw).unsqueeze(0)  # [1, H, W]

        if self.return_gt:
            gt_tensor = torch.from_numpy(gt).permute(2, 0, 1)  # [3, H, W]
            return raw_tensor, gt_tensor
        return raw_tensor

    def _get_video_item(self, idx):
        """视频模式: 返回 temporal_window 连续帧."""
        start = idx * self.frame_stride
        indices = range(start, start + self.temporal_window * self.frame_stride,
                        self.frame_stride)

        raw_frames = []
        gt_frames = []

        for i in indices:
            fname = self.raw_files[i]
            raw = np.load(os.path.join(self.data_dir, fname)).astype(np.float32)

            if self.normalize:
                raw = raw / 255.0
            raw_frames.append(torch.from_numpy(raw))

            if self.return_gt:
                gt = self._reconstruct_gt(raw * 255.0 if self.normalize else raw)
                if self.normalize:
                    gt = gt / 255.0
                gt_frames.append(torch.from_numpy(gt).permute(2, 0, 1))

        raw_tensor = torch.stack(raw_frames)  # [T, H, W]

        if self.return_gt:
            gt_tensor = torch.stack(gt_frames)  # [T, 3, H, W]
            return raw_tensor, gt_tensor
        return raw_tensor

    def _reconstruct_gt(self, raw):
        """从三角 RAW 重建全彩 RGB (borrow + ISP)."""
        nr, nc = self.n_rows, self.n_cols
        borrowed = borrow_neighbors(raw, nr, nc, edge_mode="mirror")
        corrected = correct_triangular_isp(
            raw, borrowed, nr, nc,
            iterations=3,
            edge_sensitivity=40,
        )
        return corrected  # [n_rows, n_cols, 3]

    def get_grid_params(self):
        """返回网格参数, 供模型使用."""
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "S": self.S,
            "h": self.h,
        }

    @classmethod
    def from_directories(cls, dirs, mode="image", split_ratio=(0.7, 0.15, 0.15), **kwargs):
        """
        从多个目录加载并自动划分 train/val/test。

        Args:
            dirs: 目录列表
            mode: "image" | "video"
            split_ratio: (train, val, test) 比例

        Returns:
            (train_ds, val_ds, test_ds) 或单个 TriDataset
        """
        all_items = []
        for d in dirs:
            ds = cls(d, mode=mode, **kwargs)
            all_items.append(ds)

        # 合并所有文件的索引信息
        # 简单实现: 把所有 .npy 文件路径合并到一个临时目录
        import tempfile
        import shutil

        tmpdir = tempfile.mkdtemp(prefix="tri_dataset_")
        for ds in all_items:
            for f in ds.raw_files:
                src_raw = os.path.join(ds.data_dir, f)
                src_meta = f.replace(".npy", ".json")
                src_meta = os.path.join(ds.data_dir, src_meta)
                shutil.copy(src_raw, os.path.join(tmpdir, f))
                if os.path.exists(src_meta):
                    shutil.copy(src_meta, os.path.join(tmpdir, src_meta))

        # 创建合并数据集
        merged = cls(tmpdir, mode=mode, **kwargs)

        # 简单划分
        n = len(merged)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        # 用索引切片实现划分 (修改 raw_files 列表)
        import copy
        train_ds = copy.copy(merged)
        train_ds.raw_files = merged.raw_files[:n_train]

        val_ds = copy.copy(merged)
        val_ds.raw_files = merged.raw_files[n_train:n_train + n_val]

        test_ds = copy.copy(merged)
        test_ds.raw_files = merged.raw_files[n_train + n_val:]

        return train_ds, val_ds, test_ds


# ============================================================
#  测试
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TriDataset 测试")
    parser.add_argument("data_dir", help=".tri 文件目录")
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    parser.add_argument("--temporal-window", type=int, default=8)
    args = parser.parse_args()

    print(f"加载: {args.data_dir}")
    ds = TriDataset(args.data_dir, mode=args.mode,
                    temporal_window=args.temporal_window)
    print(f"  样本数: {len(ds)}")
    print(f"  网格: {ds.n_rows}x{ds.n_cols}, S={ds.S}")

    x = ds[0]
    if isinstance(x, tuple):
        raw, gt = x
        print(f"  RAW: {raw.shape}, GT: {gt.shape}")
    else:
        print(f"  RAW: {x.shape}")
