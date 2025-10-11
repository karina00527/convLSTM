#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_fixed_paths.py

将路径写死在脚本内，读取每个 raster，对齐 mask（mask==1 保留），
为每个 raster 绘制影像与直方图，并保存。最后生成合成网格图。
"""

import os
import math
from typing import Tuple, Optional, List

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

# ---------------------- 硬编码路径（按你提供的） ----------------------
raster_paths = {
    "mask": "/home/xyf/Downloads/landuse/datas/mask/mask.tif",
    "land_use": "ANN/2008urban.tif",  # 1 = Urban, 0 = NonUrban (按你 mapping)
    "v1": "/home/xyf/Downloads/landuse/datas/greenspace/greenspace_to2020.tif",
    "v2": "/home/xyf/Downloads/landuse/datas/fibre/fibre_to2020.tif",
    "v3": "/home/xyf/Downloads/landuse/datas/3water1000m.tif",
    "v4": "/home/xyf/Downloads/landuse/datas/floodrisk/floodrisk_to2020.tif",
    "v5": "/home/xyf/Downloads/landuse/datas/roads/roads_to2020.tif",
    # "v6": "ANN/income13.tif",
    # "v7": "ANN/income18.tif",
    "v8": "/home/xyf/Downloads/landuse/datas/sitesize.tif",
    "v9": "/home/xyf/Downloads/landuse/datas/busstop_to2020.tif",
    "v10": "/home/xyf/Downloads/landuse/datas/school_to2020.tif",
    "v11": "/home/xyf/Downloads/landuse/datas/hospital_to2020.tif",
    "v12": "/home/xyf/Downloads/landuse/datas/shoppingmall/shoppingmal_to2020.tif",
}

OUT_DIR = "out"  # 输出目录（PNG 等）
MASK_VALUE = 1   # mask==1 表示保留

# ---------------------- 工具函数 ----------------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def read_and_align_to_ref(ref_ds: rasterio.io.DatasetReader, src_path: str,
                          resampling=Resampling.nearest) -> Tuple[np.ndarray, Optional[float]]:
    """
    读取 src_path 的第1波段并对齐到 ref_ds（重投影/重采样如需）。
    返回 (arr (float), src_nodata)。arr 中 nodata 被替换为 np.nan。
    """
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype(float)
        src_nd = src.nodata

        same_crs = (src.crs == ref_ds.crs)
        same_grid = (src.transform == ref_ds.transform) and (src.width == ref_ds.width) and (src.height == ref_ds.height)
        if same_crs and same_grid:
            arr = src_arr
        else:
            dst = np.full((ref_ds.height, ref_ds.width), np.nan, dtype=float)
            try:
                reproject(
                    source=src_arr,
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_ds.transform,
                    dst_crs=ref_ds.crs,
                    resampling=resampling,
                    num_threads=1,
                )
                arr = dst
                print(f"[INFO] 重投影对齐: {src_path}")
            except Exception as e:
                raise RuntimeError(f"无法重投影 {src_path} 到参考网格: {e}")

        if src_nd is not None:
            arr = np.where(arr == src_nd, np.nan, arr)
        return arr, src_nd

# 单文件绘图函数
def plot_single_raster_with_mask(raster_path: str, mask_path: str, mask_value=1,
                                 out_dir: str = "out", save_png: bool = True,
                                 cmap='viridis', vmin=None, vmax=None):
    base_name = os.path.basename(raster_path)
    base_noext = os.path.splitext(base_name)[0]
    out_png_img = os.path.join(out_dir, f"{base_noext}_masked.png")
    out_png_hist = os.path.join(out_dir, f"{base_noext}_hist.png")

    with rasterio.open(raster_path) as ds:
        arr = ds.read(1).astype(float)
        nd = ds.nodata
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)

        # 对齐 mask 到参考栅格 ds
        try:
            mask_aligned, mask_nd = read_and_align_to_ref(ds, mask_path, resampling=Resampling.nearest)
            mask_aligned = np.where(np.isfinite(mask_aligned), mask_aligned, np.nan)
        except Exception as e:
            print(f"[WARN] 无法读取或重投影 mask: {e}. 将不应用 mask.")
            mask_aligned = None

        # 应用 mask（仅保留 mask==mask_value）
        if mask_aligned is not None:
            mask_ok = np.isfinite(mask_aligned) & (mask_aligned == mask_value)
            arr_masked = np.where(mask_ok, arr, np.nan)
            kept = int(np.sum(mask_ok))
            total_mask_pixels = int(np.sum(np.isfinite(mask_aligned)))
        else:
            arr_masked = arr
            kept = int(np.sum(np.isfinite(arr_masked)))
            total_mask_pixels = int(arr_masked.size)

        # 打印统计信息
        finite_total = int(np.sum(np.isfinite(arr)))
        finite_masked = int(np.sum(np.isfinite(arr_masked)))
        print(f"File: {raster_path}")
        print(f"  Size: {arr.shape}, finite pixels (raw): {finite_total}, after mask finite: {finite_masked}")
        if mask_aligned is not None:
            print(f"  Mask finite pixels: {total_mask_pixels}, Mask=={mask_value} kept: {kept}")

        # 绘影像
        plt.figure(figsize=(8,6))
        finite_vals = arr_masked[np.isfinite(arr_masked)]
        if finite_vals.size > 0:
            auto_vmin = np.nanmin(finite_vals) if vmin is None else vmin
            auto_vmax = np.nanmax(finite_vals) if vmax is None else vmax
        else:
            auto_vmin = np.nanmin(arr) if vmin is None else vmin
            auto_vmax = np.nanmax(arr) if vmax is None else vmax

        im = plt.imshow(arr_masked, cmap=cmap, vmin=auto_vmin, vmax=auto_vmax)
        plt.colorbar(im, label='Value')
        plt.title(f"{base_noext} (masked: mask=={mask_value})")
        plt.axis('off')
        if save_png:
            plt.savefig(out_png_img, bbox_inches='tight', dpi=200)
            print(f"[OK] Saved image: {out_png_img}")
        plt.close()

        # 直方图（mask 保留像元）
        if finite_vals.size > 0:
            plt.figure(figsize=(6,4))
            plt.hist(finite_vals.ravel(), bins=50)
            plt.xlabel('Value'); plt.ylabel('Count'); plt.title(f'Histogram (masked pixels only) - {base_noext}')
            if save_png:
                plt.savefig(out_png_hist, bbox_inches='tight', dpi=150)
                print(f"[OK] Saved histogram: {out_png_hist}")
            plt.close()
        else:
            print("[WARN] 掩码后无有效像元，跳过直方图。")

    return arr_masked

def plot_all_grid(arrs: List[np.ndarray], paths: List[str], out_dir: str = "out",
                  per_row: int = 3, cmap='viridis'):
    n = len(arrs)
    if n == 0:
        print("[WARN] 无输入要绘制。")
        return
    cols = per_row
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    axes = np.array(axes).reshape(-1)
    for i, (arr, p) in enumerate(zip(arrs, paths)):
        ax = axes[i]
        base_noext = os.path.splitext(os.path.basename(p))[0]
        finite_vals = arr[np.isfinite(arr)]
        if finite_vals.size > 0:
            vmin = np.nanmin(finite_vals); vmax = np.nanmax(finite_vals)
        else:
            vmin = None; vmax = None
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(base_noext)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
    # 清空多余子图
    for j in range(n, rows*cols):
        axes[j].axis('off')

    out_combined = os.path.join(out_dir, "combined_grid.png")
    plt.tight_layout()
    plt.savefig(out_combined, dpi=200)
    plt.close()
    print(f"[OK] Saved combined grid: {out_combined}")

# ---------------------- 主运行 ----------------------
def main():
    ensure_outdir(OUT_DIR)

    # 要绘制的 raster 列表（只包含 v* 层以及 land_use，如需可以调整）
    keys_order = ["land_use", "v1", "v2", "v3", "v4", "v5", "v8", "v9", "v10", "v11", "v12"]
    raster_list = []
    for k in keys_order:
        p = raster_paths.get(k)
        if p:
            raster_list.append(p)
        else:
            print(f"[INFO] Skip missing key {k}")

    masked_arrays = []
    used_paths = []
    for pth in raster_list:
        if not os.path.exists(pth):
            print(f"[ERROR] 文件不存在，跳过: {pth}")
            continue
        arr_masked = plot_single_raster_with_mask(
            raster_path=pth,
            mask_path=raster_paths["mask"],
            mask_value=MASK_VALUE,
            out_dir=OUT_DIR,
            save_png=True,
            cmap='viridis'
        )
        masked_arrays.append(arr_masked)
        used_paths.append(pth)

    # 生成合成网格图
    if masked_arrays:
        plot_all_grid(masked_arrays, used_paths, out_dir=OUT_DIR, per_row=3)
    else:
        print("[WARN] 没有可用栅格生成合成图。")

if __name__ == "__main__":
    main()
