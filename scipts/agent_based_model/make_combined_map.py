#!/usr/bin/env python3
"""
combined_three_values.py

生成一个单波段 uint8 GeoTIFF，值含义：
  1 = mask baseline (mask == 1)
  2 = existing Urban (land_use == 1)
  3 = predicted new-build (pred == 1)
  255 = nodata (outside study area)

输入（请按你的实际路径修改下方三个变量）：
  pred_path, land_use_path, mask_path
输出：
  output/combined_three_values.tif
"""
import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ---------------- CONFIG: 改成你的路径 ----------------
pred_path = "output/predicted_2009.tif"
land_use_path = "ANN/2008urban.tif"   # 1 = Urban, 0 = NonUrban (按你 mapping)
mask_path = "/home/xyf/Downloads/landuse/datas/mask/mask.tif"

out_dir = "output"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "combined_three_values.tif")

# 编码与 nodata
VAL_MASK = 0
VAL_URBAN = 1
VAL_PRED_NEW = 2
NODATA = 255  # uint8 nodata

# ---------------- helper: read & reproject to reference grid ----------------
def read_aligned_to_ref(ref_profile, src_path):
    """读 src_path 并重投影到 ref_profile 的网格，返回 float32 数组（nodata -> np.nan）"""
    height = ref_profile['height']
    width = ref_profile['width']
    ref_transform = ref_profile['transform']
    ref_crs = ref_profile['crs']

    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype(np.float32)
        dst = np.full((height, width), np.nan, dtype=np.float32)
        try:
            reproject(
                source=src_arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
            )
        except Exception as e:
            raise RuntimeError(f"Reproject failed for {src_path}: {e}")
        if src.nodata is not None:
            dst = np.where(dst == src.nodata, np.nan, dst)
        return dst

# ---------------- read reference (land_use) ----------------
try:
    with rasterio.open(land_use_path) as ref:
        land_use = ref.read(1).astype(np.float32)
        profile = ref.profile.copy()
        height, width = land_use.shape
        # normalize ref nodata -> np.nan
        if ref.nodata is not None:
            land_use = np.where(land_use == ref.nodata, np.nan, land_use)
except Exception as e:
    print(f"[ERROR] Cannot read land_use reference: {land_use_path}\n{e}")
    sys.exit(1)

# ---------------- read and align mask and pred ----------------
try:
    mask = read_aligned_to_ref(profile, mask_path)
except Exception as e:
    print(f"[ERROR] Cannot read/align mask: {mask_path}\n{e}")
    sys.exit(1)

try:
    pred = read_aligned_to_ref(profile, pred_path)
except Exception as e:
    print(f"[ERROR] Cannot read/align predicted raster: {pred_path}\n{e}")
    sys.exit(1)

# ---------------- build combined array ----------------
combined = np.full((height, width), NODATA, dtype=np.uint8)

# mask baseline: mask == 1 (finite)
mask_ok = np.isfinite(mask) & (mask == 1)
combined[mask_ok] = VAL_MASK

# existing urban: land_use == 1 (and within mask ideally; but we'll mark urban anywhere in land_use)
land_use_valid = np.isfinite(land_use)
urban_mask = land_use_valid & (land_use == 1)
# If you prefer only mark urban inside mask, intersect with mask_ok:
urban_mask_in_mask = urban_mask & mask_ok
# Mark urban (use urban_mask_in_mask so urban inside mask overrides baseline)
combined[urban_mask_in_mask] = VAL_URBAN

# predicted new-build: pred == 1 (only consider finite pred and inside mask and not already urban)
pred_new_mask = np.isfinite(pred) & (pred == 1) & mask_ok & (~urban_mask_in_mask)
combined[pred_new_mask] = VAL_PRED_NEW

# ---------------- write output GeoTIFF (uint8, nodata=NODATA) ----------------
out_profile = profile.copy()
out_profile.update(dtype=rasterio.uint8, count=1, nodata=NODATA, compress='lzw')

with rasterio.open(out_path, "w", **out_profile) as dst:
    dst.write(combined, 1)

print(f"[OK] Wrote combined map to: {out_path}")

# ---------------- print counts for verification ----------------
total = height * width
cnt_nodata = int(np.sum(combined == NODATA))
cnt_mask = int(np.sum(combined == VAL_MASK))
cnt_urban = int(np.sum(combined == VAL_URBAN))
cnt_pred = int(np.sum(combined == VAL_PRED_NEW))

print("\nCounts:")
print(f"  total pixels: {total}")
print(f"  nodata (255): {cnt_nodata}")
print(f"  mask baseline (1): {cnt_mask}")
print(f"  existing urban in mask (2): {cnt_urban}")
print(f"  predicted new-build (3): {cnt_pred}")
if np.sum(mask_ok) > 0:
    print(f"  fraction predicted new-build (of mask): {cnt_pred / max(1, np.sum(mask_ok)):.2%}")

# ---------------- small diagnostics: show whether pred==1 occurs in urban ----------------
pred1_in_urban = int(np.sum(pred == 1 & urban_mask_in_mask))
print(f"\npred==1 pixels overlapping urban (should be 0 if pred only on NonUrban): {pred1_in_urban}")

print("\nLegend: 1=mask, 2=urban, 3=predicted new-build, 255=nodata")
print("Done.")
