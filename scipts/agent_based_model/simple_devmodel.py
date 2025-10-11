#!/usr/bin/env python3
"""
极简“新建地块”预测脚本（仅预测 2008 -> 2009 的新建）
- 只在 mask==1 且 land_use == NonUrban 的像元上预测新建（不在原有 urban 内预测）
- 自动对齐因子栅格到 land_use 参考栅格
- 输出：predicted_2009.tif (uint8: 0/1 with nodata=255)、utility_score.tif (float32)、development_probability.tif (float32)
"""
import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ----------------- 配置路径（按你的路径） -----------------
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

# 权重（单一开发商）
weights = np.array([
    -0.5, -0.75, -0.75, -0.5, -0.25,
     0.5,  0.5,   0.5,  -0.25, -0.25,
    -0.25, -0.25
], dtype=np.float32).reshape(-1, 1, 1)

# 阈值策略：按分位数（默认 80 -> 开发前20%）或绝对阈值
use_percentile = True
percentile_val = 95
absolute_threshold = None

# 输出
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)
out_pred_path = os.path.join(out_dir, "predicted_2009.tif")
out_util_path = os.path.join(out_dir, "utility_score.tif")
out_prob_path = os.path.join(out_dir, "development_probability.tif")

# nodata 值
nodata_val_uint8 = 255
util_nodata = -9999.0

# land_use 映射值（根据你的 mapping 修改）
URBAN_VAL = 1
NONURBAN_VAL = 0

# ----------------- 读取参考 land_use -----------------
try:
    with rasterio.open(raster_paths["land_use"]) as ref:
        land_use_arr = ref.read(1).astype(np.float32)
        profile = ref.profile.copy()
        height, width = land_use_arr.shape
        ref_transform = ref.transform
        ref_crs = ref.crs
except Exception as e:
    print(f"[ERROR] 无法读取 land_use: {raster_paths['land_use']}\n{e}")
    sys.exit(1)

# 把 land_use nodata 处理为 np.nan 保持可检测
if ref.nodata is not None:
    land_use_arr = np.where(land_use_arr == ref.nodata, np.nan, land_use_arr)

# ----------------- 读取 mask -----------------
try:
    with rasterio.open(raster_paths["mask"]) as src_mask:
        mask_arr = src_mask.read(1).astype(np.float32)
        if src_mask.nodata is not None:
            mask_arr = np.where(mask_arr == src_mask.nodata, np.nan, mask_arr)
except Exception as e:
    print(f"[ERROR] 无法读取 mask: {raster_paths['mask']}\n{e}")
    sys.exit(1)

# ----------------- 对齐读取函数 -----------------
def read_and_align(path):
    try:
        with rasterio.open(path) as src:
            src_arr = src.read(1).astype(np.float32)
            dst = np.empty((height, width), dtype=np.float32)
            dst[:] = np.nan
            reproject(
                source=src_arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
            )
            if src.nodata is not None:
                dst = np.where(dst == src.nodata, np.nan, dst)
            return dst
    except Exception as e:
        raise RuntimeError(f"读取/对齐失败: {path} ({e})")

# ----------------- 读取 v* 层 -----------------
v_keys = [k for k in raster_paths.keys() if k.startswith("v")]
if not v_keys:
    print("[WARN] 未找到 v* 层")
    sys.exit(1)

v_layers = []
for k in v_keys:
    p = raster_paths[k]
    print(f"[INFO] 读取并对齐 {k}: {p}")
    try:
        arr = read_and_align(p)
    except Exception as e:
        print(f"[ERROR] 读取 {k} 失败: {e}")
        sys.exit(1)
    v_layers.append(arr)

v_stack = np.stack(v_layers, axis=0)  # (n, H, W)
print(f"[INFO] 加载 {v_stack.shape[0]} 个因子，大小 {v_stack.shape[1:]}")

# ----------------- z-score 标准化 -----------------
means = np.nanmean(v_stack, axis=(1,2), keepdims=True)
stds = np.nanstd(v_stack, axis=(1,2), keepdims=True)
v_norm = np.where(stds == 0, 0.0, (v_stack - means) / stds)

# ----------------- 计算效用 U -----------------
n_factors = v_norm.shape[0]
if weights.shape[0] != n_factors:
    print(f"[WARN] 权重长度 {weights.shape[0]} 与 因子数 {n_factors} 不一致，调整权重长度。")
    if weights.shape[0] > n_factors:
        weights = weights[:n_factors]
    else:
        pad = np.zeros((n_factors - weights.shape[0], 1, 1), dtype=np.float32)
        weights = np.vstack([weights, pad])

U = np.nansum(v_norm * weights, axis=0)  # (H,W)

# ----------------- 限定“新建”候选：mask==1 且 land_use == NONURBAN -----------------
# 注意：land_use_arr 可能是 float（含 np.nan），用 == NONURBAN_VAL 比较时先确保非 nan
land_use_valid = np.isfinite(land_use_arr)  # 有意义的 land_use 像元
is_nonurban = land_use_valid & (land_use_arr == NONURBAN_VAL)
mask_ok = np.isfinite(mask_arr) & (mask_arr == 1)

# valid_mask = 只有在 mask==1 且 land_use==NonUrban 才作为“可新建”像元
valid_mask_candidates = mask_ok & is_nonurban

# 把 U 中非候选位置设为 np.nan，便于后续 percentile 计算和写出 nodata
U = np.where(valid_mask_candidates, U, np.nan)

# ----------------- 诊断打印 -----------------
u_min = float(np.nanmin(U)) if np.any(np.isfinite(U)) else float('nan')
u_max = float(np.nanmax(U)) if np.any(np.isfinite(U)) else float('nan')
u_mean = float(np.nanmean(U)) if np.any(np.isfinite(U)) else float('nan')
prop_pos = float(np.sum(U > 0) / np.sum(np.isfinite(U))) if np.sum(np.isfinite(U))>0 else 0.0

print(f"\nUtility range (on NonUrban & mask): {u_min:.6f} to {u_max:.6f}")
print(f"Mean utility (on NonUrban & mask): {u_mean:.6f}")
print(f"Proportion U>0 (on NonUrban & mask): {prop_pos:.6f}")

num_total = height * width
num_mask = int(np.sum(mask_ok))
num_nonurban = int(np.sum(is_nonurban))
num_candidates = int(np.sum(valid_mask_candidates))
print(f"\nPixel counts: total={num_total}, mask==1={num_mask}, nonurban={num_nonurban}, candidates(mask&nonurban)={num_candidates}")

# ----------------- 阈值选择 -----------------
if use_percentile:
    if np.sum(np.isfinite(U)) == 0:
        print("[WARN] 在候选像元上没有有效 U 值，退出。")
        sys.exit(1)
    thr = float(np.nanpercentile(U, percentile_val))
    print(f"[INFO] 使用分位阈值 {percentile_val} -> threshold = {thr:.6f}")
else:
    thr = 0.0 if absolute_threshold is None else float(absolute_threshold)
    print(f"[INFO] 使用绝对阈值 -> threshold = {thr:.6f}")

# ----------------- 构建输出数组：uint8 二值 + nodata（255） -----------------
out_uint8 = np.full((height, width), nodata_val_uint8, dtype=np.uint8)
# 对候选位置写 0/1
cand_idx = np.isfinite(U)
out_uint8[cand_idx] = (U[cand_idx] > thr).astype(np.uint8)

# development probability (linearly scaled over candidate U range), invalid -> util_nodata
U_min = float(np.nanmin(U)) if np.any(np.isfinite(U)) else 0.0
U_max = float(np.nanmax(U)) if np.any(np.isfinite(U)) else 1.0
if U_max > U_min:
    dev_prob = (U - U_min) / (U_max - U_min)
else:
    dev_prob = np.zeros_like(U)
dev_prob_filled = np.full((height, width), util_nodata, dtype=np.float32)
dev_prob_filled[np.isfinite(U)] = dev_prob[np.isfinite(U)].astype(np.float32)

# utility filled
util_filled = np.full((height, width), util_nodata, dtype=np.float32)
util_filled[np.isfinite(U)] = U[np.isfinite(U)].astype(np.float32)

# ----------------- 写出文件 -----------------
# predicted uint8 with nodata=255
prof_uint8 = profile.copy()
prof_uint8.update(dtype=rasterio.uint8, count=1, nodata=nodata_val_uint8)
with rasterio.open(out_pred_path, "w", **prof_uint8) as dst:
    dst.write(out_uint8, 1)
print(f"[OK] 写出 new-build 二值预测: {out_pred_path}")

# utility (float)
prof_util = profile.copy()
prof_util.update(dtype=rasterio.float32, count=1, nodata=util_nodata)
with rasterio.open(out_util_path, "w", **prof_util) as dst:
    dst.write(util_filled, 1)
print(f"[OK] 写出效用得分: {out_util_path}")

# development probability
with rasterio.open(out_prob_path, "w", **prof_util) as dst:
    dst.write(dev_prob_filled, 1)
print(f"[OK] 写出开发概率图: {out_prob_path}")

# ----------------- 写出后检查 -----------------
def print_counts(path, nodata_uint8=nodata_val_uint8):
    try:
        with rasterio.open(path) as src:
            a = src.read(1)
            vals, counts = np.unique(a, return_counts=True)
            print(f"\nValue counts for {os.path.basename(path)}:")
            for v, c in zip(vals, counts):
                label = None
                if src.dtypes[0].startswith('uint'):
                    if v == nodata_uint8:
                        label = "nodata"
                    elif v == 1:
                        label = "developed(1)"
                    elif v == 0:
                        label = "not_developed(0)"
                # print(f"  {v}: {c} px" + (f" -> {label}" if label else ""))
    except Exception as e:
        print(f"[WARN] cannot inspect {path}: {e}")

print_counts(out_pred_path)
print_counts(out_util_path)
print_counts(out_prob_path)

# ----------------- Summary -----------------
valid_count = int(np.sum(np.isfinite(U)))
dev_count = int(np.sum(out_uint8 == 1))
not_dev_count = int(np.sum(out_uint8 == 0))
nodata_count = int(np.sum(out_uint8 == nodata_val_uint8))

print(f"\nSummary (image {width}x{height}):")
print(f"  Candidates (mask==1 & NonUrban): {valid_count}")
print(f"  Developed (1): {dev_count}")
print(f"  Not developed (0): {not_dev_count}")
print(f"  nodata (255): {nodata_count}")
print(f"  Fraction developed (of candidates): {dev_count / max(1, valid_count):.2%}")

print("\nDone.")
