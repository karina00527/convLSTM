#!/usr/bin/env python3
"""
compute_contribution.py

计算每个 v* 因子在候选像元（mask==1 & land_use==NonUrban）上的贡献：
contrib_i = weight_i * zscore(value_i)  （zscore 在候选像元上计算 mean/std）

输出：
- output/contrib_summary.csv
- output/contrib_bar.png
- output/contrib_{factor}.tif （每因子贡献栅格）
- output/utility_contrib_sum.tif （贡献之和）
"""
import os
import sys
import math
import csv
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from collections import deque
import matplotlib.pyplot as plt

# ----------------- 配置（写死路径） -----------------
raster_paths = {
    "mask": "/home/xyf/Downloads/landuse/datas/mask/mask.tif",
    "land_use": "ANN/2008urban.tif",  # 1 = Urban, 0 = NonUrban
    # 请按需注释/启用 v* 层（保持与模型一致）
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

# 原始 12 因子权重（参考）
orig_weights_flat = np.array([
    -0.5, -0.75, -0.75, -0.5, -0.25,
     0.5,  0.5,   0.5,  -0.25, -0.25,
    -0.25, -0.25
], dtype=np.float32)
all_v_names = [f"v{i}" for i in range(1,13)]

# distance-to-urban 参数
DIST_WEIGHT = 0.5          # 给 distance 因子的权重
LAMBDA_DIST = 1.0 / 500.0  # 衰减率（单位同栅格坐标，针对像素实际米数可调整）

# winsorize / log1p 开关（与模型一致）
APPLY_WINSORIZE = False
APPLY_LOG1P_IF_SKEWED = True

# 输出文件夹
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# nodata for float rasters
util_nodata = -9999.0

# land_use mapping values
URBAN_VAL = 1
NONURBAN_VAL = 0

# ----------------- 读取 land_use 作为参考 -----------------
try:
    with rasterio.open(raster_paths["land_use"]) as ref:
        land_use_arr = ref.read(1).astype(np.float32)
        profile = ref.profile.copy()
        height, width = land_use_arr.shape
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_nodata = ref.nodata
except Exception as e:
    print(f"[ERROR] 无法读取 land_use: {raster_paths['land_use']}\n{e}")
    sys.exit(1)

if ref_nodata is not None:
    land_use_arr = np.where(land_use_arr == ref_nodata, np.nan, land_use_arr)

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
    if not path:
        return None
    try:
        with rasterio.open(path) as src:
            src_arr = src.read(1).astype(np.float32)
            dst = np.full((height, width), np.nan, dtype=np.float32)
            reproject(
                source=src_arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
                num_threads=1,
            )
            if src.nodata is not None:
                dst = np.where(dst == src.nodata, np.nan, dst)
            return dst
    except Exception as e:
        raise RuntimeError(f"读取/对齐失败: {path} ({e})")

# ----------------- 候选掩码 -----------------
land_use_valid = np.isfinite(land_use_arr)
is_nonurban = land_use_valid & (land_use_arr == NONURBAN_VAL)
mask_ok = np.isfinite(mask_arr) & (mask_arr == 1)
valid_mask_candidates = mask_ok & is_nonurban
print(f"[INFO] image size: {width}x{height}, candidates(mask==1 & NonUrban) = {int(np.sum(valid_mask_candidates))}")

# ----------------- 加载所有 v* 层 -----------------
v_keys = [k for k in raster_paths.keys() if k.startswith("v")]
if not v_keys:
    print("[ERROR] 未找到 v* 层，退出")
    sys.exit(1)

v_layers = []
loaded_keys = []
for k in v_keys:
    p = raster_paths[k]
    print(f"[INFO] 读取并对齐 {k}: {p}")
    arr = read_and_align(p)
    v_layers.append(arr)
    loaded_keys.append(k)

v_stack = np.stack(v_layers, axis=0)  # shape (n, H, W)
n_factors = v_stack.shape[0]
print(f"[INFO] Loaded {n_factors} factors -> keys: {loaded_keys}")

# ----------------- 计算 distance-to-urban（多源 BFS），并加入因子 -----------------
urban_mask = np.isfinite(land_use_arr) & (land_use_arr == URBAN_VAL)
if np.any(urban_mask):
    print("[INFO] 计算到最近 urban 的网格距离（BFS，多源）...")
    dist_cells = np.full((height, width), np.inf, dtype=float)
    q = deque()
    for r in range(height):
        for c in range(width):
            if urban_mask[r, c]:
                dist_cells[r, c] = 0.0
                q.append((r, c))
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    while q:
        r, c = q.popleft()
        d0 = dist_cells[r, c]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if dist_cells[nr, nc] > d0 + 1:
                    dist_cells[nr, nc] = d0 + 1
                    q.append((nr, nc))
    pixel_size = max(abs(ref_transform.a), abs(ref_transform.e))
    dist_m = dist_cells * pixel_size
    proximity = np.exp(-LAMBDA_DIST * dist_m)  # 越靠近 urban 值越接近 1
else:
    print("[WARN] 没有 urban 像元，distance 因子全 0")
    proximity = np.zeros((height, width), dtype=float)

# append
v_stack = np.vstack([v_stack, proximity[None, ...]])
loaded_keys.append("dist_to_urban")
n_factors = v_stack.shape[0]
print(f"[INFO] Added dist_to_urban -> total factors = {n_factors}")

# ----------------- 构建权重（按 loaded_keys 顺序） -----------------
weights_list = []
for k in loaded_keys:
    if k in all_v_names:
        idx = all_v_names.index(k)
        w = float(orig_weights_flat[idx])
    elif k == "dist_to_urban":
        w = float(DIST_WEIGHT)
    else:
        w = 0.0
    weights_list.append(w)
weights = np.array(weights_list, dtype=np.float32).reshape(n_factors, 1, 1)
print(f"[INFO] weights (loaded_keys order): {weights_list}")

# ----------------- 稳健化（winsorize 可选 & log1p 可选） -----------------
def winsorize_v_stack(vs, mask2d=None, p_low=0.01, p_high=0.99):
    out = vs.copy()
    for i in range(out.shape[0]):
        arr = out[i]
        if mask2d is not None:
            sel = mask2d & np.isfinite(arr)
            vals = arr[sel]
        else:
            vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            continue
        lo = float(np.nanpercentile(vals, p_low*100))
        hi = float(np.nanpercentile(vals, p_high*100))
        arr = np.where(arr < lo, lo, arr)
        arr = np.where(arr > hi, hi, arr)
        out[i] = arr
        print(f"[WINSORIZE] layer {i} -> {p_low*100:.1f}p={lo:.4g}, {p_high*100:.1f}p={hi:.4g}")
    return out

def log_transform_if_skewed(vs, mask2d=None, skew_thr=3.0):
    out = vs.copy()
    for i in range(out.shape[0]):
        arr = out[i]
        if mask2d is not None:
            sel = mask2d & np.isfinite(arr)
            vals = arr[sel]
        else:
            vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            continue
        med = float(np.nanmedian(vals))
        mu = float(np.nanmean(vals))
        ratio = (abs(mu / med) if med != 0 else np.inf)
        minv = float(np.nanmin(vals))
        if ratio > skew_thr and minv >= 0:
            out[i] = np.log1p(np.where(np.isfinite(arr), np.maximum(arr, 0.0), np.nan))
            print(f"[LOG1P] layer {i} skew ratio={ratio:.2f}, applied log1p")
    return out

if APPLY_WINSORIZE:
    v_stack = winsorize_v_stack(v_stack, mask2d=valid_mask_candidates, p_low=0.01, p_high=0.99)
else:
    print("[INFO] winsorize disabled")

if APPLY_LOG1P_IF_SKEWED:
    v_stack = log_transform_if_skewed(v_stack, mask2d=valid_mask_candidates, skew_thr=3.0)
else:
    print("[INFO] log1p transform disabled")

# ----------------- z-score（仅在候选像元上计算 mean/std） -----------------
v_masked_stats = np.where(valid_mask_candidates[None, ...], v_stack, np.nan)
means = np.nanmean(v_masked_stats, axis=(1,2), keepdims=True)
stds = np.nanstd(v_masked_stats, axis=(1,2), keepdims=True)
stds_safe = np.where(np.isfinite(stds) & (stds > 0), stds, 1.0)
v_norm = (v_stack - means) / stds_safe

# fill empty layers with zeros to avoid NaN propagation
for i in range(v_norm.shape[0]):
    if not np.any(np.isfinite(v_norm[i])):
        v_norm[i] = np.zeros((height, width), dtype=float)
        print(f"[INFO] normalized layer {i} has no finite values in candidates -> fill with zeros")

# ----------------- 计算贡献 -----------------
# per-factor contribution map (n_factors, H, W)
contrib_stack = v_norm * weights  # broadcasting
utility_map_all = np.nansum(contrib_stack, axis=0)  # total utility (all pixels, but we will mask later)

# mask to candidate pixels
utility_map = np.where(valid_mask_candidates, utility_map_all, np.nan)
contrib_stack_masked = np.where(valid_mask_candidates[None, ...], contrib_stack, np.nan)

# ----------------- 汇总每个因子贡献（在候选像元上） -----------------
summary = []
for i, k in enumerate(loaded_keys):
    arr = contrib_stack_masked[i]
    finite_sel = np.isfinite(arr)
    cnt = int(np.sum(finite_sel))
    if cnt == 0:
        mean_c = 0.0
        mean_abs = 0.0
        total = 0.0
        pos_frac = 0.0
    else:
        mean_c = float(np.nanmean(arr))
        mean_abs = float(np.nanmean(np.abs(arr)))
        total = float(np.nansum(arr))
        pos_frac = float(np.sum(arr > 0) / cnt)
    summary.append({
        "factor": k,
        "weight": float(weights_list[i]) if 'weights_list' in locals() else float(weights[i]),
        "mean_contrib": mean_c,
        "mean_abs_contrib": mean_abs,
        "total_contrib": total,
        "pos_fraction": pos_frac,
        "n_pixels": cnt,
    })

# sort by mean_abs_contrib desc
summary_sorted = sorted(summary, key=lambda x: x["mean_abs_contrib"], reverse=True)

# write CSV
csv_path = os.path.join(OUT_DIR, "contrib_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["factor","weight","mean_contrib","mean_abs_contrib","total_contrib","pos_fraction","n_pixels"])
    writer.writeheader()
    for row in summary_sorted:
        writer.writerow(row)
print(f"[OK] writen summary -> {csv_path}")

# ----------------- 绘制条形图（mean_abs_contrib） -----------------
factors = [r["factor"] for r in summary_sorted]
vals = [r["mean_abs_contrib"] for r in summary_sorted]

plt.figure(figsize=(10, max(4, len(factors)*0.4)))
y_pos = np.arange(len(factors))
plt.barh(y_pos, vals, align='center')
plt.yticks(y_pos, factors)
plt.xlabel("Mean absolute contribution (|weight * zscore|)")
plt.title("Factor contributions (mean absolute over candidate pixels)")
plt.gca().invert_yaxis()
plt.tight_layout()
bar_path = os.path.join(OUT_DIR, "contrib_bar.png")
plt.savefig(bar_path, dpi=200)
plt.close()
print(f"[OK] saved bar plot -> {bar_path}")

# ----------------- 写出每个因子的贡献栅格 & utility sum 栅格 -----------------
prof = profile.copy()
prof.update(dtype=rasterio.float32, count=1, nodata=util_nodata)

# per-factor rasters
for i, k in enumerate(loaded_keys):
    out_arr = np.full((height, width), util_nodata, dtype=np.float32)
    mask_fin = np.isfinite(contrib_stack_masked[i])
    out_arr[mask_fin] = contrib_stack_masked[i][mask_fin].astype(np.float32)
    out_fp = os.path.join(OUT_DIR, f"contrib_{k}.tif")
    with rasterio.open(out_fp, "w", **prof) as dst:
        dst.write(out_arr, 1)
    print(f"[OK] wrote {out_fp}")

# utility sum
out_sum = np.full((height, width), util_nodata, dtype=np.float32)
mask_fin_sum = np.isfinite(utility_map)
out_sum[mask_fin_sum] = utility_map[mask_fin_sum].astype(np.float32)
out_sum_fp = os.path.join(OUT_DIR, "utility_contrib_sum.tif")
with rasterio.open(out_sum_fp, "w", **prof) as dst:
    dst.write(out_sum, 1)
print(f"[OK] wrote utility sum -> {out_sum_fp}")

# ----------------- 简要打印 summary -----------------
print("\nTop contributions (mean_abs):")
for r in summary_sorted[:20]:
    print(f"  {r['factor']:12s} | mean_contrib={r['mean_contrib']:.4g}, mean_abs={r['mean_abs_contrib']:.4g}, pos_frac={r['pos_fraction']:.3f}, pixels={r['n_pixels']}")

print("\nDone.")
