#!/usr/bin/env python3
"""
极简“新建地块”预测脚本（仅预测 2008 -> 2009 的新建）
- winsorize 可选（默认关闭），保留偏态 log1p 变换
- 增加 distance-to-urban 因子（靠近城市提升效用）
- 只在 mask==1 且 land_use == NonUrban 的像元上预测
输出：
- predicted_2009.tif (uint8: 0/1 with nodata=255)
- utility_score.tif (float32)
- development_probability.tif (float32)
"""
import os
import sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from collections import deque

# ----------------- 配置路径（按你的路径） -----------------
raster_paths = {
    "mask": "/home/xyf/Downloads/landuse/datas/mask/mask.tif",
    "land_use": "ANN/2008urban.tif",  # 1 = Urban, 0 = NonUrban (按你 mapping)
    # 你可以按需注释/启用 v* 层
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

# 原始 12 因子权重（对应 v1..v12 顺序），仅作默认参考
orig_weights_flat = np.array([
    -0.5, -0.75, -0.75, -0.5, -0.25,
     0.5,  0.5,   0.5,  -0.25, -0.25,
    -0.25, -0.25
], dtype=np.float32)

all_v_names = [f"v{i}" for i in range(1,13)]

# ---------- 开关：是否启用 winsorize（默认关闭） ----------
APPLY_WINSORIZE = False

# 若需禁用 log1p 变换，也可以把此值设为 False
APPLY_LOG1P_IF_SKEWED = True

# 阈值策略：按分位数（默认 98）或绝对阈值
use_percentile = True
percentile_val = 98
absolute_threshold = None

# distance-to-urban 参数（proximity = exp(-lambda_dist * distance_m)）
DIST_WEIGHT = 0.5        # distance 因子的权重（加入 weights 列表）
LAMBDA_DIST = 1.0 / 500  # 衰减速度（单位：1/m），越大越快速衰减（更偏近城市）

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
        ref_nodata = ref.nodata
except Exception as e:
    print(f"[ERROR] 无法读取 land_use: {raster_paths['land_use']}\n{e}")
    sys.exit(1)

# 把 land_use nodata 处理为 np.nan 保持可检测
if ref_nodata is not None:
    land_use_arr = np.where(land_use_arr == ref_nodata, np.nan, land_use_arr)

# ----------------- 读取 mask -----------------
try:
    with rasterio.open(raster_paths["mask"]) as src_mask:
        mask_arr = src_mask.read(1).astype(np.float32)
        mask_nodata = src_mask.nodata
        if mask_nodata is not None:
            mask_arr = np.where(mask_arr == mask_nodata, np.nan, mask_arr)
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

# ----------------- 先计算候选 mask（用于稳健化统计） -----------------
land_use_valid = np.isfinite(land_use_arr)
is_nonurban = land_use_valid & (land_use_arr == NONURBAN_VAL)
mask_ok = np.isfinite(mask_arr) & (mask_arr == 1)
valid_mask_candidates = mask_ok & is_nonurban
num_total = height * width
print(f"Image size: {width} x {height}")
print(f"Candidate pixels (mask==1 & NonUrban): {int(np.sum(valid_mask_candidates))}")

# ----------------- 读取 v* 层 -----------------
v_keys = [k for k in raster_paths.keys() if k.startswith("v")]
if not v_keys:
    print("[WARN] 未找到 v* 层")
    sys.exit(1)

v_layers = []
loaded_keys = []
for k in v_keys:
    p = raster_paths[k]
    print(f"[INFO] 读取并对齐 {k}: {p}")
    try:
        arr = read_and_align(p)
    except Exception as e:
        print(f"[ERROR] 读取 {k} 失败: {e}")
        sys.exit(1)
    v_layers.append(arr)
    loaded_keys.append(k)

v_stack = np.stack(v_layers, axis=0)  # (n, H, W)
n_factors = v_stack.shape[0]
print(f"[INFO] 加载 {n_factors} 个因子，大小 {v_stack.shape[1:]} -> keys: {loaded_keys}")

# ----------------- 加入 distance-to-urban 因子（多源 BFS 计算格子距离） -----------------
# 计算到最近 urban 像元的曼哈顿格子步数
urban_mask = np.isfinite(land_use_arr) & (land_use_arr == URBAN_VAL)
dist_cells = np.full((height, width), np.inf, dtype=float)
if np.any(urban_mask):
    print("[INFO] 计算到最近 urban 的网格距离（BFS，多源）...")
    q = deque()
    # 初始化队列
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
    # 像素大小（近似）
    pixel_size = max(abs(ref_transform.a), abs(ref_transform.e))
    dist_m = dist_cells * pixel_size
    print(f"[INFO] distance-to-urban computed (pixel_size~{pixel_size:.3f} units).")
    # 转换为亲近度（0..1），越靠近城市越接近 1
    proximity = np.exp(-LAMBDA_DIST * dist_m)  # decay exponential
else:
    print("[WARN] 图中没有 urban 像元，distance-to-urban 全设为 0")
    proximity = np.zeros((height, width), dtype=float)

# 将 proximity 当成新因子加入 v_stack
v_stack = np.vstack([v_stack, proximity[None, ...]])
loaded_keys.append("dist_to_urban")
n_factors = v_stack.shape[0]
print(f"[INFO] 加入 distance-to-urban 因子 -> 新因子数: {n_factors}")

# ----------------- 构建权重数组（按 loaded_keys 顺序） -----------------
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
print(f"[INFO] 使用权重 (对应 loaded_keys): {weights_list}")

# ----------------- 数据稳健化：winsorize 可选 + 偏态 log1p 变换 -----------------
def winsorize_v_stack(v_stack: np.ndarray, mask2d: np.ndarray=None, p_low=0.01, p_high=0.99):
    vs = v_stack.copy()
    n = vs.shape[0]
    for i in range(n):
        arr = vs[i]
        if mask2d is not None:
            sel = mask2d & np.isfinite(arr)
            vals = arr[sel]
        else:
            vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            continue
        lo = float(np.nanpercentile(vals, p_low*100))
        hi = float(np.nanpercentile(vals, p_high*100))
        if not np.isfinite(lo):
            lo = np.nanmin(vals)
        if not np.isfinite(hi):
            hi = np.nanmax(vals)
        arr = np.where(arr < lo, lo, arr)
        arr = np.where(arr > hi, hi, arr)
        vs[i] = arr
        print(f"[WINSORIZE] layer {i} -> {p_low*100:.1f}p={lo:.4g}, {p_high*100:.1f}p={hi:.4g}")
    return vs

def log_transform_if_skewed(v_stack: np.ndarray, mask2d: np.ndarray=None, skew_threshold=3.0):
    vs = v_stack.copy()
    n = vs.shape[0]
    for i in range(n):
        arr = vs[i]
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
        if ratio > skew_threshold and minv >= 0:
            vs[i] = np.log1p(np.where(np.isfinite(arr), np.maximum(arr, 0.0), np.nan))
            print(f"[LOG1P] layer {i} skew ratio={ratio:.2f}, applied log1p")
    return vs

# 仅在候选像元上计算 percentile / skewness（避免非候选区影响）
mask_for_stats = valid_mask_candidates

if APPLY_WINSORIZE:
    v_stack = winsorize_v_stack(v_stack, mask2d=mask_for_stats, p_low=0.01, p_high=0.99)
else:
    print("[INFO] winsorize 已禁用（APPLY_WINSORIZE=False）")

if APPLY_LOG1P_IF_SKEWED:
    v_stack = log_transform_if_skewed(v_stack, mask2d=mask_for_stats, skew_threshold=3.0)
else:
    print("[INFO] log1p 偏态变换已禁用（APPLY_LOG1P_IF_SKEWED=False）")

# ----------------- z-score 标准化（仅基于候选像元统计） -----------------
v_masked_for_stats = np.where(mask_for_stats[None, ...], v_stack, np.nan)
means = np.nanmean(v_masked_for_stats, axis=(1,2), keepdims=True)
stds = np.nanstd(v_masked_for_stats, axis=(1,2), keepdims=True)
stds_safe = np.where(np.isfinite(stds) & (stds > 0), stds, 1.0)
v_norm = (v_stack - means) / stds_safe

# 若某层在候选区全为 nan，替成 0（避免传播 NaN）
for i in range(v_norm.shape[0]):
    layer = v_norm[i]
    if not np.any(np.isfinite(layer)):
        v_norm[i] = np.zeros_like(layer)
        print(f"[INFO] normalized layer {i} has no finite values in candidates -> fill with zeros")

# ----------------- 计算效用 U（线性加权） -----------------
if weights.shape[0] != v_norm.shape[0]:
    print(f"[WARN] 权重长度 {weights.shape[0]} 与 因子数 {v_norm.shape[0]} 不一致，做截断或补零。")
    if weights.shape[0] > v_norm.shape[0]:
        weights = weights[:v_norm.shape[0]]
    else:
        pad = np.zeros((v_norm.shape[0] - weights.shape[0], 1, 1), dtype=np.float32)
        weights = np.vstack([weights, pad])

U_all = np.nansum(v_norm * weights, axis=0)  # (H,W)

# ----------------- 限定“新建”候选：mask==1 且 land_use == NONURBAN -----------------
U = np.where(valid_mask_candidates, U_all, np.nan)

# ----------------- 诊断打印（效用统计） -----------------
if np.any(np.isfinite(U)):
    u_min = float(np.nanmin(U))
    u_max = float(np.nanmax(U))
    u_mean = float(np.nanmean(U))
    prop_pos = float(np.sum(U > 0) / np.sum(np.isfinite(U)))
else:
    u_min = u_max = u_mean = float('nan')
    prop_pos = 0.0

print(f"\nUtility range (on NonUrban & mask): {u_min:.6f} to {u_max:.6f}")
print(f"Mean utility (on NonUrban & mask): {u_mean:.6f}")
print(f"Proportion U>0 (on NonUrban & mask): {prop_pos:.6f}")
print(f"\nPixel counts: total={num_total}, mask==1={int(np.sum(mask_ok))}, nonurban={int(np.sum(is_nonurban))}, candidates(mask&nonurban)={int(np.sum(valid_mask_candidates))}")

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
cand_idx = np.isfinite(U)
out_uint8[cand_idx] = (U[cand_idx] > thr).astype(np.uint8)

# development probability (linearly scaled over candidate U range)
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
prof_uint8 = profile.copy()
prof_uint8.update(dtype=rasterio.uint8, count=1, nodata=nodata_val_uint8)
with rasterio.open(out_pred_path, "w", **prof_uint8) as dst:
    dst.write(out_uint8, 1)
print(f"[OK] 写出 new-build 二值预测: {out_pred_path}")

prof_util = profile.copy()
prof_util.update(dtype=rasterio.float32, count=1, nodata=util_nodata)
with rasterio.open(out_util_path, "w", **prof_util) as dst:
    dst.write(util_filled, 1)
print(f"[OK] 写出效用得分: {out_util_path}")

with rasterio.open(out_prob_path, "w", **prof_util) as dst:
    dst.write(dev_prob_filled, 1)
print(f"[OK] 写出开发概率图: {out_prob_path}")

# ----------------- 后检查 & Summary -----------------
def print_counts(path, nodata_uint8=nodata_val_uint8):
    try:
        with rasterio.open(path) as src:
            a = src.read(1)
            vals, counts = np.unique(a, return_counts=True)
            print(f"\nValue counts for {os.path.basename(path)}:")
            for v, c in zip(vals, counts):
                label = ""
                if src.dtypes[0].startswith('uint'):
                    if int(v) == nodata_uint8:
                        label = " nodata"
                    elif int(v) == 1:
                        label = " developed(1)"
                    elif int(v) == 0:
                        label = " not_developed(0)"
                # print(f"  {int(v)}: {c} px{label}")
    except Exception as e:
        print(f"[WARN] cannot inspect {path}: {e}")

print_counts(out_pred_path)
print_counts(out_util_path)
print_counts(out_prob_path)

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
