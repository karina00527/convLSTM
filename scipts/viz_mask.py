import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

# 数据路径
fp = "/home/xyf/Downloads/landuse/datas/fibre/fibre_to2020.tif"
mask_fp = "/home/xyf/Downloads/landuse/datas/mask/mask.tif"

def read_and_align_to_ref(ref_ds, src_path, resampling=Resampling.nearest):
    """
    读取 src_path，并重投影/重采样对齐到 ref_ds（rasterio dataset opened for reference）。
    返回 (arr, nodata) 两元组，arr 为 float ndarray，src nodata 已被替换为 np.nan。
    """
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype(float)
        src_nd = src.nodata

        # 如果已经完全相同网格/CRS，则原样返回（但仍把 nodata->nan）
        same_crs = (src.crs == ref_ds.crs)
        same_grid = (src.transform == ref_ds.transform) and (src.width == ref_ds.width) and (src.height == ref_ds.height)
        if same_crs and same_grid:
            arr = src_arr
        else:
            # 重投影到参考网格
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

# ---------- 读取 greenspace（参考栅格） ----------
with rasterio.open(fp) as ds:
    arr = ds.read(1).astype(float)
    nodata = ds.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    print("Reference CRS:", ds.crs)
    print("Reference pixel size:", ds.transform.a, ds.transform.e)
    print("Greenspace Min/Max (ignoring NaN):", np.nanmin(arr), np.nanmax(arr))

    # ---------- 读取并对齐 mask ----------
    try:
        mask_arr, mask_nd = read_and_align_to_ref(ds, mask_fp, resampling=Resampling.nearest)
    except Exception as e:
        print(f"[WARN] 无法读取/对齐 mask: {e}\n将不应用 mask（会在全部像素上显示）。")
        mask_arr = None

    # 把 mask 的 nodata 替换为 np.nan（read_and_align 已做，但再保险）
    if mask_arr is not None:
        mask_arr = np.where(np.isfinite(mask_arr), mask_arr, np.nan)

    # ---------- 应用 mask（仅保留 mask==1 的像元） ----------
    if mask_arr is not None:
        mask_ok = np.isfinite(mask_arr) & (mask_arr == 1)
        kept = int(np.sum(mask_ok))
        total_mask_pixels = int(np.sum(np.isfinite(mask_arr)))
        print(f"Mask finite pixels: {total_mask_pixels}, Mask==1 kept pixels: {kept}")
        # 仅在 mask==1 上保留原值，其他位置设为 np.nan
        arr_masked = np.where(mask_ok, arr, np.nan)
    else:
        # 如果没有 mask，就使用原始 arr
        arr_masked = arr
        print("[INFO] 未应用 mask，使用完整栅格。")

# ---------- 绘图：只显示被 mask 保留后的值 ----------
plt.figure(figsize=(8,6))
vmin = np.nanmin(arr_masked)
vmax = np.nanmax(arr_masked)
im = plt.imshow(arr_masked, cmap='viridis', vmin=vmin, vmax=vmax)
plt.colorbar(im, label='Distance to nearest greenspace')
plt.title('greenspace (masked: mask==1)')
plt.axis('off')
plt.show()

# ---------- 直方图：仅对被保留的像元绘制 ----------
finite = np.isfinite(arr_masked)
if finite.sum() == 0:
    print("[WARN] 掩码后没有可用像元可画直方图")
else:
    plt.figure(figsize=(6,4))
    plt.hist(arr_masked[finite].ravel(), bins=50)
    plt.xlabel('Distance'); plt.ylabel('Count'); plt.title('Histogram (masked pixels only)')
    plt.show()
