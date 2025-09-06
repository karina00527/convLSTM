# 因子对齐，把所有数据对齐
import os
import shutil
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ------ 基准（2020） ------
REF = r"D:\paper2DATA\inputdata\landuse\2020.tif"

# 需要检查/对齐的“因子/landuse”列表（不含 label；label 已对齐，别动）
INPUTS = [
    r"D:\paper2DATA\inputdata\landuse\2008.tif",
    r"D:\paper2DATA\inputdata\landuse\2012.tif",
    r"D:\paper2DATA\inputdata\landuse\2016.tif",
    r"D:\paper2DATA\inputdata\convlstm\floodrisk\floodrisk.tif",
    r"D:\paper2DATA\inputdata\convlstm\greenspace\greenspace.tif",
    r"D:\paper2DATA\inputdata\convlstm\fibre\fibre.tif",
    r"D:\paper2DATA\inputdata\convlstm\slope.tif",
    r"D:\paper2DATA\inputdata\convlstm\3water1000m.tif",
    r"D:\paper2DATA\inputdata\convlstm\dem\dem.tif",
    r"D:\paper2DATA\inputdata\busstop\busstop.tif",
    r"D:\paper2DATA\inputdata\roads\roads.tif",
    r"D:\paper2DATA\inputdata\convlstm\12label\12label.tif",
    r"D:\paper2DATA\inputdata\convlstm\16label\16label.tif",
    r"D:\paper2DATA\inputdata\convlstm\20label\20label.tif"
]

# 哪些是“离散/分类” → NEAREST；没列到的默认按“连续” → BILINEAR
NEAREST_SET = {
    r"D:\paper2DATA\inputdata\landuse\2008.tif",
    r"D:\paper2DATA\inputdata\landuse\2012.tif",
    r"D:\paper2DATA\inputdata\landuse\2016.tif",
    r"D:\paper2DATA\inputdata\convlstm\floodrisk\floodrisk.tif",
    r"D:\paper2DATA\inputdata\convlstm\fibre\fibre.tif",
    r"D:\paper2DATA\inputdata\convlstm\12label\12label.tif",
    r"D:\paper2DATA\inputdata\convlstm\16label\16label.tif",
    r"D:\paper2DATA\inputdata\convlstm\20label\20label.tif"
}

def same_grid(a, b):
    """两栅格是否同 CRS / transform / shape"""
    return (a.crs == b.crs) and (a.transform == b.transform) and \
           (a.width == b.width) and (a.height == b.height)

def align_or_copy(src_path, ref_ds, out_path, resamp):
    """与基准一致就复制；否则重采样对齐"""
    with rasterio.open(src_path) as src:
        if same_grid(src, ref_ds):
            # 不重采样，直接拷贝（保持原样）
            if os.path.abspath(src_path) != os.path.abspath(out_path):
                shutil.copy2(src_path, out_path)
            print(f"= SKIP (same grid): {os.path.basename(src_path)} → {os.path.basename(out_path)}")
            return

        # 需要对齐：只处理单波段；多波段自行扩展
        out = np.empty((ref_ds.height, ref_ds.width), dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_ds.transform, dst_crs=ref_ds.crs,
            resampling=resamp
        )
        prof = ref_ds.profile.copy()
        prof.update(count=1, dtype=out.dtype, compress="lzw")
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(out, 1)
        print(f"* RESAMP({resamp.name}): {os.path.basename(src_path)} → {os.path.basename(out_path)}")

with rasterio.open(REF) as ref_ds:
    for p in INPUTS:
        if not os.path.exists(p):
            print(f"文件不存在，跳过: {p}")
            continue
        out = p[:-4] + "_to2020.tif"
        resamp = Resampling.nearest if p in NEAREST_SET else Resampling.bilinear
        align_or_copy(p, ref_ds, out, resamp)

print(" 完成：一致→复制；不一致→对齐重采样。")
