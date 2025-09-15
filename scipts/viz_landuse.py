# ==== 自行配置这三项 ====
INPUT_DIR = r"datas/coastline"   # 你的tif文件夹（Windows可用 r"D:\data\urban"）
# INPUT_DIR = r"xx"   # 你的tif文件夹（Windows可用 r"D:\data\urban"）

PATTERN   = "*.tif"                       # 文件名通配符
OUT_PATH  = "urban_compare.png"           # 输出图片文件名
TITLE     = "城镇开发状态（历年对比）"       # 总标题（可改）

# （可选）如果你的分类编码和含义不同，可在这里改标签
VALUE_LABEL_MAP = {
    0: "未开发/不变",
    1: "新建/开发",
    2: "重建",
}

# ========================

import os, re, glob, math, warnings
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
from rasterio.warp import reproject, Resampling
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

YEAR_RE = re.compile(r'(19|20)\d{2}')

def extract_year(path: str) -> int:
    m = YEAR_RE.search(os.path.basename(path))
    return int(m.group(0)) if m else 10**9  # 抓不到年份的排最后

def list_tifs(input_dir: str, pattern: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"未找到任何匹配 {os.path.join(input_dir, pattern)} 的 GeoTIFF。")
    paths.sort(key=extract_year)
    return paths

def read_ref_grid(path: str):
    with rasterio.open(path) as src:
        ref_profile = src.profile.copy()
        ref_crs = src.crs
        ref_transform = src.transform
        ref_shape = (src.height, src.width)
        data = src.read(1)
        nodata = src.nodata
    return data, ref_profile, ref_crs, ref_transform, ref_shape, nodata

def read_and_align_to_ref(path: str, ref_crs, ref_transform, ref_shape, categorical=True, ref_nodata=None):
    with rasterio.open(path) as src:
        dst = np.zeros(ref_shape, dtype=src.dtypes[0] if categorical else np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest if categorical else Resampling.bilinear
        )
        nodata = src.nodata if src.nodata is not None else ref_nodata
    return dst, nodata

def build_cmap_and_norm(unique_vals: np.ndarray, value_label_map: Dict[int, str]):
    default_colors = {
        0: "#c8c8c8", 1: "#d62728", 2: "#ff7f0e",
        3: "#1f77b4", 4: "#2ca02c", 5: "#9467bd",
        6: "#8c564b", 7: "#e377c2", 8: "#7f7f7f", 9: "#bcbd22",
    }
    vals = np.sort(np.unique(unique_vals)).astype(int)
    colors = [default_colors.get(v, "#000000") for v in vals]
    labels = [value_label_map.get(v, f"类别 {v}") for v in vals]
    cmap = ListedColormap(colors)
    boundaries = np.concatenate([vals - 0.5, [vals[-1] + 0.5]])
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, vals, labels

def make_grid(n: int, max_cols: int = 3) -> Tuple[int, int]:
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    return rows, cols

def plot_small_multiples(arrays: List[np.ndarray], titles: List[str], cmap, norm, nodata, out_path: str, suptitle="对比图"):
    rows, cols = make_grid(len(arrays))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 8), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()
    im0 = None
    for ax, arr, ttl in zip(axes, arrays, titles):
        arr_plot = np.ma.masked_equal(arr, nodata) if nodata is not None else arr
        im = ax.imshow(arr_plot, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(ttl, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if im0 is None: im0 = im
    for j in range(len(arrays), len(axes)):
        axes[j].axis('off')
    fig.suptitle(suptitle, fontsize=14)
    cbar = fig.colorbar(im0, ax=axes.tolist(), shrink=0.9)
    cbar.set_label("类别", rotation=90)
    fig.savefig(out_path, dpi=300)
    print(f"✅ 已导出: {out_path}")
    plt.show()

def run():
    tif_paths = list_tifs(INPUT_DIR, PATTERN)
    print("将要处理的文件（按年份排序）:")
    for p in tif_paths:
        print(" -", os.path.basename(p))

    # 参考网格：以第一个tif为准
    ref_arr, ref_profile, ref_crs, ref_transform, ref_shape, ref_nodata = read_ref_grid(tif_paths[0])

    arrays, titles, all_uniques = [], [], []
    for p in tif_paths:
        arr, nodata = read_and_align_to_ref(p, ref_crs, ref_transform, ref_shape, categorical=True, ref_nodata=ref_nodata)
        arrays.append(arr)
        y = extract_year(p)
        titles.append(str(y) if y < 10**9 else os.path.basename(p))
        if nodata is not None:
            all_uniques.append(np.unique(arr[arr != nodata]))
        else:
            all_uniques.append(np.unique(arr))

    uniques = np.unique(np.concatenate(all_uniques)) if all_uniques else np.unique(arrays[0])
    cmap, norm, vals, labels = build_cmap_and_norm(uniques, VALUE_LABEL_MAP)

    plot_small_multiples(arrays, titles, cmap, norm, ref_nodata, OUT_PATH, suptitle=TITLE)

if __name__ == "__main__":
    run()
