"""Simplified raster -> parcel loader.

目标：后续只改 config.yaml，不再改这个 py。

核心特性：
1. 统一参考栅格（land_use）作为对齐基准；其它层若尺寸/CRS 不同自动重投影/重采样。
2. land_use 数值映射到标签 (mapping.land_use)。
3. 仅保留需要的地块：
    - land_use != "unknown"
    - （可选）若指定仅保留 NonUrban，可在 config.filters.include_land_use 控制。
4. 全局 mask：mask==1 保留；mask==0 或 nodata 丢弃。
5. 风险：risk>0 记为 risk_mask；若配置 max_risk 存在且 risk_val>max_risk -> 丢弃。
6. nodata 策略：drop_if_any / zero_if_nodata / none_if_nodata。
7. v1,v2,v3 以及任意 v* 因子自动 z-score（若配置 normalize）。
8. 通过 config.limit 限制最大地块数。
9. 新增：当其它栅格 CRS 可语义匹配但表示形式不同 (EPSG vs WKT) 或 CRS 不同但可重投影时，自动重投影到参考栅格（最近邻 / 双线性可扩展）。

如需新增栅格：在 config.raster.base 下添加 key；若需写入 dataclass 添加字段；仅作效用因子则使用动态 v*。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import numpy as np

import re
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling


def crs_match(crs1, crs2) -> bool:
    """检查两个 CRS 是否语义相同（忽略 EPSG vs WKT 格式差异）"""
    try:
        c1 = CRS.from_user_input(crs1)
        c2 = CRS.from_user_input(crs2)
        return c1 == c2
    except Exception:
        return str(crs1) == str(crs2)

try:  # pragma: no cover - optional dependency
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None  # type: ignore


@dataclass
class RasterParcelRecord:
    row: int
    col: int
    x: float
    y: float
    land_use_2008: str
    built_year: Optional[int]
    floor_area: float
    far_max: float
    lot_area: float
    access_score: float
    risk_mask: bool
    build_allowed: bool = True
    # 预留可拓展字段
    infrastructure_score: float = 0.5
    hazard_score: float = 0.0
    market_activity: float = 0.5
    price_per_sqm: float = 0.0
    land_price_per_sqm: float = 0.0
    v1: float = 0.0
    v2: float = 0.0
    v3: float = 0.0
    # 动态 v* 因子全集（包含 v1,v2,v3 及其他）
    factors: Dict[str, float] = field(default_factory=dict)
def load_raster_parcels(raster_cfg: Dict) -> List[RasterParcelRecord]:
    if rasterio is None:
        raise ImportError("rasterio not installed but raster configuration supplied.")

    base = raster_cfg.get("base", {})
    mapping = raster_cfg.get("mapping", {})
    land_use_map = mapping.get("land_use", {})
    nodata_policy = raster_cfg.get("nodata_policy", {})
    drop_if_any = set(nodata_policy.get("drop_if_any", []))
    zero_if_nodata = set(nodata_policy.get("zero_if_nodata", []))
    none_if_nodata = set(nodata_policy.get("none_if_nodata", []))
    filters = raster_cfg.get("filters", {})
    include_land_use = set(filters.get("include_land_use", [])) or None
    max_risk = filters.get("max_risk")  # can be None -> no threshold filter
    limit = raster_cfg.get("limit")

    land_use_path = base.get("land_use")
    if not land_use_path:
        raise ValueError("raster.base.land_use is required")

    # --- 参考栅格 ---
    try:
        with rasterio.open(land_use_path) as ref:
            width, height = ref.width, ref.height
            ref_transform, ref_crs = ref.transform, ref.crs
            land_use = ref.read(1).astype(np.float32)
            if ref.nodata is not None:
                land_use = np.where(land_use == ref.nodata, np.nan, land_use)
    except Exception as e:
        raise RuntimeError(f"无法读取 land_use 栅格 {land_use_path}: {e}")

    # --- 读取函数：若尺寸/transform 不一致或 CRS 仅语义可转换，则重投影到参考栅格 ---
    def read_strict(path: Optional[str]):
        if not path:
            return None
        with rasterio.open(path) as src:
            src_arr = src.read(1).astype(np.float32)
            nd = src.nodata
            same_crs = crs_match(src.crs, ref_crs)
            same_grid = (src.transform == ref_transform and src.width == width and src.height == height)
            if same_crs and same_grid:
                arr = src_arr
            else:
                # 若 CRS 不同但可解析，或 grid 不同 -> 重投影
                dst = np.empty((height, width), dtype=np.float32)
                dst[:] = np.nan
                try:
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
                    arr = dst
                    print(f"[INFO] 重投影以对齐参考栅格: {path}")
                except Exception as e:
                    raise ValueError(f"Raster grid/CRS mismatch and reprojection failed: {path} ({e})")
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)
        return arr

    # --- 读取可选层 ---
    built_year = read_strict(base.get("built_year"))
    floor_area = read_strict(base.get("floor_area"))
    far_max = read_strict(base.get("far_max"))
    access = read_strict(base.get("access"))
    infrastructure = read_strict(base.get("infrastructure"))
    hazard = read_strict(base.get("hazard"))
    market_activity_arr = read_strict(base.get("market_activity"))
    price_arr = read_strict(base.get("price_per_sqm"))
    land_price_arr = read_strict(base.get("land_price_per_sqm"))
    risk = read_strict(base.get("risk"))
    mask = read_strict(base.get("mask"))

    # 动态抓取所有 v* 图层
    v_layers: Dict[str, np.ndarray] = {}
    for key, path in base.items():
        if isinstance(key, str) and re.fullmatch(r"v\d+", key) and isinstance(path, str):
            try:
                arr = read_strict(path)
            except Exception as e:  # 缺失或无法读取此可选层 -> 发出警告并跳过
                print(f"[WARN] 跳过缺失或无法读取的因子 {key}: {path} ({e})")
                arr = None
            if arr is not None:
                v_layers[key] = arr

    # 标准化推迟到过滤完成后（只对最终保留像元）
    util_cfg = raster_cfg.get('utility_generic', {})
    do_norm = util_cfg.get('normalize', True)

    # --- v1/v2/v3 z-score （若存在任意一层）---
    def zscore(a: Optional[np.ndarray]):
        if a is None:
            return
        m = np.nanmean(a)
        s = np.nanstd(a)
        if not np.isfinite(s) or s == 0:
            a[:] = 0.0
        else:
            a[:] = (a - m) / s
    # 先不标准化，保留原值，后面 records 构建后再做
    v1 = v_layers.get('v1')
    v2 = v_layers.get('v2')
    v3 = v_layers.get('v3')

    # --- land_use 映射到标签 ---
    land_use_labels = np.full((height, width), "unknown", dtype=object)
    it = np.nditer(land_use, flags=['multi_index'], op_flags=['readonly'])
    for val in it:
        r, c = it.multi_index
        v = float(val)
        if not np.isfinite(v):
            continue
        lab = land_use_map.get(int(v), "unknown")
        land_use_labels[r, c] = lab

    # --- 结果列表 ---
    records: List[RasterParcelRecord] = []
    lot_area_pixel = abs(ref_transform.a) * abs(ref_transform.e)

    for row in range(height):
        for col in range(width):
            lu_label = land_use_labels[row, col]
            if lu_label == "unknown":
                continue
            if include_land_use and lu_label not in include_land_use:
                continue

            # mask 过滤（仅 mask==1 保留）
            if mask is not None:
                mv = mask[row, col]
                if not (np.isfinite(mv) and int(mv) == 1):
                    continue

            # Risk 处理
            risk_mask = False
            if risk is not None and np.isfinite(risk[row, col]):
                rv = float(risk[row, col])
                risk_mask = (rv > 0.0)
                if (max_risk is not None) and (rv > float(max_risk)):
                    continue

            # built_year
            b_year: Optional[int] = None
            if built_year is not None:
                bv = built_year[row, col]
                if np.isnan(bv):
                    if "built_year" in drop_if_any:
                        continue
                    if "built_year" in none_if_nodata:
                        b_year = None
                else:
                    b_year = int(bv)

            # floor_area
            fa_val = 0.0
            if floor_area is not None:
                fv = floor_area[row, col]
                if np.isnan(fv):
                    if "floor_area" in drop_if_any:
                        continue
                    if "floor_area" in zero_if_nodata:
                        fa_val = 0.0
                else:
                    fa_val = float(fv)

            # far_max
            far_val = 2.0
            if far_max is not None:
                fv2 = far_max[row, col]
                if np.isnan(fv2):
                    if "far_max" in drop_if_any:
                        continue
                else:
                    far_val = float(fv2) if fv2 > 0 else 2.0

            # access
            access_val = 0.5
            if access is not None:
                av = access[row, col]
                if np.isnan(av):
                    if "access" in drop_if_any:
                        continue
                else:
                    access_val = float(av)

            # infrastructure
            infra_val = 0.5
            if infrastructure is not None:
                iv = infrastructure[row, col]
                if np.isnan(iv):
                    if "infrastructure" in drop_if_any:
                        continue
                else:
                    infra_val = float(iv)

            # hazard
            hazard_val = 0.0
            if hazard is not None:
                hv = hazard[row, col]
                if np.isnan(hv):
                    if "hazard" in drop_if_any:
                        continue
                else:
                    hazard_val = float(hv)

            # market activity
            market_val = 0.5
            if market_activity_arr is not None:
                mv2 = market_activity_arr[row, col]
                if np.isnan(mv2):
                    if "market_activity" in drop_if_any:
                        continue
                else:
                    market_val = float(mv2)

            # prices
            price_val = 0.0
            if price_arr is not None:
                pv = price_arr[row, col]
                if np.isnan(pv):
                    if "price_per_sqm" in drop_if_any:
                        continue
                else:
                    price_val = float(pv)
            land_price_val = 0.0
            if land_price_arr is not None:
                lpv = land_price_arr[row, col]
                if np.isnan(lpv):
                    if "land_price_per_sqm" in drop_if_any:
                        continue
                else:
                    land_price_val = float(lpv)

            # 提取动态 v* 值
            factors: Dict[str, float] = {}
            for vk, arr in v_layers.items():
                val = arr[row, col]
                factors[vk] = float(val) if np.isfinite(val) else 0.0
            v1_val = factors.get('v1', 0.0)
            v2_val = factors.get('v2', 0.0)
            v3_val = factors.get('v3', 0.0)

            x, y = rasterio.transform.xy(ref_transform, row, col, offset="center")
            records.append(RasterParcelRecord(
                row=row,
                col=col,
                x=x,
                y=y,
                land_use_2008=lu_label,
                built_year=b_year,
                floor_area=fa_val,
                far_max=far_val,
                lot_area=lot_area_pixel,
                access_score=access_val,
                risk_mask=risk_mask,
                build_allowed=True,
                infrastructure_score=infra_val,
                hazard_score=hazard_val,
                market_activity=market_val,
                price_per_sqm=price_val,
                land_price_per_sqm=land_price_val,
                v1=v1_val,
                v2=v2_val,
                v3=v3_val,
                factors=factors,
            ))

            if limit and len(records) >= limit:
                return records

    # ---- 仅对保留的 records 做 z-score ----
    if do_norm and records:
        # 收集所有 v* 键
        all_keys = set()
        for r in records:
            all_keys.update(r.factors.keys())
        for key in all_keys:
            values = [r.factors.get(key, 0.0) for r in records]
            arr = np.array(values, dtype=float)
            mean = float(np.mean(arr)) if len(arr) > 0 else 0.0
            std = float(np.std(arr)) if len(arr) > 0 else 0.0
            if std == 0 or not np.isfinite(std):
                normed = [0.0]*len(arr)
            else:
                normed = ((arr - mean) / std).tolist()
            for r, nv in zip(records, normed):
                r.factors[key] = nv
                # 同步回 v1/v2/v3 基础字段（保持兼容）
                if key == 'v1':
                    r.v1 = nv
                elif key == 'v2':
                    r.v2 = nv
                elif key == 'v3':
                    r.v3 = nv
    return records
