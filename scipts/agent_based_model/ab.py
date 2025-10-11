"""
Urban Development Agent-Based Model — Voting & Raster Outputs

功能要点
- 从 raster_loader.py 读取地块（含 v1/v2/v3、mask==1、NonUrban），作为 ParcelAgent。
- DeveloperAgent 基于 config.utility_generic.weights 对 (v1,v2,v3) 计算效用 U。
- 投票规则（同年并行）：每类开发商各自扫描一批候选，选出效用≥阈值的“心仪地”投 1 票；
  同一年内，如果同一地块获得 ≥ vote_threshold 票 → 视为“被开发”。
- 运行结束写出两张 GeoTIFF：
  1) developed.tif：0/1，是否在 2008–2020 期间被开发；
  2) potential.tif：最后一年（end_year）“U≥阈值”的开发商占比（0~1）。

配置项（config.yaml 中可选覆盖）
voting:
  enabled: true
  util_threshold: 0.5
  vote_threshold: 2
  candidate_sample: 2000
output:
  dir: "output"
  developed_tif: "output/developed.tif"
  potential_tif: "output/potential.tif"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Tuple
import random
import math
import os
import yaml

# 数值/栅格
import numpy as np
try:
    import rasterio
    from rasterio.warp import reproject, Resampling  # 仅用于 profile 读取，不在此文件重投影
except Exception:
    rasterio = None  # 允许无栅格环境下跑最小逻辑

# 进度条（可选）
try:  # 尝试引入 tqdm；若未安装则静默回退
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - 安全回退
    tqdm = None

# ---------------------------------------------------------------------------
# Mesa 依赖的兜底（可无 mesa 运行）
# ---------------------------------------------------------------------------
try:
    from mesa import Agent, Model
    from mesa.time import BaseScheduler
except Exception:
    class Agent:
        def __init__(self, unique_id, model):
            self.unique_id = unique_id
            self.model = model
        def step(self): pass

    class Model:
        def __init__(self):
            self.running = True

    class BaseScheduler:
        def __init__(self, model):
            self.model = model
            self._agents = []
        def add(self, agent: Agent):
            self._agents.append(agent)
        def step(self):
            for a in list(self._agents):
                a.step()

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DEVELOPER_TYPES = ("MumDad", "Small", "Large", "Specialist")

DEFAULT_VOTING = {
    "enabled": True,
    "util_threshold": 0.5,  # 效用阈值：≥则开发商愿意开发（记1票）
    "vote_threshold": 2,     # 同年内 ≥ 2 个开发商愿意 → 开发
    "candidate_sample": 1000, # 每个开发商每年扫描的候选上限
    "vote_quota_per_dev": 1,  # 每个开发商每年可投票的不同地块上限
    "max_starts_per_year_global": 10**9, # 全局开工上限（默认极大，相当于不限制）
}

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class ParcelState:
    """地块的状态（2008 基线 + 动态）"""
    current_use: str = "vacant"
    land_use_2008: str = "vacant"
    built_year: Optional[int] = None
    floor_area: float = 0.0
    far_max: float = 2.0
    lot_area: float = 500.0
    zoning: str = "GEN"
    risk_mask: bool = False
    access_score: float = 0.5
    last_redevelopment_year: Optional[int] = None
    allowed_density: float = 1.0
    base_value: float = 100_000.0
    improvement_value: float = 0.0
    # 栅格定位信息（由 raster_loader 提供）
    row: Optional[int] = None
    col: Optional[int] = None
    x: Optional[float] = None
    y: Optional[float] = None
    # 其他指标（占位）
    infrastructure_score: float = 0.5
    hazard_score: float = 0.0
    market_activity: float = 0.5
    price_per_sqm: float = 0.0
    land_price_per_sqm: float = 0.0
    # survey-based 因子（已标准化）
    v1: float = 0.0
    v2: float = 0.0
    v3: float = 0.0
    # 全局 mask：1 允许建
    build_allowed: bool = True
    # 动态 v* 集合
    factors: dict = field(default_factory=dict)
    under_construction: bool = False
    developed_by_type: str = ""  # 记录首次开发的开发商类型（空表示未开发）

    # 简化的“剩余开发潜力”
    def development_potential(self) -> float:
        max_floor = self.far_max * self.lot_area
        return max(0.0, max_floor - self.floor_area)

    def eligible_for_new_build(self) -> bool:
        return self.current_use == "vacant" or (self.floor_area / (self.far_max * self.lot_area + 1e-9)) < 0.5

    def eligible_for_rebuild(self, current_year: int) -> bool:
        if self.floor_area <= 0: return False
        if self.built_year is None: return False
        age = current_year - self.built_year
        return age >= 20  # 占位阈值

# ---------------------------------------------------------------------------
# Parcel Agent
# ---------------------------------------------------------------------------
class ParcelAgent(Agent):
    """地块 Agent：保存状态，提供简单工具方法"""
    def __init__(self, unique_id: int, model: "UrbanDevModel", preset: Optional[dict] = None):
        super().__init__(unique_id, model)
        self.state = ParcelState()
        if preset:
            for k, v in preset.items():
                if hasattr(self.state, k):
                    setattr(self.state, k, v)
            self.state.current_use = self.state.land_use_2008
        else:
            # 非栅格模式下的随机初始化
            self.state.land_use_2008 = random.choice(["vacant", "low_density", "residential", "industrial"])
            self.state.current_use = self.state.land_use_2008
            if self.state.land_use_2008 != "vacant":
                self.state.built_year = random.randint(1970, 2007)
                self.state.floor_area = random.uniform(0.3, 0.8) * self.state.far_max * self.state.lot_area
            self.state.access_score = random.uniform(0.2, 0.9)
            self.state.risk_mask = random.random() < 0.05

    def step(self):
        # 简单年化增值（可替换为你的市场/通胀等）
        self.state.base_value *= 1.01

    def potential_project_return(self, developer_type: str, year: int) -> float:
        # 仅用于“遗留路径”的占位收益代理，不在 voting 模式使用
        potential_fa = self.state.development_potential()
        fa_ratio = potential_fa / (self.state.far_max * self.state.lot_area + 1e-9)
        market_randomness = random.uniform(0.85, 1.15)
        age_penalty = 0.0
        if self.state.built_year is not None:
            age = max(0, year - self.state.built_year)
            age_penalty = 0.01 * min(age / 50, 1.0)
        type_bias = {"MumDad": 0.95, "Small": 1.00, "Large": 1.05, "Specialist": 1.10}.get(developer_type, 1.0)
        access = self.state.access_score
        return (fa_ratio * 1.2 + age_penalty + 0.3 * access) * market_randomness * type_bias

    def apply_development(self, developer: "DeveloperAgent", year: int, rebuild: bool):
        # 队列施工模式下的完工动作；在 voting 模式我们直接落成，不用这个
        if rebuild:
            self.state.improvement_value *= 0.2
        potential_fa = self.state.development_potential()
        if potential_fa > 0:
            alpha = 0.8 if rebuild else 0.6
            added_fa = alpha * potential_fa
            self.state.floor_area += added_fa
            self.state.improvement_value += added_fa * 500
        self.state.last_redevelopment_year = year
        if rebuild or self.state.built_year is None:
            self.state.built_year = year
        self.state.current_use = "residential"

    # 兼容方法
    def eligible_for_rebuild(self, current_year: int) -> bool:
        return self.state.eligible_for_rebuild(current_year)
    def eligible_for_new_build(self) -> bool:
        return self.state.eligible_for_new_build()

# ---------------------------------------------------------------------------
# Developer Agent
# ---------------------------------------------------------------------------
class DeveloperAgent(Agent):
    """开发商 Agent：决策逻辑（本版本主要用于投票选择）"""
    def __init__(self, unique_id: int, model: "UrbanDevModel", developer_type: str):
        super().__init__(unique_id, model)
        if developer_type not in DEVELOPER_TYPES:
            raise ValueError(f"Invalid developer_type {developer_type}")
        self.developer_type = developer_type
        base_capital = {
            "MumDad": 250_000,
            "Small": 2_000_000,
            "Large": 25_000_000,
            "Specialist": 5_000_000,
        }[developer_type] * model.config.get("construction", {}).get("capital_multiplier", {}).get(developer_type, 1.0)
        self.capital: float = base_capital
        self.portfolio: List[int] = []
        self.active_projects: List[int] = []
        # --- Competition fields ---
        self.frustration = 0          # 累积未中标次数
        self.successes = 0            # 累积成功次数
        # 基础类型得分，可来自 traits；这里简化为 1.0，可扩展从 config.traits 读取
        traits_cfg = model.config.get('traits', {}).get(developer_type, {})
        self.type_score = float(traits_cfg.get('baseline_bias', 0.0)) + 1.0

    def compute_parcel_utility(self, parcel: ParcelAgent, year: int) -> float:
        """效用函数(可配置模式):
        mode = diffusion (默认): 使用动态 v* 因子线性组合 + 扩散邻里加成(局部+距离) + 风险减分。
        mode = normalized: 使用用户指定的距离范围归一化 (v1,v2,v3) 并映射到 [0,1] 后再加邻里。
        通过 config.utility_generic.mode 选择，未指定则走 diffusion，确保不破坏原有扩散效果。
        """
        ug = self.model.utility_generic
        if not (ug.get('enabled') and 'weights' in ug and self.developer_type in ug['weights']):
            return 0.0
        mode = ug.get('mode', 'diffusion').lower()
        w = ug['weights'][self.developer_type]

        if mode == 'normalized':
            # -------- 归一化模式（保持你提供的公式） --------
            w = ug['weights'][self.developer_type]
            ranges_all = ug.get('ranges', {})
            rng_type = ranges_all.get(self.developer_type, {})
            def get_rng(k):
                r = rng_type.get(k)
                if not r or len(r) != 2:
                    return (0.0, 2000.0)
                return (float(r[0]), float(r[1]))
            def mm(x, lo, hi):
                if hi <= lo: return 0.0
                return max(0.0, min(1.0, (float(x) - lo)/(hi - lo)))
            vkeys = list(w.keys())
            vals_inv01 = {}

            for k in vkeys:
                raw = getattr(parcel.state, k, None)
                if raw is None:
                    raw = parcel.state.factors.get(k, 0.0)
                if raw is None:
                    continue
                lo, hi = get_rng(k)
                v01 = mm(raw, lo, hi)
                vals_inv01[k] = 1.0 - v01  # 取反

            if not vals_inv01:
                return 0.0
            
            u_raw = sum(w.get(k, 0.0) * vals_inv01.get(k, 0.0) for k in vkeys)
            u_max = sum(max(0.0, w.get(k, 0.0)) for k in vkeys)
            u_min = sum(min(0.0, w.get(k, 0.0)) for k in vkeys)
            denom = max(1e-9, u_max - u_min)
            u01 = (u_raw - u_min) / denom  # 归一化到 [0,1]

            if parcel.state.risk_mask:
                u01 *= (1.0 - float(ug.get('risk_penalty', 0.3)))
            neigh_w = float(ug.get('neigh_w', 0.0))
            if neigh_w != 0.0:
                u01 = min(1.0, max(0.0, u01 + neigh_w * self.model.neighborhood_score(parcel)))
            return float(u01)

        # -------- 扩散模式 (原版本思想) --------
        base = 0.0
        # 支持任意 v* 因子 (factors dict) + 显式 v1 v2 v3
        for fk, val in parcel.state.factors.items():
            if fk in w:
                base += w.get(fk, 0.0) * float(val)
        for k in ('v1','v2','v3'):
            if k in w:
                base += w.get(k, 0.0) * float(getattr(parcel.state, k, 0.0))
        # 风险惩罚: 减去一个常数 (轻微)
        if parcel.state.risk_mask:
            base -= float(ug.get('risk_penalty_abs', 0.5))

        # 邻里扩散: 原局部+全局混合 + 距离指数衰减
        ncfg = (self.model.config.get('neighborhood') or {})
        if ncfg.get('enabled', True):
            local_ratio = self.model.neighborhood_score(parcel)  # 0~1
            w_local = float(ncfg.get('local_weight', 0.6))
            w_dist = float(ncfg.get('distance_weight', 0.4))
            dist_decay = 0.0
            if hasattr(self.model, '_dev_distance') and parcel.state.row is not None:
                r = parcel.state.row; c = parcel.state.col
                try:
                    dval = self.model._dev_distance[r, c]
                    if not np.isnan(dval):
                        lam = float(ncfg.get('distance_decay', 0.5))
                        dist_decay = math.exp(-lam * dval)
                except Exception:
                    pass
            sat_local = local_ratio / (0.3 + local_ratio) if local_ratio > 0 else 0.0
            diffusion_bonus = w_local * sat_local + w_dist * dist_decay
            base += diffusion_bonus

        return float(base)

    def step(self): return  # 决策在模型阶段统一执行

# ---------------------------------------------------------------------------
# 项目（队列施工模式保留用）
# ---------------------------------------------------------------------------
@dataclass
class Project:
    project_id: int
    developer_id: int
    parcel_id: int
    start_year: int
    duration: int
    rebuild: bool
    cost: float
    utility_at_start: float
    original_duration: int

# ---------------------------------------------------------------------------
# 模型本体
# ---------------------------------------------------------------------------
class UrbanDevModel(Model):
    def __init__(
        self,
        n_parcels: int = 100,
        developer_type_counts: Optional[Dict[str, int]] = None,
        start_year: int = 2008,
        end_year: int = 2020,
        seed: Optional[int] = None,
        config_path: str = "config.yaml",
    ):
        super().__init__()
        if seed is not None:
            random.seed(seed)

        # 读取配置
        self.config = {}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            self.config = {}

        self.start_year = start_year
        self.end_year = end_year
        self.current_year = start_year
        self.n_parcels = n_parcels
        self.developer_type_counts = developer_type_counts or {"MumDad": 8, "Small": 6, "Large": 4, "Specialist": 2}
        self.scheduler = BaseScheduler(self)
        self.parcels: List[ParcelAgent] = []
        self.developers: List[DeveloperAgent] = []
        self._create_parcels()
        for p in self.parcels:
            if getattr(p.state, "land_use_2008","") == "Urban":
                p.state.floor_area = 1e-3
                p.state.built_year = self.start_year -1
        self._create_developers()

        # 权重/参数
        # self.weights: Dict[str, Dict[str, float]] = self.config.get('weights', {}) or {
        #   "MumDad": {"profit": 0.8, "access": 0.4, "neigh": 0.2, "risk": 0.6, "cost": 0.7, "zoning": 1.0},
         #  "Small": {"profit": 1.0, "access": 0.6, "neigh": 0.4, "risk": 0.5, "cost": 0.8, "zoning": 1.0},
        #   "Large": {"profit": 1.2, "access": 0.7, "neigh": 0.8, "risk": 0.4, "cost": 1.0, "zoning": 1.0},
        #   "Specialist": {"profit": 1.0, "access": 0.9, "neigh": 0.5, "risk": 0.3, "cost": 0.9, "zoning": 1.0},
        #
        # self.thresholds: Dict[str, float] = self.config.get('threshold', {}) or {t: 0.0 for t in DEVELOPER_TYPES}
        self.max_concurrent: Dict[str, int] = self.config.get('construction', {}).get('max_concurrent', {}) or {
            "MumDad": 1, "Small": 3, "Large": 10, "Specialist": 4
        }
        self.duration_ranges: Dict[str, Tuple[int, int]] = self.config.get('construction', {}).get('duration_years', {}) or {
            "MumDad": (1, 2), "Small": (1, 3), "Large": (2, 4), "Specialist": (1, 3)
        }
        self.utility_generic = self.config.get('utility_generic', {})

        # 记录
        self.yearly_stats: List[Dict[str, float]] = []
        self.projects: List[Project] = []
        self._next_project_id = 0

        # voting 配置与输出
        self.voting = {**DEFAULT_VOTING, **(self.config.get("voting") or {})}
        self.output_cfg = self.config.get("output", {}) or {}
        # Competition config (基础版)
        self.competition_cfg = {
            'enabled': True,
            'w_type': 0.5,
            'w_frustration': 0.5,
            'frustration_decay': 0.1,
            'frustration_expand_threshold': 3,   # >= 扩大采样
            'expand_factor': 2.0,
            'lower_util_step': 0.05,             # 降低效用阈值步长
            'min_util_threshold': 0.2,
        }
        self.competition_cfg.update(self.config.get('competition', {}) or {})

        # 读取 land_use profile 以便写 GeoTIFF
        self._ref_profile = None
        raster_cfg = self.config.get("raster") or {}
        land_use_path = ((raster_cfg.get("base") or {}).get("land_use"))
        if rasterio and land_use_path and os.path.exists(land_use_path):
            with rasterio.open(land_use_path) as ref:
                prof = ref.profile.copy()
                prof.update(count=1, compress="lzw")
                self._ref_profile = prof
                self._grid_shape = (ref.height, ref.width)
        else:
            self._grid_shape = None

    # ---------------------- 初始化 ----------------------
    def _adjacent_to_seed(self, parcel: "ParcelAgent", use_2008_as_seed: bool = True) -> bool:
    
        r, c = parcel.state.row, parcel.state.col 
        if r is None or c is None or self._grid_shape is None:
            return False
        grid_index = getattr(self, '_grid_index', None)
        if grid_index is None:
            self._grid_index = {
                (p.state.row, p.state.col): p for p in self.parcels if p.state.row is not None and p.state.col is not None
            }
            grid_index = self._grid_index

        H, W = self._grid_shape
        # ② 选择邻域（8邻域；若想用4邻域，就改成4个方向）
        neighbors = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                neigh = grid_index.get((nr, nc))
                if not neigh:
                    continue
                if neigh.state.under_construction or neigh.state.floor_area > 0:
                    return True
                if use_2008_as_seed and getattr(neigh.state, "land_use_2008", "") == "Urban":
                    return True

        return False

    def _create_parcels(self):
        raster_cfg = self.config.get('raster')
        if raster_cfg:
            try:
                from raster_loader import load_raster_parcels
            except ImportError:
                raise ImportError("Raster configuration provided but raster_loader or dependencies not available.")
            records = load_raster_parcels(raster_cfg)
            for i, rec in enumerate(records):
                preset = dict(
                    land_use_2008=rec.land_use_2008,
                    built_year=rec.built_year,
                    floor_area=rec.floor_area,
                    far_max=rec.far_max,
                    lot_area=rec.lot_area,
                    access_score=rec.access_score,
                    risk_mask=rec.risk_mask,
                    build_allowed=rec.build_allowed,
                    infrastructure_score=getattr(rec, 'infrastructure_score', 0.5),
                    hazard_score=getattr(rec, 'hazard_score', 0.0),
                    market_activity=getattr(rec, 'market_activity', 0.5),
                    price_per_sqm=getattr(rec, 'price_per_sqm', 0.0),
                    land_price_per_sqm=getattr(rec, 'land_price_per_sqm', 0.0),
                    v1=getattr(rec, 'v1', 0.0),
                    v2=getattr(rec, 'v2', 0.0),
                    v3=getattr(rec, 'v3', 0.0),
                    factors=getattr(rec, 'factors', {}),
                    row=getattr(rec, 'row', None),
                    col=getattr(rec, 'col', None),
                    x=getattr(rec, 'x', None),
                    y=getattr(rec, 'y', None),
                )
                p = ParcelAgent(unique_id=i, model=self, preset=preset)
                # 轻微异质性
                p.state.base_value *= random.uniform(0.95, 1.05)
                p.state.allowed_density = p.state.far_max
                self.parcels.append(p)
                self.scheduler.add(p)
            self.n_parcels = len(self.parcels)
        else:
            for i in range(self.n_parcels):
                p = ParcelAgent(unique_id=i, model=self)
                p.state.base_value *= random.uniform(0.8, 1.4)
                p.state.allowed_density = random.choice([0.8, 1.0, 1.2, 1.5])
                self.parcels.append(p)
                self.scheduler.add(p)

    def _create_developers(self):
        uid_offset = self.n_parcels
        for dtype, count in (self.developer_type_counts or {}).items():
            for _ in range(count):
                d = DeveloperAgent(unique_id=uid_offset, model=self, developer_type=dtype)
                uid_offset += 1
                self.developers.append(d)
                self.scheduler.add(d)

    # ---------------------- 政策钩子（占位） ----------------------
    def policy_incentive(self, parcel: ParcelAgent, developer: DeveloperAgent) -> float:
        return 1.0
    def policy_allows(self, parcel: ParcelAgent, developer: DeveloperAgent) -> bool:
        return True

    # ---------------------- 年度过程（原队列模式） ----------------------
    def step_year(self):
        self._advance_projects()
        for parcel in self.parcels:
            parcel.step()
        self._recompute_neighborhood()
        for dev in self.developers:
            pass  # 决策在 decide_and_start_project；此处保留旧接口
        self._record_year_stats()
        self.current_year += 1

    # ---------------------- 年度过程（投票模式） ----------------------
    def step_year_voting(self):
        # 第1步：推进现有施工项目
        self._advance_projects()

        #utility
        utils = [dev.compute_parcel_utility(p, self.current_year) for p in self.parcels for dev in self.developers]
        print(f"\n=== Year {self.current_year} Utility Distribution ===")
        print (np.percentile(utils, [0,25,50,75,90,95,99,100]))

        vote_quota = int(self.voting.get("vote_quota_per_dev", 1))
        global_cap = int(self.voting.get("max_starts_per_year_global", 10**9))
        per_type_caps = self.voting.get("max_starts_per_year_by_type") or {}
        # 复制一份剩余额度
        per_type_remaining = {k: int(v) for k, v in per_type_caps.items()}
        util_thr = float(self.voting["util_threshold"])
        vote_thr = int(self.voting["vote_threshold"])
        K = len(self.developers)
        c_cfg = self.competition_cfg
        w_type = c_cfg['w_type']; w_fr = c_cfg['w_frustration']
        decay = c_cfg['frustration_decay']
        expand_thr = c_cfg['frustration_expand_threshold']
        expand_factor = c_cfg['expand_factor']
        lower_step = c_cfg['lower_util_step']
        min_thr = c_cfg['min_util_threshold']

        # 第2步：投票（记录开发商和效用）
        # votes[parcel_id] = List[(DeveloperAgent, utility, comp_score)]
        votes: Dict[int, List[Tuple[DeveloperAgent, float, float]]] = {}
        for dev in self.developers:
            candidates = [
                p for p in self.parcels
                if p.state.build_allowed
                and not p.state.risk_mask
                and p.state.floor_area <= 0
                and not p.state.under_construction
                and getattr(p.state, "land_use_2008","") == "NonUrban"
                and ( self._adjacent_to_seed(p, use_2008_as_seed=True) 
                    or random.random() < self.config.get("growth", {}).get("leapfrog_prob", 0.0) )
            ]
            if not candidates:
                continue
            # 动态扩大采样 (挫败次数高)
            base_sample = int(self.voting["candidate_sample"])
            if dev.frustration >= expand_thr:
                base_sample = int(base_sample * expand_factor)
            sample_n = min(base_sample, len(candidates))
            pool = random.sample(candidates, sample_n)
            scored: List[Tuple[ParcelAgent, float]] = []
            # 针对挫败开发商，动态降低效用阈值
            eff_thr = util_thr
            if dev.frustration > 0:
                eff_thr = max(util_thr - dev.frustration * lower_step, min_thr)
            for p in pool:
                u = dev.compute_parcel_utility(p, self.current_year)
                if u >= eff_thr:
                    scored.append((p, u))
            # 取前 vote_quota 个
            if scored:
                scored.sort(key=lambda x: x[1], reverse=True)
                for p, u in scored[:vote_quota]:
                    comp_score = w_type * dev.type_score + w_fr * dev.frustration
                    votes.setdefault(p.unique_id, []).append((dev, u, comp_score))

        # 第3步：确定满足票数阈值的候选项目，并应用年度开工上限
        candidate_projects = []  # List[dict]
        for pid, bidders in votes.items():
            if len(bidders) < vote_thr:
                continue
            parcel = self.parcels[pid]
            if parcel.state.floor_area > 0 or parcel.state.under_construction:
                continue
            winner_dev, winner_util, winner_comp = max(bidders, key=lambda x: (x[1], x[2]))
            candidate_projects.append({
                'pid': pid,
                'parcel': parcel,
                'bidders': bidders,
                'winner_dev': winner_dev,
                'winner_util': winner_util,
                'winner_comp': winner_comp,
            })

        # 按胜者效用排序（可加入第二关键字 winner_comp）
        candidate_projects.sort(key=lambda d: (d['winner_util'], d['winner_comp']), reverse=True)

        started_count = 0
        started_by_type = {t: 0 for t in per_type_remaining.keys()}
        actually_started = []
        trimmed = []
        trimmed_concurrency = []  # 因并发上限被裁掉

        for cand in candidate_projects:
            if started_count >= global_cap:
                trimmed.append(cand)
                continue
            wdev = cand['winner_dev']
            dt = wdev.developer_type
            # 并发上限检查（如果该类型在配置中存在）
            mc_limit = self.max_concurrent.get(dt)
            if mc_limit is not None and len(wdev.active_projects) >= mc_limit:
                trimmed_concurrency.append(cand)
                continue
            # 类型配额检查（若配置）
            if dt in per_type_remaining:
                if per_type_remaining[dt] <= 0:
                    trimmed.append(cand)
                    continue
            # 通过，创建项目
            duration = self.sample_duration(wdev.developer_type)
            project = Project(
                project_id=self._next_project_id,
                developer_id=wdev.unique_id,
                parcel_id=cand['pid'],
                start_year=self.current_year,
                duration=duration,
                rebuild=False,
                cost=0.0,
                utility_at_start=cand['winner_util'],
                original_duration=duration
            )
            self.projects.append(project)
            self._next_project_id += 1
            cand['parcel'].state.under_construction = True
            wdev.active_projects.append(project.project_id)
            actually_started.append(cand)
            started_count += 1
            if dt in per_type_remaining:
                per_type_remaining[dt] -= 1
                started_by_type[dt] = started_by_type.get(dt, 0) + 1

        # 第4步：更新挫败 / 成功（仅对真正开工的项目胜者减挫败，未开工的候选全部算失败）
        for cand in actually_started:
            wdev = cand['winner_dev']
            bidders = cand['bidders']
            wdev.successes += 1
            wdev.frustration = max(0, wdev.frustration - decay)
            for d, _, _ in bidders:
                if d is not wdev:
                    d.frustration = min(d.frustration + 1, c_cfg['cap'])
        for cand in trimmed:
            # 所有竞标者挫败 +1
            for d, _, _ in cand['bidders']:
                d.frustration = min(d.frustration + 1, c_cfg['cap'])
        self._recompute_neighborhood()
        # 第5步：记录最后一年潜力（保持不变）
        if self.current_year == self.end_year:
            frac = np.zeros(len(self.parcels), dtype=np.float32)
            for i, p in enumerate(self.parcels):
                cnt = 0
                for dev in self.developers:
                    u = dev.compute_parcel_utility(p, self.current_year)
                    if u >= util_thr:
                        cnt += 1
                frac[i] = cnt / float(K) if K > 0 else 0.0
            self._potential_fraction_last = frac

        # 记录当年开工数量及裁剪统计
        self._last_year_starts = started_count
        self._last_year_trimmed_by_cap = len(trimmed)
        self._last_year_trimmed_by_concurrency = len(trimmed_concurrency)
        self._record_year_stats()
        self.current_year += 1

    # ---------------------- 统计、邻里效应、施工推进 ----------------------
    def _record_year_stats(self):
        developed_count = sum(1 for p in self.parcels if p.state.floor_area > 0)
        avg_capital = sum(d.capital for d in self.developers) / max(1, len(self.developers))
        active_projects = sum(1 for pr in self.projects if pr.duration > 0)
        completed_this_year = sum(1 for pr in self.projects if pr.duration == 0 and pr.start_year < self.current_year)
        avg_frustration = sum(d.frustration for d in self.developers) / max(1, len(self.developers))
        total_successes = sum(d.successes for d in self.developers)
        starts_this_year = getattr(self, '_last_year_starts', 0)
        trimmed_by_cap = getattr(self, '_last_year_trimmed_by_cap', 0)
        trimmed_by_concurrency = getattr(self, '_last_year_trimmed_by_concurrency', 0)
        self.yearly_stats.append({
            "year": self.current_year,
            "developed_parcels": developed_count,
            "avg_developer_capital": avg_capital,
            "active_projects": active_projects,
            "completed_projects": completed_this_year,
            "avg_frustration": avg_frustration,
            "total_successes": total_successes,
            "starts_this_year": starts_this_year,
            "trimmed_by_cap": trimmed_by_cap,
            "trimmed_by_concurrency": trimmed_by_concurrency,
        })

    def _advance_projects(self):
        for project in list(self.projects):
            if project.duration <= 0:
                continue
            project.duration -= 1
            if project.duration == 0:
                dev = self._developer_by_id(project.developer_id)
                parcel = self.parcels[project.parcel_id]
                parcel.apply_development(dev, self.current_year, rebuild=project.rebuild)
                parcel.state.under_construction = False  # 完工解除施工状态
                # 若是该地块首次开发，记录开发商类型
                if not parcel.state.developed_by_type:
                    parcel.state.developed_by_type = dev.developer_type
                if project.project_id in dev.active_projects:
                    dev.active_projects.remove(project.project_id)

    def _developer_by_id(self, dev_id: int) -> DeveloperAgent:
        for d in self.developers:
            if d.unique_id == dev_id:
                return d
        raise KeyError(dev_id)

    def sample_duration(self, developer_type: str) -> int:
        lo, hi = self.duration_ranges.get(developer_type, (1, 3))
        return random.randint(lo, hi)

    def _recompute_neighborhood(self):
        if any(p.state.row is not None for p in self.parcels):
            index = {}
            for p in self.parcels:
                if p.state.row is not None and p.state.col is not None:
                    index[(p.state.row, p.state.col)] = p
            window_scores = {}
            for p in self.parcels:
                if p.state.row is None:
                    continue
                dev_count, total = 0, 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0: continue
                        coord = (p.state.row + dr, p.state.col + dc)
                        if coord in index:
                            neighbor = index[coord]
                            total += 1
                            if neighbor.state.floor_area > 0:
                                dev_count += 1
                window_scores[p.unique_id] = dev_count / total if total > 0 else 0.0
            self._window_scores = window_scores
            self._developed_ratio = sum(1 for p in self.parcels if p.state.floor_area > 0) / (len(self.parcels) + 1e-9)
            # 预计算到最近已开发地块的曼哈顿距离 (简化) 用于扩散 (可替换为更快算法)
            try:
                if self._grid_shape is not None:
                    H, W = self._grid_shape
                    dev_mask = np.full((H, W), False, dtype=bool)
                    for p in self.parcels:
                        r = p.state.row; c = p.state.col
                        if r is None or c is None: continue
                        if p.state.floor_area > 0 or p.state.under_construction:
                            dev_mask[r, c] = True
                    # 若尚无任何已开发，设为 NaN
                    if not dev_mask.any():
                        self._dev_distance = np.full((H, W), np.nan, dtype=float)
                    else:
                        # BFS 多源距离
                        from collections import deque
                        dist = np.full((H, W), np.inf, dtype=float)
                        q = deque()
                        for r in range(H):
                            for c in range(W):
                                if dev_mask[r, c]:
                                    dist[r, c] = 0.0
                                    q.append((r, c))
                        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
                        while q:
                            r, c = q.popleft()
                            d0 = dist[r, c]
                            for dr, dc in dirs:
                                nr, nc = r+dr, c+dc
                                if 0 <= nr < H and 0 <= nc < W:
                                    if dist[nr, nc] > d0 + 1:
                                        dist[nr, nc] = d0 + 1
                                        q.append((nr, nc))
                        self._dev_distance = dist
            except Exception as e:
                # 容错，不中断主流程
                self._dev_distance = None
        else:
            self._developed_ratio = sum(1 for p in self.parcels if p.state.floor_area > 0) / (len(self.parcels) + 1e-9)
            self._window_scores = {}

    def neighborhood_score(self, parcel: ParcelAgent) -> float:
        if hasattr(self, '_window_scores') and parcel.unique_id in getattr(self, '_window_scores'):
            local = self._window_scores[parcel.unique_id]
            return 0.7 * local + 0.3 * self._developed_ratio
        return self._developed_ratio

    def select_candidate_parcels(self, developer: DeveloperAgent) -> Iterable[ParcelAgent]:
        # 投票模式不使用该函数；留作兼容
        sample_size = min(10, len(self.parcels))
        return random.sample(self.parcels, sample_size)

    # ---------------------- 栅格输出 ----------------------
    def write_raster_outputs(self):
        if (self._ref_profile is None) or (self._grid_shape is None) or (rasterio is None):
            print("No raster profile; skip GeoTIFF export.")
            return
        H, W = self._grid_shape
        dev_bin = np.zeros((H, W), dtype=np.uint8)
        pot = np.zeros((H, W), dtype=np.float32)
        dev_type = np.zeros((H, W), dtype=np.uint8)  # 0 未开发; 1..n 对应开发商类型编码

        # 类型编码映射
        type_codes = {t: i+1 for i, t in enumerate(DEVELOPER_TYPES)}

        for p in self.parcels:
            r = getattr(p.state, "row", None)
            c = getattr(p.state, "col", None)
            if r is None or c is None:
                continue
            if p.state.floor_area > 0:
                dev_bin[r, c] = 1
                if p.state.developed_by_type:
                    dev_type[r, c] = type_codes.get(p.state.developed_by_type, 0)

        if hasattr(self, "_potential_fraction_last"):
            for i, p in enumerate(self.parcels):
                r = getattr(p.state, "row", None)
                c = getattr(p.state, "col", None)
                if r is None or c is None:
                    continue
                pot[r, c] = float(self._potential_fraction_last[i])

        out_dir = self.output_cfg.get("dir", "output")
        os.makedirs(out_dir, exist_ok=True)
        dev_path = self.output_cfg.get("developed_tif", os.path.join(out_dir, "developed.tif"))
        pot_path = self.output_cfg.get("potential_tif", os.path.join(out_dir, "potential.tif"))
        type_path = self.output_cfg.get("developed_type_tif", os.path.join(out_dir, "developed_type.tif"))

        prof_bin = self._ref_profile.copy(); prof_bin.update(dtype=rasterio.uint8, count=1, nodata=0)
        prof_pot = self._ref_profile.copy(); prof_pot.update(dtype=rasterio.float32, count=1, nodata=np.nan)

        with rasterio.open(dev_path, "w", **prof_bin) as dst:
            dst.write(dev_bin, 1)
        with rasterio.open(pot_path, "w", **prof_pot) as dst:
            dst.write(pot.astype(np.float32), 1)
        # 写开发类型栅格（同 developed 二进制 profile，但 nodata=0）
        prof_type = self._ref_profile.copy(); prof_type.update(dtype=rasterio.uint8, count=1, nodata=0)
        with rasterio.open(type_path, "w", **prof_type) as dst:
            dst.write(dev_type, 1)

        # 输出 per-type 统计 CSV
        csv_path = os.path.join(out_dir, "developer_type_summary.csv")
        try:
            import csv
            counts = {t: 0 for t in DEVELOPER_TYPES}
            for p in self.parcels:
                if p.state.floor_area > 0 and p.state.developed_by_type:
                    counts[p.state.developed_by_type] += 1
            total_dev = sum(counts.values()) or 1
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["developer_type", "developed_parcels", "share"])
                for t in DEVELOPER_TYPES:
                    v = counts[t]
                    writer.writerow([t, v, v/total_dev])
        except Exception as e:
            print(f"Warn: cannot write developer_type_summary.csv: {e}")

        print(f"✓ 写出开发栅格: {dev_path}\n✓ 写出潜力栅格(比例): {pot_path}\n✓ 写出开发类型栅格: {type_path}\n✓ 开发类型统计: {csv_path}")

    # ---------------------- 运行入口 ----------------------
    def run_model(self):
        if self.voting.get("enabled", False):
            total_years = self.end_year - self.current_year + 1
            use_bar = (tqdm is not None) and (self.config.get('progress', {}).get('enabled', True))
            pbar = tqdm(total=total_years, desc="Years", disable=not use_bar) if use_bar else None
            # 采用显式 while 以保持现有 step_year_voting 内部 year 自增逻辑
            while self.current_year <= self.end_year:
                # 年度公共过程 (年初状态更新)
                for parcel in self.parcels:
                    parcel.step()
                self._recompute_neighborhood()
                # 投票式同时开发 (内部会把 current_year 自增)
                self.step_year_voting()
                if pbar:
                    # 更新进度：一个 simulation year 完成
                    pbar.update(1)
                    # 显示少量关键信息
                    last_stat = self.yearly_stats[-1] if self.yearly_stats else {}
                    pbar.set_postfix({
                        'year': last_stat.get('year', self.current_year-1),
                        'starts': last_stat.get('starts_this_year', 0),
                        'dev%': f"{100.0 * (last_stat.get('developed_parcels',0) / max(1,len(self.parcels))):.1f}"
                    })
            if pbar:
                pbar.close()
            self.write_raster_outputs()
        else:
            total_years = self.end_year - self.current_year + 1
            use_bar = (tqdm is not None) and (self.config.get('progress', {}).get('enabled', True))
            pbar = tqdm(total=total_years, desc="Years", disable=not use_bar) if use_bar else None
            while self.current_year <= self.end_year:
                self.step_year()
                if pbar:
                    pbar.update(1)
                    last_stat = self.yearly_stats[-1] if self.yearly_stats else {}
                    pbar.set_postfix({
                        'year': last_stat.get('year', self.current_year-1),
                        'dev%': f"{100.0 * (last_stat.get('developed_parcels',0) / max(1,len(self.parcels))):.1f}"
                    })
            if pbar:
                pbar.close()
            self.write_raster_outputs()

# ---------------------------------------------------------------------------
# 简单自检
# ---------------------------------------------------------------------------
def sanity_run():
    model = UrbanDevModel(n_parcels=50, seed=42)  # 若配置 raster，将被覆盖
    model.run_model()
    assert model.yearly_stats, "Yearly stats should not be empty"
    yrs = [s['year'] for s in model.yearly_stats]
    print(f"Simulated years: {yrs[0]}-{yrs[-1]}")
    print(f"Parcels: {len(model.parcels)}, Developers: {len(model.developers)}")

if __name__ == "__main__":
    sanity_run()
