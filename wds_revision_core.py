"""
wds_revision_core.py
====================

Single source of truth for the major-revision pipeline of
"Physics Constrained Generative False Data Injection Attacks and Adaptive
Explainable Defence for Water Distribution Systems".

All three notebooks (DT_WDS*, FDI_Attack1, FDI_Defence_WDS1) import this module
so that the attack side and the defence side share ONE configuration, ONE
temporal grid, ONE seed schedule, ONE dataset artefact and ONE set of metric
definitions (Reviewer 1 #11, Reviewer 2 major #6 and #8).

Nothing in this module fabricates data. Every function either computes from
simulation / model outputs or raises.

Contents
--------
  1.  RevisionConfig, set_all_seeds                        (R1-11)
  2.  load_network_strict, sanitize_inp                    (Net3 parse failure)
  3.  ClosedLoopWDS  - EPANET-toolkit closed loop at 5 min (R2-M1, R2-M6, R2-M8)
  4.  Sensor-FDI strategies S1-S5                          (R1-8, R2-M1)
  5.  physical_impact                                      (R1-8, R2-M8)
  6.  Windowing, labels, chronological split, scaling      (R2-M8 leakage)
  7.  detection_rate / evasion_rate (Eq. 18)               (R2-M2)
  8.  calibrate_threshold (single documented rule)         (R1-11)
  9.  mttd_from_onset                                      (MTTD bug)
 10.  friedman_nemenyi, wilson_ci, block_bootstrap_ci      (R2-M8 stats)
 11.  diagnose_ranking                                     (R2-M7)
 12.  LinearPPOThreshold (label-free state option)         (R1-10, R2-M4)
 13.  Explainability: grouped exact Shapley, LIME-lite,
      attention->sensor mapping, agreement, counterfactual (R1-9, R2-m8)
 14.  resampling_fidelity (5 min vs 1 h)                   (R2-M6)
 15.  Artefact I/O with manifest + hashes                  (reproducibility)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# 1. Configuration and seeds
# =============================================================================


@dataclass
class RevisionConfig:
    """Shared configuration for every notebook. Export with `to_json`."""

    # temporal grid (common to attack and defence -> R2-M6)
    timestep_s: int = 300
    duration_days: float = 7.0
    window: int = 30                # detector window W (samples)
    gen_window: int = 144           # generator window T (samples)
    stride: int = 1
    label_rule: str = "any"        # window attacked if ANY sample in attack interval

    # splits and seeds (common -> R1-11)
    split: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    seeds: Tuple[int, ...] = (11, 22, 33, 44, 55)

    # sensor model
    noise_frac: float = 0.01        # sigma = noise_frac * mean |signal| per channel
    p_min_m: float = 14.0           # minimum service pressure [m]

    # threat model (R2-M1): sensor FDI only, M_u = 0
    frac_pressure_compromised: float = 0.5
    frac_flow_compromised: float = 1.0 / 3.0
    compromise_control_tanks: bool = True  # level sensors used by the controller
    allow_command_injection: bool = False  # never True for reported results

    # attack episode
    attack_onset_frac: float = 0.62  # onset just after the train split so val AND test see it
    attack_duration_h: float = 48.0

    # strategies
    s1_bias_sigma: float = 3.0
    s2_replay_lag_h: float = 24.0
    s2_scale: float = 1.05
    s4_beta: float = 0.5
    s5_rounds: int = 5
    s5_candidates: int = 16

    # detection
    threshold_rule: str = "percentile"   # percentile of validation-normal scores
    threshold_percentile: float = 99.0

    # paths
    artefact_dir: str = "revision_artifacts"

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @property
    def steps_per_day(self) -> int:
        return int(round(86400 / self.timestep_s))


def set_all_seeds(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch. Documented in Sec. 7.7."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # noqa: WPS433

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# =============================================================================
# 2. Network loading (no silent fallback)
# =============================================================================

_ALLOWED_STAT = {"AVERAGED", "AVERAGE", "MINIMUM", "MAXIMUM", "RANGE", "NONE"}


def sanitize_inp(src: str | Path, dst: Optional[str | Path] = None) -> Path:
    """Fix [TIMES] 'Statistic' values that WNTR rejects.

    DT_WDS3 failed on the supplied Net3.inp with
    'Statistic must be one of AVERAGED, MINIMUM, MAXIMUM, RANGE or NONE' and
    silently fell back to a 20-node built-in network. This rewrites the
    offending line to NONE (a reporting option only; hydraulics unaffected).
    """
    src = Path(src)
    text = src.read_text(encoding="utf-8", errors="replace").splitlines()
    out, section, changed = [], None, False
    for line in text:
        s = line.strip()
        if s.startswith("["):
            section = s.upper()
        if section == "[TIMES]" and re.match(r"(?i)^\s*statistic\b", line):
            parts = line.split()
            val = parts[1].upper() if len(parts) > 1 else ""
            if val not in _ALLOWED_STAT:
                line = " Statistic            NONE"
                changed = True
        out.append(line)
    if dst is None:
        dst = Path(tempfile.gettempdir()) / f"sanitized_{src.name}"
    Path(dst).write_text("\n".join(out) + "\n", encoding="utf-8")
    if changed:
        print(f"[sanitize_inp] rewrote invalid [TIMES] Statistic in {src.name}")
    return Path(dst)


def load_network_strict(inp_path: str | Path):
    """Load an EPANET model with WNTR. Never falls back to a synthetic network."""
    import wntr

    inp_path = Path(inp_path)
    if not inp_path.exists():
        raise FileNotFoundError(f"INP not found: {inp_path}")
    try:
        return wntr.network.WaterNetworkModel(str(inp_path)), inp_path
    except Exception as exc:  # retry once after sanitising
        clean = sanitize_inp(inp_path)
        try:
            return wntr.network.WaterNetworkModel(str(clean)), clean
        except Exception as exc2:
            raise RuntimeError(
                f"Could not parse {inp_path} (original error: {exc}; after sanitising: {exc2}). "
                "Refusing to substitute a synthetic network."
            ) from exc2


def bundled_inp(name: str) -> Path:
    """Path of the reference Net1/Net2/Net3 shipped with WNTR (for reproducibility)."""
    import wntr

    p = Path(wntr.__file__).parent / "library" / "networks" / f"{name}.inp"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


# =============================================================================
# 3. Closed-loop simulator (EPANET toolkit, SCADA polling every timestep_s)
# =============================================================================

# EPANET toolkit codes
EN_ELEVATION, EN_BASEDEMAND, EN_DEMAND, EN_HEAD, EN_PRESSURE = 0, 1, 9, 10, 11
EN_FLOW, EN_STATUS, EN_ENERGY = 8, 11, 13
EN_NODECOUNT, EN_LINKCOUNT, EN_CONTROLCOUNT = 0, 2, 5
EN_JUNCTION, EN_RESERVOIR, EN_TANK = 0, 1, 2
EN_PUMP = 2
EN_DURATION, EN_HYDSTEP, EN_REPORTSTEP = 0, 1, 4
_US_FLOW_UNITS = {0, 1, 2, 3, 4}   # CFS GPM MGD IMGD AFD
_TO_M3S = {0: 0.0283168, 1: 6.30902e-5, 2: 0.0438126, 3: 0.0526168, 4: 0.0142764,
           5: 1e-3, 6: 1.66667e-5, 7: 0.0115741, 8: 1 / 3600.0, 9: 1 / 86400.0}


@dataclass
class LevelRule:
    """SCADA rule: pump OPEN when measured tank level < low, CLOSED when > high."""
    pump: str
    tank: str
    low: float   # [m]
    high: float  # [m]
    synthetic: bool = False  # True if generated because the INP had no level rule


def derive_level_rules(wn) -> List[LevelRule]:
    """Level rules from INP controls; generate documented rules for pumps without one.

    Net1: rules taken from the INP. Net3: INP controls are time-based, so level
    rules are generated (30 % / 80 % of the level range of the nearest tank) and
    flagged `synthetic=True` - this MUST be stated in the paper. Net2 has no pumps:
    returns [] and sensor FDI then cannot change the hydraulics (report this).
    """
    import networkx as nx

    rules: Dict[str, LevelRule] = {}
    for cname in wn.control_name_list:
        c = wn.get_control(cname)
        txt = str(c)
        m = re.match(r"IF TANK (\S+) LEVEL (BELOW|ABOVE) ([\d.eE+-]+) THEN PUMP (\S+) STATUS IS (OPEN|CLOSED)", txt)
        if not m:
            continue
        tank, rel, val, pump, st = m.groups()
        r = rules.get(pump, LevelRule(pump, tank, np.nan, np.nan))
        if rel == "BELOW" and st == "OPEN":
            r.low = float(val)
        if rel == "ABOVE" and st == "CLOSED":
            r.high = float(val)
        rules[pump] = r
    tanks = list(wn.tank_name_list)
    G = wn.to_graph().to_undirected()
    for pump in wn.pump_name_list:
        r = rules.get(pump)
        if r is not None and np.isfinite(r.low) and np.isfinite(r.high):
            continue
        if not tanks:
            continue
        p = wn.get_link(pump)
        dists = {t: nx.shortest_path_length(G, p.end_node_name, t) if nx.has_path(G, p.end_node_name, t) else 1e9
                 for t in tanks}
        t = min(dists, key=dists.get)
        tk = wn.get_node(t)
        lo = tk.min_level + 0.3 * (tk.max_level - tk.min_level)
        hi = tk.min_level + 0.8 * (tk.max_level - tk.min_level)
        rules[pump] = LevelRule(pump, t, float(lo), float(hi), synthetic=True)
    return list(rules.values())


@dataclass
class SensorLayout:
    names: List[str]            # e.g. "pressure::10", "flow::9", "tank_level::2"
    kinds: List[str]            # pressure / flow / tank_level
    ids: List[str]

    @property
    def m(self) -> int:
        return len(self.names)

    def index(self, kind: str) -> np.ndarray:
        return np.array([i for i, k in enumerate(self.kinds) if k == kind], dtype=int)


def build_sensor_layout(wn) -> SensorLayout:
    """Pressure at junctions, flow on all links, level at tanks (Table 2)."""
    names, kinds, ids = [], [], []
    for j in wn.junction_name_list:
        names.append(f"pressure::{j}"); kinds.append("pressure"); ids.append(j)
    for l in wn.link_name_list:
        names.append(f"flow::{l}"); kinds.append("flow"); ids.append(l)
    for t in wn.tank_name_list:
        names.append(f"tank_level::{t}"); kinds.append("tank_level"); ids.append(t)
    return SensorLayout(names, kinds, ids)


def select_compromised(layout: SensorLayout, rules: List[LevelRule], cfg: RevisionConfig,
                       seed: int = 0) -> np.ndarray:
    """Coverage-based compromised set S_a (Table 3). Deterministic given seed."""
    rng = np.random.default_rng(seed)
    idx: List[int] = []
    p = layout.index("pressure")
    idx += list(p[: int(math.ceil(cfg.frac_pressure_compromised * len(p)))])
    f = layout.index("flow")
    k = int(math.ceil(cfg.frac_flow_compromised * len(f)))
    idx += list(rng.choice(f, size=k, replace=False)) if k > 0 else []
    if cfg.compromise_control_tanks:
        ctrl_tanks = {r.tank for r in rules}
        idx += [i for i, (kd, i_d) in enumerate(zip(layout.kinds, layout.ids))
                if kd == "tank_level" and i_d in ctrl_tanks]
    return np.array(sorted(set(int(i) for i in idx)), dtype=int)


# attack callback: (k, y_meas_vector, history_meas[:k], context) -> y_tilde vector
AttackFn = Callable[[int, np.ndarray, np.ndarray, dict], np.ndarray]


@dataclass
class ClosedLoopResult:
    t_s: np.ndarray                 # [N] seconds
    y_true: np.ndarray              # [N, m] true sensor values (SI)
    y_meas: np.ndarray              # [N, m] noisy measurement before attack
    y_scada: np.ndarray             # [N, m] values received by SCADA (after attack)
    label: np.ndarray               # [N] 1 during attack interval
    pump_status: np.ndarray         # [N, n_pumps]
    pump_power_kw: np.ndarray       # [N, n_pumps]
    junction_pressure: np.ndarray   # [N, n_junctions] true
    junction_demand: np.ndarray     # [N, n_junctions] true
    tank_level: np.ndarray          # [N, n_tanks] true
    layout: SensorLayout
    pumps: List[str]
    junctions: List[str]
    tanks: List[str]
    rules: List[LevelRule]
    compromised: np.ndarray
    meta: dict


class ClosedLoopWDS:
    """Closed-loop sensor-FDI simulation with the EPANET 2.2 toolkit.

    The INP controls are removed and replaced by a Python SCADA controller that
    polls sensors every `timestep_s` and switches pumps from the *received*
    (possibly falsified) tank levels (Eqs. 11-14 of the paper). The attacker
    modifies measurements only (M_u = 0).
    """

    def __init__(self, inp_path: str | Path, cfg: RevisionConfig,
                 rules: Optional[List[LevelRule]] = None):
        self.cfg = cfg
        self.wn, self.inp_path = load_network_strict(inp_path)
        self.rules = rules if rules is not None else derive_level_rules(self.wn)
        self.layout = build_sensor_layout(self.wn)

    # ------------------------------------------------------------------
    def run(self, seed: int, attack: Optional[AttackFn] = None,
            attack_window: Optional[Tuple[int, int]] = None,
            compromised: Optional[np.ndarray] = None,
            noise_scale: Optional[np.ndarray] = None,
            context: Optional[dict] = None) -> ClosedLoopResult:
        from wntr.epanet.toolkit import ENepanet

        cfg = self.cfg
        rng = np.random.default_rng(seed)
        tmp = Path(tempfile.mkdtemp(prefix="clwds_"))
        inp = tmp / "net.inp"
        shutil.copy(self.inp_path, inp)
        en = ENepanet()
        en.ENopen(str(inp), str(tmp / "net.rpt"), str(tmp / "net.bin"))
        try:
            n_ctrl = en.ENgetcount(EN_CONTROLCOUNT)
            for ci in range(n_ctrl, 0, -1):
                en.ENdeletecontrol(ci)
            dur = int(cfg.duration_days * 86400)
            en.ENsettimeparam(EN_DURATION, dur)
            en.ENsettimeparam(EN_HYDSTEP, cfg.timestep_s)
            en.ENsettimeparam(EN_REPORTSTEP, cfg.timestep_s)
            fu = en.ENgetflowunits()
            us = fu in _US_FLOW_UNITS
            p_conv = 0.70307 if us else 1.0      # psi -> m
            h_conv = 0.3048 if us else 1.0       # ft -> m
            q_conv = _TO_M3S[fu]

            lay = self.layout
            nidx = {n: en.ENgetnodeindex(n) for n in self.wn.node_name_list}
            lidx = {l: en.ENgetlinkindex(l) for l in self.wn.link_name_list}
            junctions = list(self.wn.junction_name_list)
            tanks = list(self.wn.tank_name_list)
            pumps = list(self.wn.pump_name_list)
            tank_elev = {t: en.ENgetnodevalue(nidx[t], EN_ELEVATION) for t in tanks}
            pos = {n: i for i, n in enumerate(lay.names)}

            N = int(dur // cfg.timestep_s) + 1
            m = lay.m
            Y = np.zeros((N, m)); JP = np.zeros((N, len(junctions))); JD = np.zeros_like(JP)
            TL = np.zeros((N, len(tanks))); PS = np.zeros((N, len(pumps))); PW = np.zeros_like(PS)
            Ym = np.zeros_like(Y); Ys = np.zeros_like(Y); lab = np.zeros(N, dtype=np.int8)

            comp = np.array([], dtype=int) if compromised is None else np.asarray(compromised, int)
            ctx = dict(context or {})
            ctx.update(layout=lay, compromised=comp, cfg=cfg, rng=rng)

            en.ENopenH(); en.ENinitH(0)
            k, t, next_sample = 0, 0, 0
            while True:
                t = en.ENrunH()
                if t >= next_sample and k < N:
                    next_sample += cfg.timestep_s
                    # --- true state -> sensors
                    for i, name in enumerate(lay.names):
                        kind, oid = name.split("::", 1)
                        if kind == "pressure":
                            Y[k, i] = en.ENgetnodevalue(nidx[oid], EN_PRESSURE) * p_conv
                        elif kind == "flow":
                            Y[k, i] = en.ENgetlinkvalue(lidx[oid], EN_FLOW) * q_conv
                        else:
                            Y[k, i] = (en.ENgetnodevalue(nidx[oid], EN_HEAD) - tank_elev[oid]) * h_conv
                    for i, j in enumerate(junctions):
                        JP[k, i] = en.ENgetnodevalue(nidx[j], EN_PRESSURE) * p_conv
                        JD[k, i] = en.ENgetnodevalue(nidx[j], EN_DEMAND) * q_conv
                    for i, tk in enumerate(tanks):
                        TL[k, i] = (en.ENgetnodevalue(nidx[tk], EN_HEAD) - tank_elev[tk]) * h_conv
                    for i, p in enumerate(pumps):
                        PS[k, i] = en.ENgetlinkvalue(lidx[p], EN_STATUS)
                        PW[k, i] = en.ENgetlinkvalue(lidx[p], EN_ENERGY)   # kW
                    scale = noise_scale if noise_scale is not None else np.maximum(np.abs(Y[k]), 1e-6) * cfg.noise_frac
                    Ym[k] = Y[k] + rng.normal(0.0, 1.0, m) * scale
                    ytil = Ym[k].copy()
                    if attack is not None and attack_window is not None:
                        wins = attack_window if isinstance(attack_window, list) else [attack_window]
                        in_attack = any(a <= k < b for a, b in wins)
                    else:
                        in_attack = False
                    if in_attack:
                        cand = attack(k, Ym[k].copy(), Ys[:k], ctx)
                        ytil[comp] = cand[comp]        # M_y restricts to S_a
                        lab[k] = 1
                    Ys[k] = ytil
                    # --- SCADA controller acts on RECEIVED levels
                    for r in self.rules:
                        lvl = ytil[pos[f"tank_level::{r.tank}"]]
                        li = lidx[r.pump]
                        if lvl < r.low:
                            en.ENsetlinkvalue(li, EN_STATUS, 1.0)
                        elif lvl > r.high:
                            en.ENsetlinkvalue(li, EN_STATUS, 0.0)
                    k += 1
                tstep = en.ENnextH()
                if tstep <= 0:
                    break
            en.ENcloseH()
        finally:
            en.ENclose()
            shutil.rmtree(tmp, ignore_errors=True)
        k = min(k, N)
        sl = slice(0, k)
        return ClosedLoopResult(
            t_s=np.arange(k) * cfg.timestep_s, y_true=Y[sl], y_meas=Ym[sl], y_scada=Ys[sl],
            label=lab[sl], pump_status=PS[sl], pump_power_kw=PW[sl], junction_pressure=JP[sl],
            junction_demand=JD[sl], tank_level=TL[sl], layout=self.layout, pumps=pumps,
            junctions=junctions, tanks=tanks, rules=self.rules, compromised=comp,
            meta=dict(seed=seed, inp=str(self.inp_path), timestep_s=cfg.timestep_s,
                      attack_window=attack_window, flow_units=int(fu),
                      synthetic_rules=[r.pump for r in self.rules if r.synthetic]))


# =============================================================================
# 4. Sensor-FDI strategies (all act on measurements only)
# =============================================================================


def attack_window_for(n_steps: int, cfg: RevisionConfig) -> Tuple[int, int]:
    on = int(cfg.attack_onset_frac * n_steps)
    return on, min(n_steps, on + int(cfg.attack_duration_h * 3600 / cfg.timestep_s))

def attack_episodes(n_steps: int, cfg: RevisionConfig, n_episodes: int = 6,
                    episode_h: Optional[float] = None, gap_frac: float = 0.5
                    ) -> List[Tuple[int, int]]:
    """Evenly spaced attack episodes so that every chronological split (train/val/
    test) contains both attacked and normal windows (needed for a valid AUROC on
    each split and for training the supervised RF / PPO reward). Returns a list of
    (onset, stop) index pairs. Episodes never touch the very start (detectors need
    clean warm-up) and never overlap.
    """
    dur = int((episode_h or cfg.attack_duration_h) * 3600 / cfg.timestep_s)
    dur = max(cfg.window, min(dur, n_steps // (2 * n_episodes)))
    seg = n_steps // n_episodes
    eps = []
    for i in range(n_episodes):
        onset = i * seg + int(gap_frac * (seg - dur))
        onset = max(cfg.window, min(onset, n_steps - dur - 1))
        eps.append((onset, onset + dur))
    return eps


def combined_window(n_steps: int, cfg: RevisionConfig, episodes: List[Tuple[int, int]]):
    """Return (episodes, label_time) with a mask marking all episode intervals."""
    lab = np.zeros(n_steps, dtype=np.int8)
    for a, b in episodes:
        lab[a:b] = 1
    return episodes, lab


def make_s1_constant_bias(sigma: np.ndarray, cfg: RevisionConfig) -> AttackFn:
    """S1: y + b*sigma on compromised channels (tank levels biased upward to stop pumps)."""
    def f(k, y, hist, ctx):
        return y + cfg.s1_bias_sigma * sigma
    return f


def make_s2_scaled_replay(cfg: RevisionConfig) -> AttackFn:
    """S2: replay the measurement received `lag` earlier, scaled by s2_scale."""
    lag = int(cfg.s2_replay_lag_h * 3600 / cfg.timestep_s)

    def f(k, y, hist, ctx):
        if k - lag < 0 or len(hist) < k - lag + 1:
            return y
        return cfg.s2_scale * hist[k - lag]
    return f


def make_s3_gan_pure(gen_sequence: np.ndarray, onset: int) -> AttackFn:
    """S3: replace compromised channels with a generated sequence [L, m] (SI units)."""
    def f(k, y, hist, ctx):
        i = min(k - onset, len(gen_sequence) - 1)
        return gen_sequence[i]
    return f


def make_s4_gan_blend(gen_sequence: np.ndarray, onset: int, beta: float) -> AttackFn:
    """S4: (1-beta) * y + beta * generated."""
    def f(k, y, hist, ctx):
        i = min(k - onset, len(gen_sequence) - 1)
        return (1.0 - beta) * y + beta * gen_sequence[i]
    return f


def select_adaptive_sequence(candidates: Sequence[np.ndarray], score_fn: Callable[[np.ndarray], float],
                             rounds: int, refine_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None
                             ) -> Tuple[np.ndarray, List[float]]:
    """S5: detector-in-the-loop adaptation by iterative selection (+ optional refinement).

    Each round keeps the candidate with the LOWEST detector score and, if
    `refine_fn` is given (e.g. a generator fine-tuning step), produces new
    candidates from it. Describe S5 in the paper exactly as implemented.
    """
    pool = list(candidates)
    trace: List[float] = []
    best = None
    for r in range(rounds):
        scores = [float(score_fn(c)) for c in pool]
        i = int(np.argmin(scores))
        best = pool[i]
        trace.append(scores[i])
        if refine_fn is None:
            break
        pool = [refine_fn(best, j) for j in range(len(pool))]
    return best, trace


# =============================================================================
# 5. Physical impact (attack vs. baseline, same seed)
# =============================================================================


def physical_impact(attacked: ClosedLoopResult, baseline: ClosedLoopResult, cfg: RevisionConfig) -> Dict[str, float]:
    """Impact of ONE attack scenario relative to the no-attack run with the same seed.

    Demand-driven EPANET delivers full demand even at low pressure, so the
    demand deficit is the share of demand at junctions whose true pressure is
    below p_min (unserved-demand proxy) - state this in the paper.
    """
    h = cfg.timestep_s / 3600.0
    n = min(len(attacked.t_s), len(baseline.t_s))
    # EPANET reports very large negative pressures for nodes disconnected from all
    # sources (e.g. after a tank empties). Count them as violations, cap the deficit.
    disc = attacked.junction_pressure[:n] < -1e3
    P = np.where(disc, 0.0, attacked.junction_pressure[:n])
    Pb = np.where(baseline.junction_pressure[:n] < -1e3, 0.0, baseline.junction_pressure[:n])
    D, Db = np.clip(attacked.junction_demand[:n], 0, None), np.clip(baseline.junction_demand[:n], 0, None)
    viol = P < cfg.p_min_m
    viol_b = Pb < cfg.p_min_m
    energy = float(attacked.pump_power_kw[:n].sum() * h)
    energy_b = float(baseline.pump_power_kw[:n].sum() * h)
    return dict(
        disconnected_node_hours=float(disc.sum() * h),
        pressure_violation_node_hours=float(viol.sum() * h),
        pressure_violation_node_hours_excess=float((viol.sum() - viol_b.sum()) * h),
        max_pressure_deficit_m=float(np.max(np.clip(cfg.p_min_m - P, 0, None))) if P.size else 0.0,
        demand_deficit_ratio=float((D * viol).sum() / max(D.sum(), 1e-12)),
        demand_deficit_ratio_excess=float((D * viol).sum() / max(D.sum(), 1e-12)
                                          - (Db * viol_b).sum() / max(Db.sum(), 1e-12)),
        pump_energy_kwh=energy,
        extra_pump_energy_kwh=energy - energy_b,
        pump_switches=int(np.abs(np.diff(attacked.pump_status[:n], axis=0)).sum()),
        pump_switches_baseline=int(np.abs(np.diff(baseline.pump_status[:n], axis=0)).sum()),
        tank_level_min_m=float(attacked.tank_level[:n].min()) if attacked.tank_level.size else float("nan"),
        tank_level_max_m=float(attacked.tank_level[:n].max()) if attacked.tank_level.size else float("nan"),
    )


def tank_overflow_minutes(res: ClosedLoopResult, wn) -> float:
    mins = 0.0
    for i, t in enumerate(res.tanks):
        mx = wn.get_node(t).max_level
        mins += float((res.tank_level[:, i] >= mx - 1e-6).sum() * res.meta["timestep_s"] / 60.0)
    return mins


# =============================================================================
# 6. Windows, labels, split, scaling (leakage-free)
# =============================================================================


def chronological_bounds(n: int, split=(0.6, 0.2, 0.2)) -> Tuple[int, int]:
    return int(split[0] * n), int((split[0] + split[1]) * n)


def make_windows(X: np.ndarray, y: np.ndarray, W: int, stride: int = 1, rule: str = "any",
                 t0: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return windows [n, W, m], labels [n], end index [n] (absolute, + t0)."""
    starts = np.arange(0, len(X) - W + 1, stride)
    Xw = np.stack([X[s:s + W] for s in starts]).astype(np.float32)
    if rule == "any":
        yw = np.array([int(y[s:s + W].max()) for s in starts], dtype=np.int8)
    elif rule == "last":
        yw = np.array([int(y[s + W - 1]) for s in starts], dtype=np.int8)
    else:
        raise ValueError(rule)
    return Xw, yw, starts + W - 1 + t0


@dataclass
class SplitData:
    Xtr: np.ndarray; ytr: np.ndarray; etr: np.ndarray
    Xva: np.ndarray; yva: np.ndarray; eva: np.ndarray
    Xte: np.ndarray; yte: np.ndarray; ete: np.ndarray
    mean: np.ndarray; std: np.ndarray


def split_scale_window(Y: np.ndarray, label: np.ndarray, cfg: RevisionConfig) -> SplitData:
    """Split in time FIRST, fit scaling on normal training samples only, THEN window.

    Replaces the defence notebook's `StandardScaler().fit_transform(all)` +
    `train_test_split(test_size=0.3, shuffle)` (Reviewer 2 major #8).
    """
    n = len(Y)
    a, b = chronological_bounds(n, cfg.split)
    tr_norm = Y[:a][label[:a] == 0]
    if len(tr_norm) < cfg.window:
        raise ValueError("Training period contains too few normal samples.")
    mu, sd = tr_norm.mean(0), tr_norm.std(0) + 1e-8
    Z = (Y - mu) / sd
    Xtr, ytr, etr = make_windows(Z[:a], label[:a], cfg.window, cfg.stride, cfg.label_rule, 0)
    Xva, yva, eva = make_windows(Z[a:b], label[a:b], cfg.window, cfg.stride, cfg.label_rule, a)
    Xte, yte, ete = make_windows(Z[b:], label[b:], cfg.window, cfg.stride, cfg.label_rule, b)
    assert etr.max() < a <= eva.min() and eva.max() < b <= ete.min(), "window leakage across splits"
    return SplitData(Xtr, ytr, etr, Xva, yva, eva, Xte, yte, ete, mu, sd)


# =============================================================================
# 7-9. Detection rate (Eq. 18), threshold rule, MTTD
# =============================================================================


def detection_rate(decisions: np.ndarray, labels: np.ndarray, level: str = "window",
                   episodes: Optional[np.ndarray] = None) -> float:
    """DR at window level (fraction of attacked windows alarmed) or episode level.

    ER = 1 - DR. Smaller DR = stealthier (Eq. 18 corrected).
    """
    decisions, labels = np.asarray(decisions).astype(int), np.asarray(labels).astype(int)
    if level == "window":
        att = labels == 1
        return float(decisions[att].mean()) if att.any() else float("nan")
    if level == "episode":
        if episodes is None:
            raise ValueError("episode ids required")
        ids = np.unique(episodes[labels == 1])
        return float(np.mean([decisions[(episodes == e) & (labels == 1)].max() for e in ids])) if len(ids) else float("nan")
    raise ValueError(level)


def calibrate_threshold(val_normal_scores: np.ndarray, cfg: RevisionConfig) -> float:
    """THE single fixed-threshold rule used for every detector (Table A1)."""
    if cfg.threshold_rule != "percentile":
        raise ValueError(cfg.threshold_rule)
    return float(np.percentile(val_normal_scores, cfg.threshold_percentile))


def mttd_from_onset(decisions: np.ndarray, end_idx: np.ndarray, onset: int, stop: int,
                    timestep_s: int) -> float:
    """Seconds from attack onset to the first alarm among windows ending in [onset, stop).

    Returns NaN (not the horizon) when no alarm is raised; report the miss rate.
    Replaces `np.argmax(preds==1)` over a shuffled test set.
    """
    sel = (end_idx >= onset) & (end_idx < stop) & (np.asarray(decisions) == 1)
    if not sel.any():
        return float("nan")
    return float((end_idx[sel].min() - onset) * timestep_s)


# =============================================================================
# 10. Statistics
# =============================================================================


def friedman_nemenyi(M: np.ndarray, names: Sequence[str], alpha: float = 0.05,
                     higher_is_better: bool = True) -> dict:
    """Friedman omnibus + Nemenyi all-pairs (Demsar 2006).

    M: [N blocks, k methods]. Blocks must be genuinely distinct experimental
    units (strategy x network x seed), NOT copies of the same number.
    """
    from scipy.stats import friedmanchisquare, rankdata, studentized_range

    M = np.asarray(M, float)
    N, k = M.shape
    if N < 2 or k < 3:
        raise ValueError("need N>=2 blocks and k>=3 methods")
    if np.allclose(M, M[0:1, :]):
        raise ValueError("all blocks identical - statistics would be meaningless")
    stat, p = friedmanchisquare(*M.T)
    R = np.vstack([rankdata(-row if higher_is_better else row) for row in M])
    mean_ranks = R.mean(0)
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
    pairs = []
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            pval = float(studentized_range.sf(diff / se * np.sqrt(2), k, np.inf))
            pairs.append(dict(A=names[i], B=names[j], rank_diff=diff, p_nemenyi=pval, significant=diff > cd))
    return dict(chi2=float(stat), p=float(p), N=N, k=k, cd=float(cd),
                mean_ranks=pd.Series(mean_ranks, index=list(names)).sort_values(),
                pairs=pd.DataFrame(pairs))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def block_bootstrap_ci(y: np.ndarray, s: np.ndarray, metric: Callable, block: int, n_boot: int = 1000,
                       seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Moving-block bootstrap CI for a metric on time-ordered test windows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    nb = int(math.ceil(n / block))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block + 1), nb)
        idx = np.concatenate([np.arange(st, min(n, st + block)) for st in starts])[:n]
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], s[idx]))
    point = metric(y, s)
    if not vals:
        return point, float("nan"), float("nan")
    return float(point), float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


# =============================================================================
# 11. Diagnosis of near-chance ranking (R2-M7)
# =============================================================================


def diagnose_ranking(sd: SplitData, Y_clean_scaled: np.ndarray, Y_att_scaled: np.ndarray,
                     label_time: np.ndarray, noise_std_scaled: np.ndarray, cfg: RevisionConfig) -> dict:
    """Oracle AUROC, supervised probe AUROC, attack-to-noise ratio, label alignment."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    delta = Y_att_scaled - Y_clean_scaled
    att = label_time.astype(bool)
    anr = float(np.linalg.norm(delta[att], axis=1).mean() / max(np.linalg.norm(noise_std_scaled), 1e-12)) if att.any() else float("nan")
    oracle = float(roc_auc_score(label_time, np.linalg.norm(delta, axis=1))) if 0 < att.sum() < len(att) else float("nan")
    Xtr = np.concatenate([sd.Xtr, sd.Xva]).reshape(len(sd.Xtr) + len(sd.Xva), -1)
    ytr = np.concatenate([sd.ytr, sd.yva])
    probe = float("nan")
    if len(np.unique(ytr)) == 2 and len(np.unique(sd.yte)) == 2:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        probe = float(roc_auc_score(sd.yte, clf.predict_proba(sd.Xte.reshape(len(sd.Xte), -1))[:, 1]))
    return dict(attack_to_noise_ratio=anr, oracle_auroc=oracle, supervised_probe_auroc=probe,
                train_has_attacks=bool(ytr.any()), test_attack_fraction=float(sd.yte.mean()))


# =============================================================================
# 12. PPO threshold agent (linear softmax policy, NumPy)
# =============================================================================


class LinearPPOThreshold:
    """Proximal policy optimisation for threshold adjustment.

    Policy: softmax over {decrease, keep, increase} with linear features.
    Value : linear baseline. Clipped surrogate objective (Schulman et al. 2017).

    state_mode="label_free"  -> [mu, sigma, q50, q95, tau, alarm_rate, drift, 1]  (deployable)
    state_mode="label_based" -> [mu, sigma, TPR, FPR, tau, 1]                     (simulation only)
    The REWARD always uses labels and is therefore available only in the twin;
    the policy is trained offline and evaluated frozen (Sec. 6.3).
    """

    ACTIONS = np.array([-1, 0, 1])

    def __init__(self, state_mode: str = "label_free", horizon: int = 48, dtau: float = 0.02,
                 lam_tpr: float = 1.0, lam_fpr: float = 1.0, lam_move: float = 0.01,
                 clip: float = 0.2, gamma: float = 0.95, lr: float = 0.05, epochs: int = 8, seed: int = 0):
        assert state_mode in ("label_free", "label_based")
        self.state_mode, self.L, self.dtau = state_mode, horizon, dtau
        self.l1, self.l2, self.l3 = lam_tpr, lam_fpr, lam_move
        self.clip, self.gamma, self.lr, self.epochs = clip, gamma, lr, epochs
        self.rng = np.random.default_rng(seed)
        d = 8 if state_mode == "label_free" else 6
        self.theta = np.zeros((3, d))
        self.w = np.zeros(d)
        self.ref_mu = 0.0

    def _state(self, s: np.ndarray, y: Optional[np.ndarray], tau: float) -> np.ndarray:
        dec = s >= tau
        if self.state_mode == "label_free":
            return np.array([s.mean(), s.std(), np.quantile(s, .5), np.quantile(s, .95), tau,
                             dec.mean(), s.mean() - self.ref_mu, 1.0])
        tpr = dec[y == 1].mean() if (y == 1).any() else 0.0
        fpr = dec[y == 0].mean() if (y == 0).any() else 0.0
        return np.array([s.mean(), s.std(), tpr, fpr, tau, 1.0])

    def _pi(self, x):
        z = self.theta @ x
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def _reward(self, s, y, tau_new, tau_old):
        dec = s >= tau_new
        tpr = dec[y == 1].mean() if (y == 1).any() else 0.0
        fpr = dec[y == 0].mean() if (y == 0).any() else 0.0
        return self.l1 * tpr - self.l2 * fpr - self.l3 * abs(tau_new - tau_old)

    def fit(self, scores: np.ndarray, labels: np.ndarray, tau0: float, n_episodes: int = 30,
            normal_ref: Optional[np.ndarray] = None) -> "LinearPPOThreshold":
        """scores/labels: time-ordered TRAIN+VAL stream (labels from the twin)."""
        s_all = (scores - scores.min()) / (np.ptp(scores) + 1e-12)
        self.s_min, self.s_ptp = scores.min(), np.ptp(scores) + 1e-12
        tau0 = (tau0 - self.s_min) / self.s_ptp
        self.ref_mu = float(((normal_ref - self.s_min) / self.s_ptp).mean()) if normal_ref is not None else float(s_all.mean())
        T = len(s_all) // self.L
        for _ in range(n_episodes):
            tau, X, A, R, P = tau0, [], [], [], []
            for t in range(T):
                seg = slice(t * self.L, (t + 1) * self.L)
                x = self._state(s_all[seg], labels[seg], tau)
                pi = self._pi(x)
                a = int(self.rng.choice(3, p=pi))
                tau_new = float(np.clip(tau + self.ACTIONS[a] * self.dtau, 0.0, 1.0))
                nxt = slice((t + 1) * self.L, (t + 2) * self.L) if t + 1 < T else seg
                r = self._reward(s_all[nxt], labels[nxt], tau_new, tau)
                X.append(x); A.append(a); R.append(r); P.append(pi[a]); tau = tau_new
            X, A, R, P = np.array(X), np.array(A), np.array(R), np.array(P)
            G = np.zeros_like(R); g = 0.0
            for i in range(len(R) - 1, -1, -1):
                g = R[i] + self.gamma * g; G[i] = g
            for _e in range(self.epochs):
                V = X @ self.w
                adv = G - V
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                grad = np.zeros_like(self.theta)
                for x, a, ad, p_old in zip(X, A, adv, P):
                    pi = self._pi(x)
                    ratio = pi[a] / (p_old + 1e-12)
                    if (ad >= 0 and ratio < 1 + self.clip) or (ad < 0 and ratio > 1 - self.clip):
                        onehot = np.eye(3)[a]
                        grad += ratio * ad * np.outer(onehot - pi, x)
                self.theta += self.lr * grad / len(X)
                self.w += self.lr * ((G - V)[:, None] * X).mean(0)
        self.tau0 = tau0
        return self

    def run(self, scores: np.ndarray, labels_for_state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Frozen greedy policy on a TEST stream. Labels are used only if state_mode='label_based'."""
        if self.state_mode == "label_based" and labels_for_state is None:
            raise ValueError("label_based state needs labels (simulation only)")
        s = (scores - self.s_min) / self.s_ptp
        tau = self.tau0
        dec = np.zeros(len(s), dtype=int)
        taus = np.zeros(len(s))
        for t0 in range(0, len(s), self.L):
            seg = slice(t0, t0 + self.L)
            dec[seg] = (s[seg] >= tau).astype(int)
            taus[seg] = tau
            x = self._state(s[seg], None if labels_for_state is None else labels_for_state[seg], tau)
            a = int(np.argmax(self._pi(x)))
            tau = float(np.clip(tau + self.ACTIONS[a] * self.dtau, 0.0, 1.0))
        return dec, taus


# =============================================================================
# 13. Explainability
# =============================================================================


def _mask_groups(Xw: np.ndarray, coalition: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Replace sensor channels NOT in coalition by baseline (per-channel mean)."""
    Z = Xw.copy()
    off = ~coalition.astype(bool)
    Z[:, :, off] = baseline[off]
    return Z


def exact_group_shapley(score_fn: Callable[[np.ndarray], np.ndarray], Xw: np.ndarray,
                        baseline: np.ndarray, max_groups: int = 12) -> np.ndarray:
    """Exact Shapley values over sensor channels (2^m coalitions). Returns [n, m]."""
    from itertools import combinations

    n, W, m = Xw.shape
    if m > max_groups:
        raise ValueError(f"{m} groups: use kernel_group_shapley or group sensors by type")
    fact = [math.factorial(i) for i in range(m + 1)]
    cache: Dict[Tuple[int, ...], np.ndarray] = {}

    def v(S):
        key = tuple(sorted(S))
        if key not in cache:
            c = np.zeros(m); c[list(key)] = 1
            cache[key] = score_fn(_mask_groups(Xw, c, baseline))
        return cache[key]

    phi = np.zeros((n, m))
    for i in range(m):
        others = [j for j in range(m) if j != i]
        for r in range(m):
            wgt = fact[r] * fact[m - r - 1] / fact[m]
            for S in combinations(others, r):
                phi[:, i] += wgt * (v(S + (i,)) - v(S))
    return phi


def kernel_group_shapley(score_fn, Xw, baseline, n_samples: int = 512, seed: int = 0) -> np.ndarray:
    """KernelSHAP over sensor groups for large m (weighted least squares)."""
    rng = np.random.default_rng(seed)
    n, W, m = Xw.shape
    Z = rng.integers(0, 2, size=(n_samples, m)); Z[0] = 0; Z[1] = 1
    k = Z.sum(1)
    with np.errstate(divide="ignore"):
        wts = (m - 1) / (np.array([math.comb(m, int(kk)) if 0 < kk < m else 1 for kk in k]) * k * (m - k))
    wts[(k == 0) | (k == m)] = 1e6
    F = np.stack([score_fn(_mask_groups(Xw, z, baseline)) for z in Z])   # [S, n]
    A = np.hstack([np.ones((n_samples, 1)), Z])
    Wd = np.diag(wts)
    coef = np.linalg.pinv(A.T @ Wd @ A) @ A.T @ Wd @ F                  # [(m+1), n]
    return coef[1:].T


def lime_group(score_fn, Xw, baseline, n_samples: int = 512, kernel_width: float = 0.25, seed: int = 0) -> np.ndarray:
    """LIME with sensor-group masks and a weighted ridge surrogate. Returns [n, m]."""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed)
    n, W, m = Xw.shape
    Z = rng.integers(0, 2, size=(n_samples, m)); Z[0] = 1
    dist = 1.0 - Z.mean(1)
    wts = np.exp(-(dist ** 2) / kernel_width ** 2)
    F = np.stack([score_fn(_mask_groups(Xw, z, baseline)) for z in Z])
    out = np.zeros((n, m))
    for i in range(n):
        out[i] = Ridge(alpha=1e-3).fit(Z, F[:, i], sample_weight=wts).coef_
    return out


def attention_to_sensor(attn: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    """Map temporal self-attention [n, W, W] to per-sensor importance [n, m].

    received(t) = mean over queries of attention paid to key t; sensor importance
    = sum_t received(t) * |x_t,s|. Document this mapping in Sec. 9.3.
    """
    received = attn.mean(axis=1)                       # [n, W]
    return np.einsum("nw,nws->ns", received, np.abs(Xw))


def attribution_agreement(a: np.ndarray, b: np.ndarray, k: int = 5) -> dict:
    from scipy.stats import kendalltau

    taus, jac = [], []
    for u, v in zip(np.abs(a), np.abs(b)):
        tv = kendalltau(u, v).statistic
        taus.append(0.0 if np.isnan(tv) else tv)
        tu, tw = set(np.argsort(-u)[:k]), set(np.argsort(-v)[:k])
        jac.append(len(tu & tw) / len(tu | tw))
    q = lambda x: (float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75)))
    return dict(kendall_tau=q(taus), topk_jaccard=q(jac))


def counterfactual_distance(decide_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                            target: np.ndarray, tol: float = 1e-3, max_iter: int = 30) -> float:
    """Minimal L2 move along the segment x -> target (e.g. normal mean) that flips the decision.

    Bisection on t in [0,1]; returns NaN if the decision never flips.
    Replaces the random-sign search in the defence notebook.
    """
    d0 = decide_fn(x[None])[0]
    if decide_fn(target[None])[0] == d0:
        return float("nan")
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if decide_fn((x + mid * (target - x))[None])[0] == d0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(np.linalg.norm((hi * (target - x)).ravel()))


# =============================================================================
# 14. Resampling fidelity (R2-M6)
# =============================================================================


def resampling_fidelity(inp_path: str | Path, cfg: RevisionConfig, seed: int = 0) -> Dict[str, float]:
    """Compare the closed loop at 5 min with the same loop at 60 min."""
    cfg5 = dataclasses.replace(cfg, timestep_s=300)
    cfg60 = dataclasses.replace(cfg, timestep_s=3600)
    r5 = ClosedLoopWDS(inp_path, cfg5).run(seed)
    r60 = ClosedLoopWDS(inp_path, cfg60).run(seed)
    days = cfg.duration_days
    idx = np.arange(0, len(r5.t_s), 12)[: len(r60.t_s)]
    out = dict(
        pump_switches_per_day_5min=float(np.abs(np.diff(r5.pump_status, axis=0)).sum() / days),
        pump_switches_per_day_60min=float(np.abs(np.diff(r60.pump_status, axis=0)).sum() / days),
        rmse_junction_pressure_m=float(np.sqrt(np.mean((r5.junction_pressure[idx] - r60.junction_pressure[: len(idx)]) ** 2))),
        rmse_tank_level_m=float(np.sqrt(np.mean((r5.tank_level[idx] - r60.tank_level[: len(idx)]) ** 2))) if r5.tank_level.size else float("nan"),
    )
    v5 = r5.junction_pressure < cfg.p_min_m
    v60 = np.repeat(r60.junction_pressure < cfg.p_min_m, 12, axis=0)[: len(v5)]
    out["violation_time_missed_frac"] = float((v5 & ~v60).sum() / max(v5.sum(), 1))
    # hourly aggregation of 5-min telemetry (what resampling to 1 h would feed the loop)
    Yh = r5.y_true[: (len(r5.y_true) // 12) * 12].reshape(-1, 12, r5.y_true.shape[1])
    out["within_hour_std_over_signal_std"] = float(np.median(Yh.std(1).mean(0) / (r5.y_true.std(0) + 1e-9)))
    return out


# =============================================================================
# 15. Artefacts
# =============================================================================


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_run(res: ClosedLoopResult, path: str | Path, extra: Optional[dict] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t_s=res.t_s, y_true=res.y_true, y_meas=res.y_meas, y_scada=res.y_scada,
                        label=res.label, pump_status=res.pump_status, pump_power_kw=res.pump_power_kw,
                        junction_pressure=res.junction_pressure, junction_demand=res.junction_demand,
                        tank_level=res.tank_level, compromised=res.compromised,
                        sensor_names=np.array(res.layout.names))
    meta = dict(res.meta, **(extra or {}), file=path.name, sha256=sha256(path),
                rules=[dataclasses.asdict(r) for r in res.rules])
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def load_run(path: str | Path) -> Tuple[Dict[str, np.ndarray], dict]:
    """Strict loader: raises if the file or its manifest is missing or the hash differs."""
    path = Path(path)
    meta_p = path.with_suffix(".json")
    if not path.exists() or not meta_p.exists():
        raise FileNotFoundError(f"missing artefact {path} (+ .json). Run the attack notebook first; "
                                "synthetic fallback is disabled.")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if meta.get("sha256") and meta["sha256"] != sha256(path):
        raise ValueError(f"hash mismatch for {path}")
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    return data, meta
