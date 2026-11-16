"""What could the ball carrier see? Field-of-view cones, visual memory, P(seen) (PLAN 4.3).

For a decision released at tracking frame t0, an option o counts as **seen** in one Monte Carlo
draw of the carrier's facing if, at some 10 fps sample t in [t0 - memory, t0 - min_lag]:
  * o is within +/- cone of the facing direction at t, and
  * o has moved <= max_move_m between t and t0 (otherwise the picture he has is stale).
P(seen) = share of draws in which o counts as seen. Draws come from joint-level noise
(sightline.pose.sample_yaw). Samples without usable pose are "not looking": a carrier with
no pose in the window sees nothing, so decisions with low window coverage are flagged.

Options are generic points with a trajectory (teammates now; carry targets / shot later).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sightline import pose as pose_mod
from sightline.io.tracking import TRACKING_FPS


@dataclass(frozen=True)
class VisionConfig:
    cone_deg: float = 60.0
    memory_s: float = 1.5
    max_move_m: float = 3.0
    min_lag_s: float = 0.0
    kind: str = "head"  # "head" or "torso"
    n_draws: int = 200
    min_coverage: float = 0.5  # share of window samples with usable pose
    seed: int = 20261116

    @classmethod
    def from_dict(cls, d: dict, **over) -> VisionConfig:
        v, p = d.get("vision", {}), d.get("pose", {})
        kw = dict(cone_deg=v.get("cone_half_angle_deg", 60.0), memory_s=v.get("memory_s", 1.5),
                  max_move_m=v.get("max_displacement_m", 3.0), n_draws=p.get("mc_draws", 200),
                  seed=d.get("seed", 20261116))
        return cls(**(kw | over))


@dataclass
class DecisionGeometry:
    """Everything needed to evaluate P(seen) for one decision, at 10 fps lags 0..L-1."""

    event_id: str
    carrier_id: int
    option_ids: np.ndarray  # (M,)
    carrier_xy: np.ndarray  # (L, 2), NaN where missing
    option_xy: np.ndarray  # (L, M, 2)
    yaw_draws: np.ndarray  # (L, D), NaN where no usable pose
    yaw_point: np.ndarray = field(default_factory=lambda: np.empty(0))  # (L,)

    @property
    def n_lags(self) -> int:
        return len(self.carrier_xy)


def p_seen(g: DecisionGeometry, cone_deg: float, memory_s: float, max_move_m: float = 3.0,
           min_lag_s: float = 0.0, placebo: bool = False) -> tuple[np.ndarray, float]:
    """(P(seen) per option (M,), window pose coverage)."""
    seen, coverage = seen_draws(g, cone_deg, memory_s, max_move_m, min_lag_s, placebo)
    return seen.mean(axis=1), coverage


def seen_draws(g: DecisionGeometry, cone_deg: float, memory_s: float, max_move_m: float = 3.0,
               min_lag_s: float = 0.0, placebo: bool = False) -> tuple[np.ndarray, float]:
    """(seen indicator per option and Monte Carlo draw, shape (M, D); window pose coverage)."""
    lo = int(round(min_lag_s * TRACKING_FPS))
    hi = min(int(round(memory_s * TRACKING_FPS)), g.n_lags - 1)
    lags = np.arange(lo, hi + 1)
    yaw = g.yaw_draws[lags]  # (l, D)
    cxy = g.carrier_xy[lags]  # (l, 2)
    oxy = g.option_xy[lags]  # (l, M, 2)
    bearing = np.arctan2(oxy[..., 1] - cxy[:, None, 1], oxy[..., 0] - cxy[:, None, 0])  # (l, M)
    off = np.abs(pose_mod.wrap(bearing[:, :, None] - yaw[:, None, :]))  # (l, M, D)
    if placebo:
        off = np.pi - off
    moved = np.hypot(*(g.option_xy[0][None] - oxy).transpose(2, 0, 1))  # (l, M)
    in_view = (off <= np.radians(cone_deg)) & (moved <= max_move_m)[:, :, None]
    in_view &= ~np.isnan(yaw)[:, None, :]
    coverage = float(np.mean(~np.isnan(yaw[:, 0]))) if len(lags) else 0.0
    return in_view.any(axis=0), coverage


def with_static_options(g: DecisionGeometry, points: np.ndarray,
                        ids: list[str]) -> DecisionGeometry:
    """Append static options (e.g. carry end points, pitch frame (P, 2)) to a geometry."""
    L = g.n_lags
    extra = np.repeat(np.asarray(points, float)[None], L, axis=0)
    return DecisionGeometry(g.event_id, g.carrier_id,
                            np.r_[g.option_ids.astype(object), np.array(ids, dtype=object)],
                            g.carrier_xy, np.concatenate([g.option_xy, extra], axis=1),
                            g.yaw_draws, g.yaw_point)


def build_geometry(decisions: pd.DataFrame, players: pd.DataFrame, pose: pd.DataFrame,
                   cfg: VisionConfig, max_memory_s: float = 2.5) -> list[DecisionGeometry]:
    """Geometry for each decision (teammates as options). `pose` = 10 fps aligned table."""
    L = int(round(max_memory_s * TRACKING_FPS)) + 1
    team = players.drop_duplicates("player_id").set_index("player_id").team_id
    pos = players.set_index(["frame", "player_id"])[["x", "y"]]
    usable = pose[pose.sync_ok].copy()
    usable = usable[usable[f"{cfg.kind}_yaw"].notna()]

    # one batched Monte Carlo over every (frame, carrier) we need
    need = {(int(t0) - lag, int(c)) for t0, c in zip(decisions.frame_end, decisions.player_id,
                                                     strict=True) for lag in range(L)}
    key = pd.MultiIndex.from_frame(usable[["frame", "player_id"]])
    sel = usable[key.isin(list(need))].reset_index(drop=True)
    rng = np.random.default_rng(cfg.seed)
    n_joints = len(pose_mod.HEAD_JOINTS if cfg.kind == "head" else pose_mod.TORSO_JOINTS)
    row_of = {(int(f), int(p)): i for i, (f, p) in enumerate(zip(sel.frame, sel.player_id,
                                                                 strict=True))}
    point = sel[f"{cfg.kind}_yaw"].to_numpy()

    out = []
    for d in decisions.itertuples():
        t0, cid = int(d.frame_end), int(d.player_id)
        frames = t0 - np.arange(L)
        at0 = players[players.frame == t0]
        mates = at0.player_id[(at0.player_id.map(team) == team.get(cid))
                              & (at0.player_id != cid)].to_numpy()
        if len(mates) == 0:
            continue
        cxy = pos.reindex(pd.MultiIndex.from_arrays([frames, np.full(L, cid)])).to_numpy()
        idx = pd.MultiIndex.from_product([frames, mates])
        oxy = pos.reindex(idx).to_numpy().reshape(L, len(mates), 2)
        yd = np.full((L, cfg.n_draws), np.nan)
        yp = np.full(L, np.nan)
        rows = [(i, row_of.get((int(f), cid))) for i, f in enumerate(frames)]
        rows = [(i, r) for i, r in rows if r is not None]
        # persistent error over the window: one noise draw per MC sample, shared by all lags
        noise = rng.standard_normal((cfg.n_draws, n_joints, 2))
        if rows:
            li, ri = map(list, zip(*rows, strict=True))
            yd[li] = pose_mod.sample_yaw(sel.iloc[ri], cfg.kind, noise=noise)
            yp[li] = point[ri]
        out.append(DecisionGeometry(d.event_id, cid, mates, cxy, oxy, yd, yp))
    return out


def p_seen_table(geoms: list[DecisionGeometry], cfg: VisionConfig,
                 grid: dict[str, dict] | None = None) -> pd.DataFrame:
    """Long table: one row per (decision, option) with P(seen) columns.

    `grid` maps a column suffix to keyword overrides of p_seen, e.g.
    {"c45_m1.0": {"cone_deg": 45, "memory_s": 1.0}, "placebo": {"placebo": True}}.
    The default config always produces `p_seen` and `coverage`.
    """
    base = dict(cone_deg=cfg.cone_deg, memory_s=cfg.memory_s, max_move_m=cfg.max_move_m,
                min_lag_s=cfg.min_lag_s)
    rows = []
    for g in geoms:
        p, cov = p_seen(g, **base)
        df = pd.DataFrame({"event_id": g.event_id, "carrier_id": g.carrier_id,
                           "option_id": g.option_ids, "p_seen": p, "coverage": cov})
        for name, kw in (grid or {}).items():
            df[f"p_seen_{name}"] = p_seen(g, **(base | kw))[0]
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------------------------
# Scanning (independent check, PLAN 4.3): head turns > turn_deg within window_s


def count_scans(yaw: np.ndarray, fps: int = 25, turn_deg: float = 40.0,
                window_s: float = 0.5, smooth: int = 5) -> int:
    """Number of distinct head turns of > turn_deg within window_s in a yaw series (radians).

    NaNs split the series into segments; each segment is unwrapped and median-smoothed
    (`smooth` frames) before differencing over the window. Consecutive frames exceeding the
    threshold form one scan.
    """
    yaw = np.asarray(yaw, float)
    w = int(round(window_s * fps))
    n = 0
    valid = ~np.isnan(yaw)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(int)))
    for s, e in zip(edges[::2], edges[1::2], strict=True):
        seg = np.unwrap(yaw[s:e])
        if len(seg) <= w:
            continue
        if smooth > 1 and len(seg) >= smooth:
            seg = pd.Series(seg).rolling(smooth, center=True, min_periods=1).median().to_numpy()
        big = np.abs(seg[w:] - seg[:-w]) > np.radians(turn_deg)
        n += int(np.sum(np.diff(np.r_[False, big].astype(int)) == 1))
    return n


def scans_before(decisions: pd.DataFrame, pose25: pd.DataFrame, lookback_s: float = 5.0,
                 kind: str = "head", **kw) -> pd.DataFrame:
    """Scans by the carrier in the `lookback_s` before receiving (frame_start), 25 fps pose.

    Returns event_id, n_scans, pose_share (share of lookback frames with usable yaw).
    """
    by_player = {p: g.set_index("pose_frame")[f"{kind}_yaw"]
                 for p, g in pose25.groupby("player_id")}
    n = int(round(lookback_s * 25))
    rows = []
    for d in decisions.itertuples():
        f1 = int(np.floor(2.5 * d.frame_start))
        s = by_player.get(d.player_id)
        y = (s.reindex(np.arange(f1 - n, f1 + 1)).to_numpy() if s is not None
             else np.full(n + 1, np.nan))
        rows.append({"event_id": d.event_id, "n_scans": count_scans(y, **kw),
                     "pose_share": float(np.mean(~np.isnan(y)))})
    return pd.DataFrame(rows)
