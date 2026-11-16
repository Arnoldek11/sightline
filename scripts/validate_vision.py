"""Phase 2 validation of the perception model on the two pose games (PLAN 4.6, checks 1-3).

Writes <data>/processed/<id>/vision.parquet (P(seen) per decision x teammate, full grid) and
reports/vision_validation.json. Usage: uv run python scripts/validate_vision.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from sightline.paths import reports_dir
from sightline.pipeline import POSE_GAMES, load_config, load_processed, processed_dir
from sightline.vision import VisionConfig, build_geometry, p_seen_table, scans_before

CONES, MEMS = (45, 60, 90), (1.0, 1.5, 2.5)
GRID = {f"c{c}_m{m}": {"cone_deg": c, "memory_s": m} for c in CONES for m in MEMS}
GRID |= {"placebo": {"placebo": True}, "minlag0.3": {"min_lag_s": 0.3},
         "minlag0.3_placebo": {"min_lag_s": 0.3, "placebo": True}}


def auc_within(df: pd.DataFrame, col: str) -> float:
    """P(receiver's P(seen) > a random other teammate's), ties count 1/2, averaged per pass."""
    vals = []
    for _, g in df.groupby("event_id"):
        r = g.loc[g.is_target, col]
        o = g.loc[~g.is_target, col].to_numpy()
        if len(r) != 1 or len(o) == 0:
            continue
        r = r.iloc[0]
        vals.append(np.mean((r > o) + 0.5 * (r == o)))
    return float(np.mean(vals)) if vals else float("nan")


def summarise(df: pd.DataFrame, col: str) -> dict:
    rec = df[df.is_target]
    oth = df[~df.is_target]
    return {"n_passes": int(rec.event_id.nunique()),
            "p_seen_receiver": round(float(rec[col].mean()), 3),
            "p_seen_other": round(float(oth[col].mean()), 3),
            "auc": round(auc_within(df, col), 3)}


def run_game(mid: int, cfg_d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    dec = load_processed(mid, "decisions")
    players = load_processed(mid, "players")
    pose = load_processed(mid, "pose")
    tabs = {}
    for kind in ("head", "torso"):
        cfg = VisionConfig.from_dict(cfg_d, kind=kind)
        geoms = build_geometry(dec, players, pose, cfg)
        tabs[kind] = p_seen_table(geoms, cfg, GRID if kind == "head" else None)
    t = tabs["head"].merge(tabs["torso"][["event_id", "option_id", "p_seen"]]
                           .rename(columns={"p_seen": "p_seen_torso"}),
                           on=["event_id", "option_id"], how="left")
    meta = dec.set_index("event_id")
    t["is_target"] = t.option_id.to_numpy() == meta.target_id.reindex(t.event_id).to_numpy(
        dtype=float, na_value=np.nan)
    for c in ("end_type", "is_open_play", "counter", "attack_dir", "frame_end"):
        t[c] = meta[c].reindex(t.event_id).to_numpy()
    # option bearing relative to the attacking direction at release (0 = straight ahead)
    pos = players.set_index(["frame", "player_id"])[["x", "y"]]
    o = pos.reindex(pd.MultiIndex.from_arrays([t.frame_end, t.option_id])).to_numpy()
    c = pos.reindex(pd.MultiIndex.from_arrays([t.frame_end, t.carrier_id])).to_numpy()
    rel = (o - c) * t.attack_dir.to_numpy()[:, None]
    t["rel_bearing_deg"] = np.degrees(np.abs(np.arctan2(rel[:, 1], rel[:, 0])))
    t["dist_m"] = np.hypot(rel[:, 0], rel[:, 1])
    t.to_parquet(processed_dir(mid) / "vision.parquet")
    scans = scans_before(dec, load_processed(mid, "pose25"),
                         cfg_d["vision"]["scan_lookback_s"], turn_deg=cfg_d["vision"]
                         ["scan_turn_deg"], window_s=cfg_d["vision"]["scan_window_s"])
    scans = scans.merge(dec[["event_id", "player_id", "is_open_play"]], on="event_id")
    scans.to_parquet(processed_dir(mid) / "scans.parquet")
    return t, scans


def main() -> None:
    cfg_d = load_config("default")
    min_cov = VisionConfig().min_coverage
    out: dict = {"config": {"cone_deg": cfg_d["vision"]["cone_half_angle_deg"],
                            "memory_s": cfg_d["vision"]["memory_s"], "min_coverage": min_cov,
                            "n_draws": cfg_d["pose"]["mc_draws"]}, "games": {}}
    allv, alls = [], []
    for mid in POSE_GAMES:
        t0 = time.perf_counter()
        v, s = run_game(mid, cfg_d)
        print(f"[{mid}] {v.event_id.nunique()} decisions, {len(v):,} options "
              f"({time.perf_counter() - t0:.0f}s)")
        allv.append(v.assign(match_id=mid))
        alls.append(s.assign(match_id=mid))
    v = pd.concat(allv, ignore_index=True)
    s = pd.concat(alls, ignore_index=True)
    out["coverage"] = {"decisions": int(v.event_id.nunique()),
                       "share_coverage_ge_min": round(float(
                           v.groupby("event_id").coverage.first().ge(min_cov).mean()), 3)}
    passes = v[(v.end_type == "pass") & (v.coverage >= min_cov)
               & v.event_id.isin(v.loc[v.is_target, "event_id"])]
    res = {}
    for subset, sub in {"all_passes": passes, "open_play": passes[passes.is_open_play],
                        "counter": passes[passes.counter.astype(bool)]}.items():
        res[subset] = {"default": summarise(sub, "p_seen"),
                       "torso": summarise(sub, "p_seen_torso")}
        for name in GRID:
            res[subset][name] = summarise(sub, f"p_seen_{name}")
    for mid in POSE_GAMES:
        res[f"game_{mid}"] = {"default": summarise(passes[passes.match_id == mid], "p_seen")}
    # hard case: receivers behind/beside the carrier (> 90 deg from attacking direction)
    back_events = passes.loc[passes.is_target & (passes.rel_bearing_deg > 90), "event_id"]
    back = passes[passes.event_id.isin(back_events) & (passes.rel_bearing_deg > 90)]
    res["receiver_behind"] = {"default": summarise(back, "p_seen"),
                              "minlag0.3": summarise(back, "p_seen_minlag0.3")}
    out["receiver_sanity"] = res
    # P(seen) by option bearing (all decisions, default config)
    bins = pd.cut(v.rel_bearing_deg, [0, 45, 90, 135, 180], include_lowest=True)
    out["p_seen_by_bearing"] = {str(k): round(float(x), 3) for k, x in
                                v[v.coverage >= min_cov].groupby(bins, observed=True)
                                .p_seen.mean().items()}
    # construct validity proxy: more scanning -> sees more of what is behind
    behind = (v[(v.coverage >= min_cov) & (v.rel_bearing_deg > 90)]
              .groupby("event_id").p_seen.mean().rename("p_seen_behind"))
    sv = s[s.pose_share >= 0.5].merge(behind, left_on="event_id", right_index=True)
    out["scanning"] = {
        "decisions": len(sv),
        "mean_scans_5s": round(float(sv.n_scans.mean()), 2),
        "spearman_scans_vs_p_seen_behind_decision": round(float(
            sv[["n_scans", "p_seen_behind"]].corr("spearman").iloc[0, 1]), 3),
    }
    pl = sv.groupby("player_id").agg(n=("event_id", "size"), scans=("n_scans", "mean"),
                                     behind=("p_seen_behind", "mean"))
    pl = pl[pl.n >= 15]
    out["scanning"]["players_ge15"] = len(pl)
    out["scanning"]["spearman_player_level"] = round(float(
        pl[["scans", "behind"]].corr("spearman").iloc[0, 1]), 3)
    path = reports_dir() / "vision_validation.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("coverage", "p_seen_by_bearing", "scanning")},
                     indent=2))
    for k in ("all_passes", "open_play", "counter", "receiver_behind"):
        print(k, {n: res[k][n] for n in res[k] if n in ("default", "placebo", "minlag0.3",
                                                        "minlag0.3_placebo", "torso")})


if __name__ == "__main__":
    main()
