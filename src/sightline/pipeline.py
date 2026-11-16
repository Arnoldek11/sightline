"""Build the canonical per-game tables (Phase 1 data layer).

Writes to <data_dir>/processed/<match_id>/:
    players.parquet    10 fps player-frames with kinematics + attack_dir
    ball.parquet       10 fps ball + possession
    decisions.parquet  on-ball decision points with counter tags
    counters.parquet   one row per regain with counter flags
    pose.parquet       (pose games) 10 fps pose rows aligned to tracking + head/torso yaw
    pose25.parquet     (pose games) full 25 fps head/torso yaw, for scanning measures
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yaml

from sightline import pose as pose_mod
from sightline.counters import CounterConfig, detect_counters, tag_decisions
from sightline.io.events import decision_points, load_events
from sightline.io.pose import load_pose, pose_zip_path
from sightline.io.tracking import load_meta, load_tracking
from sightline.paths import REPO_ROOT, data_dir
from sightline.sync import align_to_tracking

POSE_GAMES = (1925299, 1996435)


def load_config(name: str = "default") -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / f"{name}.yaml").read_text())


def processed_dir(match_id: int) -> Path:
    p = data_dir() / "processed" / str(match_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_game(match_id: int, with_pose: bool | None = None, log=print) -> dict:
    cfg, ccfg = load_config("default"), load_config("counters")
    out = processed_dir(match_id)
    t = time.perf_counter()
    meta = load_meta(match_id)
    players, ball = load_tracking(match_id, meta=meta)
    log(f"[{match_id}] tracking {len(players):,} player-frames ({time.perf_counter()-t:.0f}s)")
    dec = decision_points(load_events(match_id))
    counters = detect_counters(dec, ball, players, meta, CounterConfig.from_dict(ccfg))
    dec = tag_decisions(dec, counters, ccfg.get("fast_attack_window_s", 15.0))
    players.to_parquet(out / "players.parquet")
    ball.to_parquet(out / "ball.parquet")
    dec.to_parquet(out / "decisions.parquet")
    counters.to_parquet(out / "counters.parquet")
    summary = {"match_id": match_id, "player_frames": len(players), "decisions": len(dec),
               "regains": len(counters),
               "rule_counters": int(counters.rule_counter.sum()) if len(counters) else 0,
               "label_counters": int(counters.label_counter.sum()) if len(counters) else 0,
               "fast_attacks": int(counters.fast_attack.sum()) if len(counters) else 0}
    if with_pose is None:
        with_pose = match_id in POSE_GAMES and pose_zip_path(match_id).exists()
    if with_pose:
        t = time.perf_counter()
        pose = load_pose(match_id)
        max_err = cfg["pose"]["max_joint_err_cm"]
        pose = pd.concat([pose, pose_mod.facing(pose, max_err)], axis=1)
        pose[["pose_frame", "period", "player_id", "head_yaw", "torso_yaw"]].to_parquet(
            out / "pose25.parquet")
        aligned = align_to_tracking(pose, players, cfg["sync"]["max_pose_xy_offset_m"])
        aligned.to_parquet(out / "pose.parquet")
        summary |= {"pose_rows_25fps": len(pose), "pose_rows_10fps": len(aligned),
                    "sync_median_m": float(aligned.sync_offset_m.median()),
                    "sync_ok_share": float(aligned.sync_ok.mean())}
        log(f"[{match_id}] pose {len(pose):,} rows ({time.perf_counter()-t:.0f}s)")
    return summary


def load_processed(match_id: int, name: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir() / "processed" / str(match_id) / f"{name}.parquet")
