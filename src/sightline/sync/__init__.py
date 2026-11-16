"""Pose (25 fps) <-> tracking (10 fps) alignment.

SkillCorner: pose_frame = 2.5 * tracking_frame. Each tracking frame t is matched to the
pose frame nearest 2.5 t (exact for even t; for odd t, 2.5 t is a half-integer and we take
the frame below, so the match is at most 0.02 s off). Player ids are shared between files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tracking_to_pose_frame(t):
    """Pose frame matched to tracking frame(s) t."""
    return np.floor(2.5 * np.asarray(t)).astype(np.int64)


def pose_to_tracking_frame(f):
    """Tracking frame whose matched pose frame is f, or -1 if f is not a matched frame."""
    f = np.asarray(f)
    t = np.ceil(f / 2.5).astype(np.int64)
    return np.where(tracking_to_pose_frame(t) == f, t, -1)


def align_to_tracking(pose: pd.DataFrame, players: pd.DataFrame,
                      max_offset_m: float = 1.0) -> pd.DataFrame:
    """Keep pose rows on the 10 fps grid, join tracking XY, flag sync quality.

    Adds `frame` (tracking frame), `sync_offset_m` (midHip vs tracking XY) and `sync_ok`.
    """
    t = pose_to_tracking_frame(pose.pose_frame.to_numpy())
    out = pose.loc[t >= 0].copy()
    out["frame"] = t[t >= 0]
    out = out.merge(players[["frame", "player_id", "x", "y"]], on=["frame", "player_id"],
                    how="inner")
    out["sync_offset_m"] = np.hypot(out.midHip_x - out.x, out.midHip_y - out.y)
    out["sync_ok"] = out.sync_offset_m <= max_offset_m
    return out.drop(columns=["x", "y"])
