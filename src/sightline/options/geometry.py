"""Geometry and features for option valuation (PLAN 4.2). All in the *attacking frame*:
the team in possession attacks +x, goal centre at (L/2, 0); att = pitch * attack_dir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GOAL_HALF_WIDTH = 3.66
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16

# physics constants for time-to-intercept (typical values; logged in DECISIONS D-008)
BALL_SPEED = 15.0  # m/s, average ground pass speed
PLAYER_VMAX = 7.0  # m/s
REACTION_S = 0.3
REACH_M = 1.0  # a defender within 1 m of the ball's path can play it
CARRY_SPEED = 5.0  # m/s with the ball


@dataclass
class Snapshot:
    """Players at one frame, in the attacking frame of `team_id`."""

    mate_ids: np.ndarray
    mates: np.ndarray  # (M, 2)
    opp_ids: np.ndarray
    opps: np.ndarray  # (K, 2)
    opps_v: np.ndarray  # (K, 2)
    opp_outfield: np.ndarray  # (K,) bool
    pitch_length: float
    pitch_width: float

    def flipped(self) -> Snapshot:
        """Same frame seen by the opponent (their attacking frame)."""
        return Snapshot(self.opp_ids, -self.opps, self.mate_ids, -self.mates,
                        np.zeros_like(self.mates), np.ones(len(self.mates), bool),
                        self.pitch_length, self.pitch_width)


def snapshot(frame_players: pd.DataFrame, team_id: int, attack_dir: int, gk_ids: set,
             pitch_length: float, pitch_width: float) -> Snapshot:
    fp = frame_players
    m = fp.team_id.to_numpy() == team_id
    xy = fp[["x", "y"]].to_numpy(float) * attack_dir
    v = fp[["vx", "vy"]].to_numpy(float) * attack_dir
    ids = fp.player_id.to_numpy()
    return Snapshot(ids[m], xy[m], ids[~m], xy[~m], np.nan_to_num(v[~m]),
                    ~np.isin(ids[~m], list(gk_ids)), pitch_length, pitch_width)


STATE_FEATURES = ["x", "abs_y", "dist_goal", "goal_angle", "in_box", "n_def_goal_side",
                  "nearest_def", "n_def_5m", "n_mates_ahead", "n_block"]


def _n_in_shot_triangle(p: np.ndarray, opps: np.ndarray, gx: float) -> np.ndarray:
    """Opponents inside the triangle ball - left post - right post, per point (P,)."""
    if len(opps) == 0:
        return np.zeros(len(p), int)
    a = p[:, None, :]
    b = np.array([gx, GOAL_HALF_WIDTH])
    c = np.array([gx, -GOAL_HALF_WIDTH])
    q = opps[None, :, :]

    def cross(o, u, v):
        return (u[..., 0] - o[..., 0]) * (v[..., 1] - o[..., 1]) - \
            (u[..., 1] - o[..., 1]) * (v[..., 0] - o[..., 0])

    d1, d2, d3 = cross(a, b, q), cross(b + 0 * a, c + 0 * a, q), cross(c + 0 * a, a, q)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return (~(neg & pos)).sum(1)


def state_features(points: np.ndarray, snap: Snapshot) -> pd.DataFrame:
    """Features of a ball location (P, 2) for the team in possession, given the snapshot."""
    p = np.atleast_2d(points)
    L = snap.pitch_length
    gx = L / 2
    dx, dy = gx - p[:, 0], p[:, 1]
    a1 = np.arctan2(GOAL_HALF_WIDTH - dy, dx)
    a2 = np.arctan2(-GOAL_HALF_WIDTH - dy, dx)
    d_opp = np.hypot(p[:, None, 0] - snap.opps[None, :, 0], p[:, None, 1] - snap.opps[None, :, 1])
    outf = snap.opp_outfield[None, :]
    return pd.DataFrame({
        "x": p[:, 0],
        "abs_y": np.abs(p[:, 1]),
        "dist_goal": np.hypot(dx, dy),
        "goal_angle": np.abs(a1 - a2),
        "in_box": ((p[:, 0] >= gx - BOX_DEPTH) & (np.abs(p[:, 1]) <= BOX_HALF_WIDTH)).astype(int),
        "n_def_goal_side": ((snap.opps[None, :, 0] > p[:, None, 0]) & outf).sum(1),
        "nearest_def": d_opp.min(1) if d_opp.shape[1] else np.full(len(p), 50.0),
        "n_def_5m": (d_opp < 5).sum(1),
        "n_mates_ahead": (snap.mates[None, :, 0] > p[:, None, 0]).sum(1),
        "n_block": _n_in_shot_triangle(p, snap.opps[snap.opp_outfield], gx),
    })


def _arrival_times(points: np.ndarray, snap: Snapshot) -> np.ndarray:
    """Earliest time (s) any opponent reaches each point, shape points.shape[:-1]."""
    start = snap.opps + snap.opps_v * REACTION_S  # keep running during the reaction time
    d = np.linalg.norm(points[..., None, :] - start, axis=-1)
    t = REACTION_S + np.maximum(d - REACH_M, 0) / PLAYER_VMAX
    return t.min(-1) if t.shape[-1] else np.full(points.shape[:-1], 99.0)


PASS_FEATURES = ["min_margin", "target_margin", "length", "forward", "n_lane", "target_def"]


def pass_features(carrier: np.ndarray, targets: np.ndarray, snap: Snapshot,
                  n_samples: int = 12) -> pd.DataFrame:
    """Physics features of passes from `carrier` (2,) to each target (T, 2).

    margin = earliest defender arrival - ball arrival at points along the line (s);
    negative = a defender gets there first.
    """
    t = np.atleast_2d(targets)
    s = np.linspace(0, 1, n_samples + 1)[1:]
    pts = carrier + s[None, :, None] * (t - carrier)[:, None, :]  # (T, S, 2)
    t_ball = np.linalg.norm(pts - carrier, axis=-1) / BALL_SPEED
    margin = _arrival_times(pts, snap) - t_ball
    seg = t - carrier
    ln = np.linalg.norm(seg, axis=1)
    if len(snap.opps):
        rel = snap.opps[None] - carrier  # (1, K, 2)
        proj = np.clip((rel * seg[:, None]).sum(-1) / np.maximum(ln, 1e-6)[:, None] ** 2, 0, 1)
        closest = carrier + proj[..., None] * seg[:, None]
        lane_d = np.linalg.norm(snap.opps[None] - closest, axis=-1)
        n_lane = (lane_d < 3).sum(1)
        target_def = np.linalg.norm(t[:, None] - snap.opps[None], axis=-1).min(1)
    else:
        n_lane = np.zeros(len(t))
        target_def = np.full(len(t), 50.0)
    return pd.DataFrame({"min_margin": np.clip(margin.min(1), -3, 3),
                         "target_margin": np.clip(margin[:, -1], -3, 3),
                         "length": ln, "forward": seg[:, 0], "n_lane": n_lane,
                         "target_def": np.minimum(target_def, 30)})


CARRY_FEATURES = ["min_margin", "end_margin", "length", "forward", "end_def", "start_def"]


def carry_features(carrier: np.ndarray, ends: np.ndarray, snap: Snapshot,
                   n_samples: int = 6) -> pd.DataFrame:
    e = np.atleast_2d(ends)
    s = np.linspace(0, 1, n_samples + 1)[1:]
    pts = carrier + s[None, :, None] * (e - carrier)[:, None, :]
    t_car = 0.15 + np.linalg.norm(pts - carrier, axis=-1) / CARRY_SPEED
    margin = _arrival_times(pts, snap) - t_car
    seg = e - carrier
    dd = (np.linalg.norm(e[:, None] - snap.opps[None], axis=-1).min(1) if len(snap.opps)
          else np.full(len(e), 50.0))
    sd = (np.linalg.norm(carrier - snap.opps, axis=1).min() if len(snap.opps) else 50.0)
    return pd.DataFrame({"min_margin": np.clip(margin.min(1), -3, 3),
                         "end_margin": np.clip(margin[:, -1], -3, 3),
                         "length": np.linalg.norm(seg, axis=1), "forward": seg[:, 0],
                         "end_def": np.minimum(dd, 30), "start_def": np.full(len(e), min(sd, 30))})


SHOT_FEATURES = ["dist_goal", "goal_angle", "n_block", "nearest_def"]


def carry_endpoints(carrier: np.ndarray, distances=(5.0, 10.0), n_dirs: int = 8,
                    pitch_length: float = 105.0, pitch_width: float = 68.0) -> np.ndarray:
    ang = np.arange(n_dirs) * 2 * np.pi / n_dirs
    pts = np.array([carrier + d * np.array([np.cos(a), np.sin(a)])
                    for d in distances for a in ang])
    pts[:, 0] = np.clip(pts[:, 0], -pitch_length / 2, pitch_length / 2)
    pts[:, 1] = np.clip(pts[:, 1], -pitch_width / 2, pitch_width / 2)
    return pts
