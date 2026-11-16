"""SkillCorner match sheet + 10 fps tracking -> canonical long tables (pandas).

Canonical player-frame table columns:
    frame, period, player_id, team_id, x, y, is_detected,
    attack_dir (+1 if the player's team attacks +x in this period, else -1),
    vx, vy, speed, ax, ay  (smoothed, m/s and m/s^2; see add_kinematics)
Ball table: frame, period, ball_x, ball_y, ball_z, ball_detected, poss_player_id, poss_group.
Coordinates: metres, origin at pitch centre (SkillCorner native).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
from scipy.signal import savgol_filter

from sightline.paths import data_dir

TRACKING_FPS = 10


def match_dir(match_id: int, root: Path | None = None) -> Path:
    root = root or data_dir() / "opendata" / "data" / "matches"
    return Path(root) / str(match_id)


@dataclass
class MatchMeta:
    match_id: int
    home_team_id: int
    away_team_id: int
    pitch_length: float
    pitch_width: float
    periods: pd.DataFrame  # period, start_frame, end_frame
    players: pd.DataFrame  # player_id, team_id, number, short_name, role, position_group
    home_side: list[str]  # per period: "left_to_right" / "right_to_left"

    def attack_dir(self, team_id: int, period: int) -> int:
        """+1 if `team_id` attacks towards +x in `period`, else -1."""
        home_dir = 1 if self.home_side[period - 1] == "left_to_right" else -1
        return home_dir if team_id == self.home_team_id else -home_dir


def load_meta(match_id: int, root: Path | None = None) -> MatchMeta:
    m = orjson.loads((match_dir(match_id, root) / f"{match_id}_match.json").read_bytes())
    players = pd.DataFrame(
        {
            "player_id": p["id"],
            "team_id": p["team_id"],
            "number": p.get("number"),
            "short_name": p.get("short_name"),
            "role": (p.get("player_role") or {}).get("acronym"),
            "position_group": (p.get("player_role") or {}).get("position_group"),
        }
        for p in m["players"]
    )
    periods = pd.DataFrame(m["match_periods"])[["period", "start_frame", "end_frame"]]
    return MatchMeta(
        match_id=match_id,
        home_team_id=m["home_team"]["id"],
        away_team_id=m["away_team"]["id"],
        pitch_length=float(m["pitch_length"]),
        pitch_width=float(m["pitch_width"]),
        periods=periods,
        players=players,
        home_side=list(m["home_team_side"]),
    )


def read_tracking_jsonl(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the raw JSONL into (players, ball) tables. Frames without players are skipped."""
    pf, pp, px, py, pdet = [], [], [], [], []
    ball = []
    with open(path, "rb") as fh:
        for line in fh:
            r = orjson.loads(line)
            if not r["player_data"] or r["period"] is None:
                continue
            f = r["frame"]
            for p in r["player_data"]:
                pf.append(f)
                pp.append(p["player_id"])
                px.append(p["x"])
                py.append(p["y"])
                pdet.append(bool(p["is_detected"]))
            b, poss = r["ball_data"], r["possession"]
            ball.append((f, r["period"], b["x"], b["y"], b["z"], b["is_detected"],
                         poss["player_id"], poss["group"]))
    ball_df = pd.DataFrame(ball, columns=["frame", "period", "ball_x", "ball_y", "ball_z",
                                          "ball_detected", "poss_player_id", "poss_group"])
    players = pd.DataFrame({"frame": pf, "player_id": pp, "x": px, "y": py, "is_detected": pdet})
    players = players.merge(ball_df[["frame", "period"]], on="frame", how="left")
    return players, ball_df


def add_kinematics(df: pd.DataFrame, window: int = 7, poly: int = 2,
                   fps: int = TRACKING_FPS) -> pd.DataFrame:
    """Savitzky-Golay smoothed velocity/acceleration per player over contiguous frame runs.

    Tracking identity is ~97% accurate, so raw differences are noisy; a 0.7 s quadratic
    window is a standard compromise. Runs shorter than `window` get finite differences.
    """
    df = df.sort_values(["player_id", "frame"]).reset_index(drop=True)
    out = np.full((len(df), 4), np.nan)
    new_run = ((df.player_id != df.player_id.shift()) | (df.frame.diff() != 1)).to_numpy()
    starts = np.flatnonzero(new_run)
    ends = np.r_[starts[1:], len(df)]
    xy = df[["x", "y"]].to_numpy(float)
    dt = 1.0 / fps
    for s, e in zip(starts, ends, strict=True):
        seg = xy[s:e]
        n = len(seg)
        if n >= window:
            v = savgol_filter(seg, window, poly, deriv=1, delta=dt, axis=0)
            a = savgol_filter(seg, window, poly, deriv=2, delta=dt, axis=0)
        elif n >= 2:
            v = np.gradient(seg, dt, axis=0)
            a = np.gradient(v, dt, axis=0) if n >= 3 else np.zeros_like(seg)
        else:
            v = a = np.zeros_like(seg)
        out[s:e, :2], out[s:e, 2:] = v, a
    df["vx"], df["vy"], df["ax"], df["ay"] = out.T
    df["speed"] = np.hypot(df.vx, df.vy)
    return df


def add_attack_dir(players: pd.DataFrame, meta: MatchMeta) -> pd.DataFrame:
    dirs = {(t, p): meta.attack_dir(t, p)
            for t in (meta.home_team_id, meta.away_team_id) for p in meta.periods.period}
    players["attack_dir"] = [dirs.get((t, p), 0) for t, p in
                             zip(players.team_id, players.period, strict=True)]
    return players


def load_tracking(match_id: int, root: Path | None = None,
                  meta: MatchMeta | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonical (players, ball) tables for one match."""
    meta = meta or load_meta(match_id, root)
    players, ball = read_tracking_jsonl(
        match_dir(match_id, root) / f"{match_id}_tracking_extrapolated.jsonl")
    players = players.merge(meta.players[["player_id", "team_id"]], on="player_id", how="left")
    players = add_kinematics(add_attack_dir(players, meta))
    return players.sort_values(["frame", "player_id"]).reset_index(drop=True), ball
