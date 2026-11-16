"""Training datasets for the option models, built from the processed per-game tables.

* shots:  one row per shot (carrier location at release) -> goal (xG)
* passes: one row per targeted pass, point-to-player at release -> completed
* carries: one row per possession flagged `carry`, start -> end location -> ball kept
* states: ball location every 0.5 s of live possession -> xG of own shots within 10 s,
          before the opponent regains (the value target V, DECISIONS D-008)
Everything is in the attacking frame of the team in possession.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sightline.io.tracking import load_meta
from sightline.options.geometry import (
    Snapshot,
    carry_features,
    pass_features,
    state_features,
)
from sightline.pipeline import load_processed


@dataclass
class GameData:
    match_id: int
    meta: object
    decisions: pd.DataFrame
    ball: pd.DataFrame
    frames: np.ndarray
    starts: np.ndarray
    pid: np.ndarray
    team: np.ndarray
    xy: np.ndarray
    v: np.ndarray
    gk: set
    pos: dict

    @classmethod
    def load(cls, match_id: int) -> GameData:
        meta = load_meta(match_id)
        pl = load_processed(match_id, "players").sort_values(["frame", "player_id"])
        fr = pl.frame.to_numpy()
        uniq, starts = np.unique(fr, return_index=True)
        xy = pl[["x", "y"]].to_numpy(float)
        return cls(match_id, meta, load_processed(match_id, "decisions"),
                   load_processed(match_id, "ball"), uniq, np.r_[starts, len(fr)],
                   pl.player_id.to_numpy(), pl.team_id.to_numpy(), xy,
                   np.nan_to_num(pl[["vx", "vy"]].to_numpy(float)),
                   set(meta.players.loc[meta.players.role == "GK", "player_id"]),
                   {(f, p): i for i, (f, p) in enumerate(zip(fr, pl.player_id, strict=True))})

    def _slice(self, frame: int) -> slice | None:
        i = np.searchsorted(self.frames, frame)
        if i >= len(self.frames) or self.frames[i] != frame:
            return None
        return slice(self.starts[i], self.starts[i + 1])

    def snap(self, frame: int, team_id: int, attack_dir: int) -> Snapshot | None:
        s = self._slice(frame)
        if s is None:
            return None
        m = self.team[s] == team_id
        xy = self.xy[s] * attack_dir
        v = self.v[s] * attack_dir
        ids = self.pid[s]
        return Snapshot(ids[m], xy[m], ids[~m], xy[~m], v[~m],
                        ~np.isin(ids[~m], list(self.gk)), self.meta.pitch_length,
                        self.meta.pitch_width)

    def xy_of(self, frame: int, player_id: int, attack_dir: int) -> np.ndarray | None:
        i = self.pos.get((frame, player_id))
        return None if i is None else self.xy[i] * attack_dir


def shots(g: GameData) -> pd.DataFrame:
    rows = []
    for d in g.decisions[g.decisions.end_type == "shot"].itertuples():
        c = g.xy_of(int(d.frame_end), int(d.player_id), int(d.attack_dir))
        if c is None:
            continue
        snap = g.snap(int(d.frame_end), int(d.team_id), int(d.attack_dir))
        f = state_features(c, snap).iloc[0]
        rows.append({"match_id": g.match_id, "event_id": d.event_id, "team_id": d.team_id,
                     "frame": int(d.frame_end), "dist_goal": f.dist_goal,
                     "goal_angle": f.goal_angle, "n_block": f.n_block,
                     "nearest_def": f.nearest_def, "goal": bool(d.lead_to_goal)})
    return pd.DataFrame(rows)


def passes(g: GameData) -> pd.DataFrame:
    d = g.decisions
    d = d[(d.end_type == "pass") & d.target_id.notna()
          & d.pass_outcome.isin(["successful", "unsuccessful"])]
    rows = []
    for r in d.itertuples():
        t0, a = int(r.frame_end), int(r.attack_dir)
        c, t = g.xy_of(t0, int(r.player_id), a), g.xy_of(t0, int(r.target_id), a)
        snap = g.snap(t0, int(r.team_id), a)
        if c is None or t is None or snap is None:
            continue
        f = pass_features(c, t, snap).iloc[0].to_dict()
        rows.append({"match_id": g.match_id, "event_id": r.event_id,
                     "completed": r.pass_outcome == "successful"} | f)
    return pd.DataFrame(rows)


def carries(g: GameData) -> pd.DataFrame:
    d = g.decisions
    d = d[d.carry.fillna(False).astype(bool)]
    rows = []
    for r in d.itertuples():
        a = int(r.attack_dir)
        c = g.xy_of(int(r.frame_start), int(r.player_id), a)
        e = g.xy_of(int(r.frame_end), int(r.player_id), a)
        snap = g.snap(int(r.frame_start), int(r.team_id), a)
        if c is None or e is None or snap is None or np.hypot(*(e - c)) < 2:
            continue
        f = carry_features(c, e, snap).iloc[0].to_dict()
        rows.append({"match_id": g.match_id, "event_id": r.event_id,
                     "kept": r.end_type != "possession_loss"} | f)
    return pd.DataFrame(rows)


def states(g: GameData, shot_xg: pd.Series, step: int = 5, horizon: int = 100) -> pd.DataFrame:
    """V training rows. `shot_xg`: xG per shot event_id (from the xG model)."""
    b = g.ball.sort_values("frame").reset_index(drop=True)
    meta = g.meta
    team_of_group = {"home team": meta.home_team_id, "away team": meta.away_team_id}
    poss = b.poss_group.map(team_of_group).to_numpy(dtype=float)  # NaN = dead / unknown
    frames = b.frame.to_numpy()
    # own-team shots: (frame, team, xG)
    sh = g.decisions[g.decisions.end_type == "shot"][["event_id", "frame_end", "team_id"]]
    sh = sh.assign(xg=sh.event_id.map(shot_xg).fillna(0.0))
    rows = []
    for i in range(0, len(b), step):
        team = poss[i]
        if np.isnan(team) or np.isnan(b.ball_x[i]):
            continue
        f = int(frames[i])
        per = int(b.period[i])
        a = meta.attack_dir(int(team), per)
        # window ends at horizon or when the other team gains possession
        w = poss[i + 1:i + 1 + horizon]
        other = np.flatnonzero((~np.isnan(w)) & (w != team))
        end = f + (int(other[0]) + 1 if len(other) else horizon)
        hit = sh[(sh.team_id == team) & (sh.frame_end > f) & (sh.frame_end <= end)]
        snap = g.snap(f, int(team), a)
        if snap is None:
            continue
        feats = state_features(np.array([b.ball_x[i], b.ball_y[i]]) * a, snap).iloc[0]
        rows.append({"match_id": g.match_id, "frame": f, "team_id": int(team),
                     "y_xg": float(hit.xg.sum()), "y_shot": bool(len(hit))} | feats.to_dict())
    return pd.DataFrame(rows)
