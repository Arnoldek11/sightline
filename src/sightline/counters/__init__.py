"""Counter-attack / fast-attack detection (PLAN 4.1).

A *regain* is the first on-ball possession of a team possession that starts with an in-play
recovery or interception. From the regain frame t0, inside the same team possession:

* rule counter: within `window_s` the ball progresses >= `min_progress_m` towards the
  opponent goal, OR it reaches the final third while <= `max_defenders_goal_side` outfield
  opponents were goal-side of the ball at t0;
* label counter: any possession of the sequence sits in a SkillCorner quick_break /
  transition phase;
* fast attack (widened, F4): shot or box entry within `fast_attack_window_s`;
* **counter (headline, DECISIONS D-005)** = label counter OR (rule counter AND fast attack).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sightline.io.tracking import TRACKING_FPS, MatchMeta

BOX_DEPTH_M, BOX_HALF_WIDTH_M = 16.5, 20.16
FLAGS = ("counter", "rule_counter", "label_counter", "fast_attack")


@dataclass(frozen=True)
class CounterConfig:
    window_s: float = 10.0
    min_progress_m: float = 25.0
    max_defenders_goal_side: int = 5
    fast_attack_window_s: float = 15.0

    @classmethod
    def from_dict(cls, d: dict) -> CounterConfig:
        p = d.get("primary", d)
        return cls(window_s=p.get("window_s", 10.0),
                   min_progress_m=p.get("min_progress_m", 25.0),
                   max_defenders_goal_side=p.get("max_defenders_goal_side", 5),
                   fast_attack_window_s=d.get("fast_attack_window_s", 15.0))


def _sequence_end(decisions: pd.DataFrame, i: int) -> int:
    """Last frame of the team possession starting at decision row i."""
    r = decisions.iloc[i]
    later = decisions.iloc[i + 1:]
    other = later[(later.team_id != r.team_id) | (later.period != r.period)]
    if len(other):
        return int(other.frame_start.iloc[0]) - 1
    return int(max(r.frame_end, later.frame_end.max() if len(later) else r.frame_end))


def detect_counters(decisions: pd.DataFrame, ball: pd.DataFrame, players: pd.DataFrame,
                    meta: MatchMeta, cfg: CounterConfig | None = None) -> pd.DataFrame:
    """One row per regain with counter flags. `decisions` from io.events.decision_points."""
    cfg = cfg or CounterConfig()
    decisions = decisions.sort_values("frame_start").reset_index(drop=True)
    ball_i = ball.set_index("frame")
    gk = set(meta.players.loc[meta.players.role == "GK", "player_id"])
    L = meta.pitch_length
    rows = []
    for i in np.flatnonzero(decisions.is_regain.to_numpy()):
        r = decisions.iloc[i]
        t0 = int(r.frame_start)
        seq_end = _sequence_end(decisions, i)
        d = meta.attack_dir(int(r.team_id), int(r.period))
        w_end = min(seq_end, t0 + int(cfg.window_s * TRACKING_FPS))
        fa_end = min(seq_end, t0 + int(cfg.fast_attack_window_s * TRACKING_FPS))
        b = ball_i.loc[(ball_i.index >= t0) & (ball_i.index <= fa_end), ["ball_x", "ball_y"]]
        b = b.dropna()
        if b.empty or t0 not in b.index:
            continue
        bx = b.ball_x * d
        bx0 = float(bx.loc[t0])
        in_w = bx.index <= w_end
        progress = float(bx[in_w].max() - bx0)
        reached_ft = bool((bx[in_w] >= L / 6).any())
        opp = players[(players.frame == t0) & (players.team_id != r.team_id)
                      & ~players.player_id.isin(gk)]
        goal_side = int((opp.x * d > bx0).sum())
        box = (bx >= L / 2 - BOX_DEPTH_M) & (b.ball_y.abs() <= BOX_HALF_WIDTH_M)
        seq = decisions[(decisions.frame_start >= t0) & (decisions.frame_start <= seq_end)
                        & (decisions.team_id == r.team_id)]
        shot_fast = bool(((seq.end_type == "shot") & (seq.frame_end <= fa_end)).any())
        rows.append({
            "regain_event_id": r.event_id, "period": int(r.period), "team_id": int(r.team_id),
            "frame_regain": t0, "frame_seq_end": seq_end,
            "ball_x_att_regain": bx0, "progress_m": progress,
            "reached_final_third": reached_ft, "defenders_goal_side": goal_side,
            "rule_counter": (progress >= cfg.min_progress_m)
            or (reached_ft and goal_side <= cfg.max_defenders_goal_side),
            "label_counter": bool(seq.is_counter_label.any()),
            "fast_attack": shot_fast or bool(box.any()),
            "n_decisions": len(seq),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["counter"] = out.label_counter | (out.rule_counter & out.fast_attack)
    return out


def tag_decisions(decisions: pd.DataFrame, counters: pd.DataFrame,
                  max_len_s: float = 15.0) -> pd.DataFrame:
    """Add `regain_event_id` + counter flags to decisions inside each regain sequence."""
    out = decisions.copy()
    for c in FLAGS:
        out[c] = False
    out["regain_event_id"] = pd.NA
    for r in counters.itertuples():
        end = min(r.frame_seq_end, r.frame_regain + int(max_len_s * TRACKING_FPS))
        m = ((out.team_id == r.team_id) & (out.frame_start >= r.frame_regain)
             & (out.frame_start <= end))
        out.loc[m, "regain_event_id"] = r.regain_event_id
        for c in FLAGS:
            out.loc[m, c] = getattr(r, c)
    return out
