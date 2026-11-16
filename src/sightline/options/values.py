"""Value every option at every decision (PLAN 4.2).

V(option) = P(success) * V(state after success) + (1 - P(success)) * V_loss
V_loss    = - V_opp(spot): the opponent's value with the ball at the loss spot (their frame).
Units: expected goals from own shots in the next 10 s (D-008). Shots: V = xG.
Loss spot = the target point (passes) / end point (carries), a simplification (D-008).
Ex-ante only: everything uses the snapshot at the release frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sightline.options.data import GameData
from sightline.options.geometry import (
    carry_endpoints,
    carry_features,
    pass_features,
    state_features,
)
from sightline.options.models import CrossFitted


@dataclass
class OptionModels:
    xg: CrossFitted
    p_pass: CrossFitted
    p_carry: CrossFitted
    value: CrossFitted
    shot_max_dist: float = 30.0


def option_table(g: GameData, decisions: pd.DataFrame, m: OptionModels) -> pd.DataFrame:
    """Long table: one row per (decision, option)."""
    mid = g.match_id
    out = []
    for d in decisions.itertuples():
        t0, a, team = int(d.frame_end), int(d.attack_dir), int(d.team_id)
        snap = g.snap(t0, team, a)
        c = g.xy_of(t0, int(d.player_id), a)
        if snap is None or c is None:
            continue
        opp = snap.flipped()
        keep = snap.mate_ids != d.player_id
        mates, mate_ids = snap.mates[keep], snap.mate_ids[keep]
        ends = carry_endpoints(c, pitch_length=snap.pitch_length, pitch_width=snap.pitch_width)
        spots = np.vstack([mates, ends])
        v_after = m.value.predict(mid, state_features(spots, snap))
        v_opp = m.value.predict(mid, state_features(-spots, opp))
        p = np.r_[m.p_pass.predict(mid, pass_features(c, mates, snap)),
                  m.p_carry.predict(mid, carry_features(c, ends, snap))]
        types = ["pass"] * len(mates) + ["carry"] * len(ends)
        ids = [str(int(i)) for i in mate_ids] + [f"carry{k}" for k in range(len(ends))]
        rows = pd.DataFrame({"event_id": d.event_id, "option_type": types, "option_id": ids,
                             "x_att": spots[:, 0], "y_att": spots[:, 1], "p_success": p,
                             "v_success": v_after, "v_loss": -v_opp})
        sf = state_features(c, snap)
        if sf.dist_goal.iloc[0] <= m.shot_max_dist:
            xg = float(m.xg.predict(mid, sf)[0])
            rows = pd.concat([rows, pd.DataFrame([{
                "event_id": d.event_id, "option_type": "shot", "option_id": "shot",
                "x_att": c[0], "y_att": c[1], "p_success": 1.0, "v_success": xg,
                "v_loss": 0.0}])], ignore_index=True)
        rows["value"] = rows.p_success * rows.v_success + (1 - rows.p_success) * rows.v_loss
        chosen = (str(int(d.target_id)) if d.end_type == "pass" and pd.notna(d.target_id)
                  else "shot" if d.end_type == "shot" else None)
        rows["is_chosen"] = rows.option_id == chosen
        rows["v_current"] = float(m.value.predict(mid, sf)[0])
        out.append(rows)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
