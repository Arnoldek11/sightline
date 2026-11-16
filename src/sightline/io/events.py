"""SkillCorner dynamic events + phases of play.

Event x/y are pitch metres, origin at the centre, but **rotated 180 deg so the event's team
always attacks +x** (`attacking_side` per row). Pitch-frame (tracking) coordinates are
event * attack_dir with attack_dir = +1 for left_to_right, -1 otherwise. Verified on real data
in scripts/check_data_layer.py (carrier tracking XY vs event start: median 0.19 m).
decision_points() exposes both: `x_start_att` (attacking frame) and `x_start` (pitch frame).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sightline.io.tracking import match_dir

COUNTER_PHASE_LABELS = ("quick_break", "transition")
REGAIN_START_TYPES = ("recovery", "pass_interception")

_DECISION_COLS = {
    "event_id": "event_id",
    "match_id": "match_id",
    "period": "period",
    "frame_start": "frame_start",
    "frame_end": "frame_end",
    "player_id": "player_id",
    "team_id": "team_id",
    "player_position": "player_position",
    "start_type": "start_type",
    "end_type": "end_type",
    "player_targeted_id": "target_id",
    "pass_outcome": "pass_outcome",
    "x_start": "x_start",
    "y_start": "y_start",
    "x_end": "x_end",
    "y_end": "y_end",
    "phase_index": "phase_index",
    "team_in_possession_phase_type": "phase_type",
    "first_player_possession_in_team_possession": "first_in_team_possession",
    "game_interruption_before": "interruption_before",
    "lead_to_goal": "lead_to_goal",
    "carry": "carry",
    "possession_epv_for_start": "sc_epv_start",
}


def load_events(match_id: int, root: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(match_dir(match_id, root) / f"{match_id}_dynamic_events.csv",
                       low_memory=False)


def load_phases(match_id: int, root: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(match_dir(match_id, root) / f"{match_id}_phases_of_play.csv")


def decision_points(events: pd.DataFrame) -> pd.DataFrame:
    """One row per on-ball possession; the decision is taken at `frame_end`."""
    pp = events[events.event_type == "player_possession"]
    d = pp[list(_DECISION_COLS)].rename(columns=_DECISION_COLS).copy()
    d["attack_dir"] = np.where(pp.attacking_side.to_numpy() == "left_to_right", 1, -1)
    for c in ("x_start", "y_start", "x_end", "y_end"):
        d[f"{c}_att"] = d[c]
        d[c] = d[c] * d.attack_dir  # 180 deg rotation back to the pitch frame
    d["target_id"] = d.target_id.astype("Int64")
    d["first_in_team_possession"] = d.first_in_team_possession.fillna(False).astype(bool)
    d["is_open_play"] = d.interruption_before.isna()
    d["is_regain"] = d.first_in_team_possession & d.start_type.isin(REGAIN_START_TYPES)
    d["is_counter_label"] = d.phase_type.isin(COUNTER_PHASE_LABELS)
    return d.sort_values("frame_start").reset_index(drop=True)
