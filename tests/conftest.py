"""Synthetic fixtures only. No SkillCorner data is used in tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sightline.io.pose import KEY_JOINTS
from sightline.io.tracking import MatchMeta


def skeleton_row(yaw: float, cx: float = 0.0, cy: float = 0.0, err: float = 5.0,
                 torso_yaw: float | None = None) -> dict:
    """A player facing `yaw` (head) and `torso_yaw` (shoulders, defaults to yaw)."""
    torso_yaw = yaw if torso_yaw is None else torso_yaw
    f = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-f[1], f[0]])  # +90 deg from facing
    tf = np.array([np.cos(torso_yaw), np.sin(torso_yaw)])
    tleft = np.array([-tf[1], tf[0]])
    c = np.array([cx, cy])
    pts = {
        "nose": c + 0.10 * f, "lEar": c + 0.08 * left, "rEar": c - 0.08 * left,
        "lEye": c + 0.08 * f + 0.03 * left, "rEye": c + 0.08 * f - 0.03 * left,
        "neck": c, "lShoulder": c + 0.20 * tleft, "rShoulder": c - 0.20 * tleft,
        "midHip": c, "lHip": c + 0.10 * tleft, "rHip": c - 0.10 * tleft,
    }
    row = {}
    for j in KEY_JOINTS:
        row[f"{j}_x"], row[f"{j}_y"], row[f"{j}_err"] = pts[j][0], pts[j][1], err
    return row


@pytest.fixture
def meta() -> MatchMeta:
    players = pd.DataFrame({
        "player_id": [1, 2, 3, 11, 12, 13],
        "team_id": [100, 100, 100, 200, 200, 200],
        "number": [1, 2, 3, 1, 2, 3], "short_name": list("abcdef"),
        "role": ["GK", "CM", "CF", "GK", "CB", "CB"],
        "position_group": ["Goalkeeper", "Midfield", "Forward", "Goalkeeper", "Def", "Def"],
    })
    return MatchMeta(match_id=1, home_team_id=100, away_team_id=200, pitch_length=105.0,
                     pitch_width=68.0,
                     periods=pd.DataFrame({"period": [1, 2], "start_frame": [0, 1000],
                                           "end_frame": [999, 1999]}),
                     players=players, home_side=["left_to_right", "right_to_left"])
