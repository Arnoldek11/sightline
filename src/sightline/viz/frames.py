"""Render a decision point: carrier's vision cone, teammates coloured by P(seen)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402
from mplsoccer import Pitch  # noqa: E402


def render_decision(ax, frame_players: pd.DataFrame, ball_xy, carrier_id: int,
                    head_yaw: float, options: pd.DataFrame, target_id: int | None,
                    cone_deg: float = 60.0, pitch_length: float = 105.0,
                    pitch_width: float = 68.0, title: str = "") -> None:
    """frame_players: player_id, team_id, x, y at release. options: option_id, p_seen."""
    pitch = Pitch(pitch_type="skillcorner", pitch_length=pitch_length,
                  pitch_width=pitch_width, line_color="#9aa0a6", pitch_color="white",
                  line_zorder=1)
    pitch.draw(ax=ax)
    c = frame_players.set_index("player_id").loc[carrier_id]
    if np.isfinite(head_yaw):
        ax.add_patch(Wedge((c.x, c.y), 40, np.degrees(head_yaw) - cone_deg,
                           np.degrees(head_yaw) + cone_deg, color="#f2b705", alpha=0.18,
                           zorder=2))
        ax.plot([c.x, c.x + 6 * np.cos(head_yaw)], [c.y, c.y + 6 * np.sin(head_yaw)],
                color="#c48a00", lw=2, zorder=5)
    opp = frame_players[frame_players.team_id != c.team_id]
    ax.scatter(opp.x, opp.y, s=60, c="#d0d0d0", edgecolors="#606060", zorder=3)
    mates = frame_players.merge(options, left_on="player_id", right_on="option_id")
    sc = ax.scatter(mates.x, mates.y, s=90, c=mates.p_seen, cmap="RdYlGn", vmin=0, vmax=1,
                    edgecolors="black", zorder=4)
    ax.scatter([c.x], [c.y], s=140, c="#1f4e9c", edgecolors="black", zorder=6)
    if target_id is not None and target_id in set(mates.player_id):
        t = mates.set_index("player_id").loc[target_id]
        ax.scatter([t.x], [t.y], s=320, facecolors="none", edgecolors="#1f4e9c", lw=2,
                   zorder=6)
    if ball_xy is not None and np.all(np.isfinite(ball_xy)):
        ax.scatter([ball_xy[0]], [ball_xy[1]], s=25, c="black", zorder=7)
    ax.set_title(title, fontsize=8)
    plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.01, label="P(seen)")
