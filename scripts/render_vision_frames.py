"""Render N random decision points from the pose games for a human check (PLAN Phase 2).

Output: reports/vision_frames/<match>_<event>.png + index.csv. Usage:
    uv run python scripts/render_vision_frames.py [--n 30] [--seed 7]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sightline.io.tracking import load_meta
from sightline.paths import reports_dir
from sightline.pipeline import POSE_GAMES, load_processed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = reports_dir() / "vision_frames"
    out.mkdir(parents=True, exist_ok=True)
    picks = []
    for mid in POSE_GAMES:
        v = load_processed(mid, "vision")
        ev = v[(v.coverage >= 0.5)].event_id.drop_duplicates()
        picks += [(mid, e) for e in ev.sample(a.n // len(POSE_GAMES), random_state=a.seed)]
    rows = []
    for mid in POSE_GAMES:
        meta = load_meta(mid)
        v = load_processed(mid, "vision")
        d = load_processed(mid, "decisions").set_index("event_id")
        pl = load_processed(mid, "players")
        ball = load_processed(mid, "ball").set_index("frame")
        pz = load_processed(mid, "pose").set_index(["frame", "player_id"]).head_yaw
        names = meta.players.set_index("player_id").short_name
        for _, e in [p for p in picks if p[0] == mid]:
            r = d.loc[e]
            t0 = int(r.frame_end)
            fp = pl[pl.frame == t0][["player_id", "team_id", "x", "y"]]
            yaw = pz.get((t0, int(r.player_id)), np.nan)
            opts = v[v.event_id == e][["option_id", "p_seen"]]
            tgt = None if pd.isna(r.target_id) else int(r.target_id)
            b = ball.loc[t0, ["ball_x", "ball_y"]].to_numpy(float) if t0 in ball.index else None
            fig, ax = plt.subplots(figsize=(9, 6))
            title = (f"{mid} · {e} · {names.get(r.player_id, r.player_id)} · {r.end_type}"
                     f" · {r.phase_type} · P(seen|target)="
                     f"{opts.set_index('option_id').p_seen.get(tgt, np.nan):.2f}")
            from sightline.viz.frames import render_decision
            render_decision(ax, fp, b, int(r.player_id), yaw, opts, tgt,
                            pitch_length=meta.pitch_length, pitch_width=meta.pitch_width,
                            title=title)
            f = out / f"{mid}_{e}.png"
            fig.savefig(f, dpi=110, bbox_inches="tight")
            plt.close(fig)
            rows.append({"match_id": mid, "event_id": e, "file": f.name, "end_type": r.end_type,
                         "phase_type": r.phase_type, "head_yaw_deg": np.degrees(yaw),
                         "reviewer_ok": "", "reviewer_note": ""})
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
    print(f"{len(rows)} frames -> {out}")


if __name__ == "__main__":
    main()
