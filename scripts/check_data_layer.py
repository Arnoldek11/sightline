"""Phase 1 checks on real data (run after `reproduce.py --full`).

1. Event coordinates live in the tracking frame (carrier XY at frame_start vs event x/y).
2. Counter definitions: counts per game and rule-vs-label agreement (PLAN 4.1).
3. Pose coverage of decisions in the pose games, by subset (all / open play / counters).
Writes reports/data_layer_checks.json and prints a summary.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sightline.paths import reports_dir
from sightline.pipeline import POSE_GAMES, load_processed
from sightline.sync import tracking_to_pose_frame


def main() -> None:
    full = json.loads((reports_dir() / "data_layer_full.json").read_text())
    games = [g["match_id"] for g in full["games"]]
    out: dict = {}

    # 1. event vs tracking coordinates
    errs = []
    for m in games[:5]:
        d = load_processed(m, "decisions")
        p = load_processed(m, "players")[["frame", "player_id", "x", "y"]]
        j = d.merge(p, left_on=["frame_start", "player_id"], right_on=["frame", "player_id"])
        errs.append(np.hypot(j.x_start - j.x, j.y_start - j.y))
    e = np.concatenate(errs)
    out["event_vs_tracking_m"] = {"n": len(e), "median": float(np.median(e)),
                                  "p95": float(np.percentile(e, 95))}

    # 2. counter definitions
    cs = pd.concat([load_processed(m, "counters").assign(match_id=m) for m in games])
    ds = pd.concat([load_processed(m, "decisions").assign(match_id=m) for m in games])
    per_game = cs.groupby("match_id")[["rule_counter", "label_counter", "fast_attack"]].sum()
    out["regains"] = len(cs)
    out["counters_total"] = per_game.sum().astype(int).to_dict()
    out["counters_per_game_min_median"] = {c: [int(per_game[c].min()),
                                               float(per_game[c].median())]
                                           for c in per_game}
    ct = pd.crosstab(cs.rule_counter, cs.label_counter)
    out["rule_x_label"] = {f"rule={r}_label={c}": int(ct.loc[r, c])
                           for r in ct.index for c in ct.columns}
    out["decisions_in"] = {c: int(ds[c].sum()) for c in
                           ("rule_counter", "label_counter", "fast_attack", "is_counter_label")}
    out["decisions_in"]["any_counter"] = int((ds.rule_counter | ds.label_counter).sum())
    out["decisions_total"] = len(ds)
    out["decisions_open_play"] = int(ds.is_open_play.sum())

    # 3. pose coverage at decision release, pose games
    cov = {}
    for m in POSE_GAMES:
        d = load_processed(m, "decisions")
        pz = load_processed(m, "pose")
        ok = pz[pz.sync_ok & pz.head_yaw.notna()][["frame", "player_id"]]
        ok = set(zip(ok.frame, ok.player_id, strict=True))
        d["has_pose"] = [(f, p) in ok for f, p in zip(d.frame_end, d.player_id, strict=True)]
        cov[m] = {name: {"n": int(sub.sum()), "with_pose": int((sub & d.has_pose).sum())}
                  for name, sub in {
                      "all": d.frame_end.notna(), "open_play": d.is_open_play,
                      "rule_counter": d.rule_counter, "label_counter": d.label_counter,
                      "fast_attack": d.fast_attack}.items()}
        assert tracking_to_pose_frame(0) == 0
    out["pose_coverage_at_release"] = cov

    path = reports_dir() / "data_layer_checks.json"
    path.write_text(json.dumps(out, indent=2, default=int))
    print(json.dumps(out, indent=2, default=int))


if __name__ == "__main__":
    main()
