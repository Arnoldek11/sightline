"""Phase 0 / Gate G0 audit on a full pose match.

Streams the pose zip (never extracted), parses only the frames needed, and reports:
  * pose <-> tracking sync (midHip vs tracking XY, every 1 s)
  * pose coverage overall and for the carrier at pass release
  * receiver sanity: P(seen | intended receiver) vs P(seen | other teammate), over a
    cone x memory grid, plus a 180-degree placebo rotation

Deterministic prototype (no Monte Carlo yet). Usage:
    uv run python scripts/audit_pose.py 1925299 [--out reports/audit_1925299.json]
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sightline.paths import data_dir, reports_dir

MAX_ERR_CM = 15.0
CONES = [45, 60, 90]
MEMORIES_S = [1.0, 1.5, 2.5]
MAX_MOVE_M = 3.0
TRACK_FPS = 10
FRAME_RE = re.compile(rb'"frame":\s*(\d+)')


def pose_frame(t: int) -> int:
    return int(round(2.5 * t))


def head_yaw(j: dict) -> float | None:
    need = ("nose", "lEar", "rEar")
    if any(k not in j or j[k]["p90_mae_cm"] > MAX_ERR_CM for k in need):
        return None
    n = np.array(j["nose"]["xyz"][:2])
    e = (np.array(j["lEar"]["xyz"][:2]) + np.array(j["rEar"]["xyz"][:2])) / 2
    v = n - e
    if np.hypot(*v) < 0.02:
        return None
    return float(np.arctan2(v[1], v[0]))


def torso_yaw(j: dict) -> float | None:
    if any(k not in j or j[k]["p90_mae_cm"] > MAX_ERR_CM for k in ("lShoulder", "rShoulder")):
        return None
    lv = np.array(j["lShoulder"]["xyz"][:2])
    rv = np.array(j["rShoulder"]["xyz"][:2])
    d = rv - lv
    w = np.hypot(*d)
    if not 0.15 <= w <= 0.60:
        return None
    return float(np.arctan2(d[0], -d[1]))  # l->r rotated +90 deg (SkillCorner convention)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def load_tracking(path: Path) -> dict[int, dict[int, tuple[float, float]]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["player_data"]:
                out[r["frame"]] = {p["player_id"]: (p["x"], p["y"]) for p in r["player_data"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("match_id", type=int)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    mid = a.match_id
    mdir = data_dir() / "opendata" / "data" / "matches" / str(mid)

    team_of = {p["id"]: p["team_id"] for p in
               json.load(open(mdir / f"{mid}_match.json", encoding="utf-8"))["players"]}
    track = load_tracking(mdir / f"{mid}_tracking_extrapolated.jsonl")
    ev = pd.read_csv(mdir / f"{mid}_dynamic_events.csv", low_memory=False)
    pp = ev[(ev.event_type == "player_possession") & (ev.end_type == "pass")
            & ev.player_targeted_id.notna()].copy()
    pp["player_targeted_id"] = pp.player_targeted_id.astype(int)

    max_win = int(max(MEMORIES_S) * TRACK_FPS)
    need: dict[int, set[int]] = defaultdict(set)  # pose frame -> carrier ids needed
    for r in pp.itertuples():
        for t in range(r.frame_end - max_win, r.frame_end + 1):
            need[pose_frame(t)].add(r.player_id)

    carrier_yaw: dict[tuple[int, int], tuple[float | None, float | None]] = {}
    sync_off, cover_n, cover_j, head_torso = [], 0, 0, []
    with zipfile.ZipFile(data_dir() / "pose" / f"{mid}.jsonl.zip") as zf, \
            zf.open(f"{mid}.jsonl") as fh:
        for line in fh:
            m = FRAME_RE.search(line, 0, 40)
            if not m:
                continue
            f = int(m.group(1))
            stat = f % 25 == 0
            if not stat and f not in need:
                continue
            rec = json.loads(line)
            for p in rec["player_data"]:
                j = p.get("joints")
                if stat:
                    cover_n += 1
                    cover_j += bool(j)
                    t = f * 2 // 5
                    if j and "midHip" in j and t in track and p["player_id"] in track[t]:
                        tx, ty = track[t][p["player_id"]]
                        hx, hy = j["midHip"]["xyz"][:2]
                        sync_off.append(np.hypot(hx - tx, hy - ty))
                    if j:
                        h, s = head_yaw(j), torso_yaw(j)
                        if h is not None and s is not None:
                            head_torso.append(abs(wrap(h - s)))
                if f in need and p["player_id"] in need[f]:
                    carrier_yaw[(f, p["player_id"])] = (
                        (head_yaw(j), torso_yaw(j)) if j else (None, None))

    # receiver sanity
    rows = []
    for r in pp.itertuples():
        t0, cid = r.frame_end, r.player_id
        if t0 not in track or cid not in track[t0]:
            continue
        mates = [pid for pid in track[t0] if team_of.get(pid) == team_of.get(cid) and pid != cid]
        if r.player_targeted_id not in mates:
            continue
        rel = {"rel_has_pose": carrier_yaw.get((pose_frame(t0), cid), (None,))[0] is not None}
        for src in ("head", "torso"):
            k = 0 if src == "head" else 1
            samples = []  # (lag_s, yaw, {mate: angle_off, moved})
            for lag in range(0, max_win + 1):
                t = t0 - lag
                y = carrier_yaw.get((pose_frame(t), cid), (None, None))[k]
                if y is None or t not in track or cid not in track[t]:
                    continue
                cx, cy = track[t][cid]
                d = {}
                for pid in mates:
                    if pid in track[t]:
                        mx, my = track[t][pid]
                        ang = abs(wrap(np.arctan2(my - cy, mx - cx) - y))
                        ex, ey = track[t0][pid]
                        d[pid] = (ang, np.hypot(ex - mx, ey - my))
                samples.append((lag / TRACK_FPS, d))
            for cone in CONES:
                for mem in MEMORIES_S:
                    for placebo in (False, True):
                        ss = [d for lag, d in samples if lag <= mem + 1e-9]
                        if not ss:
                            continue
                        c = np.radians(cone)

                        def seen(pid, ss=ss, c=c, placebo=placebo):
                            for d in ss:
                                if pid in d:
                                    ang, mv = d[pid]
                                    if placebo:
                                        ang = np.pi - ang
                                    if ang <= c and mv <= MAX_MOVE_M:
                                        return True
                            return False

                        others = [seen(p) for p in mates if p != r.player_targeted_id]
                        key = f"{src}_c{cone}_m{mem}{'_placebo' if placebo else ''}"
                        rel[key + "_recv"] = seen(r.player_targeted_id)
                        rel[key + "_other"] = float(np.mean(others)) if others else np.nan
        rows.append(rel)
    df = pd.DataFrame(rows)

    def q(x, p):
        return round(float(np.percentile(x, p)), 3) if len(x) else None

    out = {
        "match_id": mid,
        "player_frames_sampled": cover_n,
        "share_player_frames_with_pose": round(cover_j / max(cover_n, 1), 3),
        "sync_midhip_vs_tracking_m": {"n": len(sync_off), "median": q(sync_off, 50),
                                      "p95": q(sync_off, 95), "share_gt_1m":
                                      round(float(np.mean(np.array(sync_off) > 1)), 4)},
        "head_vs_torso_abs_deg": {"median": q(np.degrees(head_torso), 50),
                                  "p90": q(np.degrees(head_torso), 90)},
        "passes_with_target": len(pp),
        "passes_evaluable": len(df),
        "carrier_pose_at_release": round(float(df.rel_has_pose.mean()), 3) if len(df) else None,
        "receiver_sanity": {},
    }
    for src in ("head", "torso"):
        for cone in CONES:
            for mem in MEMORIES_S:
                for placebo in (False, True):
                    key = f"{src}_c{cone}_m{mem}{'_placebo' if placebo else ''}"
                    if key + "_recv" not in df:
                        continue
                    sub = df[df[key + "_recv"].notna()]
                    out["receiver_sanity"][key] = {
                        "n": len(sub),
                        "p_seen_receiver": round(float(sub[key + "_recv"].astype(float).mean()), 3),
                        "p_seen_other": round(float(sub[key + "_other"].mean()), 3),
                    }
    path = a.out or reports_dir() / f"audit_{mid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "receiver_sanity"}, indent=2))
    for k in ("head_c60_m1.5", "head_c60_m1.5_placebo", "torso_c60_m1.5",
              "torso_c60_m1.5_placebo"):
        print(k, out["receiver_sanity"].get(k))


if __name__ == "__main__":
    main()
