# DATA_AUDIT

Status: **complete for G0** (16 Nov 2026). Full-match numbers from `scripts/audit_pose.py` → `reports/audit_<id>.json`.
Source: SkillCorner/opendata (shallow clone, HEAD of 16 Nov 2026) + its `data/bodypose/README.md` and `MANIFEST.json`.

## Q1. How many tracking games? Which ids?
**20 games**, all with `tracking_extrapolated.jsonl` (git-LFS, ~97 MB each):
1874553, 1886347, 1899585, 1925299, 1927964, 1953632, 1959846, 1986691, 1996435, 1996436,
2006229, 2006363, 2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236, 2017461.

## Q2. Which games have dynamic events? Are both pose games among them?
**All 20 games** have `dynamic_events.csv` (the "10 games" on the website is outdated). **Both pose games (1925299, 1996435) are covered.** → Fallback F3 not needed.
Event types (1925299): passing_option 2734, player_possession 1054, on_ball_engagement 936, off_ball_run 498.
Highly relevant columns (322 total): per-option `xthreat`, `xpass_completion`, `passing_option_score`, `dangerous`, `difficult_pass_target`, `targeted`/`received`; on possessions `possession_epv_*`, `pass_epv_*`, `n_passing_options*`, `passing_option_ease_start/end`, `overall_pressure_*`.
**Implication for §4.2:** SkillCorner already values passing options. Our own V model stays primary (it must cover carries + shots and be reproducible for tracking-only clubs); SkillCorner xthreat/xpass become the strongest cross-check. Note: SkillCorner's passing options are only the options it detected, not all teammates; coverage check moves to Phase 3.

## Q3. Pose joints: is there a head joint?
**Yes.** 29 joints incl. `nose`, `lEye`, `rEye`, `lEar`, `rEar`, `neck`, shoulders, hips. Head yaw can be computed from ears→nose geometry; torso from shoulders/hips.
In the sample, whenever a player has pose, all head joints are present; median p90 error ≈ 7 cm (head), 5–6 cm (neck/shoulders).
Prior art in the repo: `src/features/pose_orientation.py` (shoulder-only heading with quality gates: error ≤ 15 cm, shoulder width 0.15–0.60 m). We cite it as the baseline and reuse its gates; vision cones, memory, head yaw and loss split are beyond it.

## Q4. Pose ↔ tracking mapping
- Pose 25 fps, tracking 10 fps: `pose_frame = 2.5 × tracking_frame`; every 5th pose frame hits an even tracking frame exactly.
- Each pose record carries `frame`, `period`, `timestamp`, `possession`, and per player `player_id` + interpolated `x`, `y`, `is_detected` → **same player ids as tracking**, no mapping table needed (confirmed on both full matches).
- Joint `xyz` are in pitch metres for x/y (z centroid-relative).
- Full matches, pose midHip vs **tracking-file** XY (every 1 s, n≈61k/63k): median **0.22 m / 0.20 m**, p95 0.79 / 0.57 m, share > 1 m **3.5% / 1.6%** (1925299 / 1996435). Those frames get dropped.

## Q5. Joint errors and carrier pose coverage
Sample (1 phase, 309 frames): 50% of player-frames carry joints; carrier (from `possession.player_id`, set on 71 frames) has pose in 100% of them; share of head joints > 15 cm ≈ 3%.
Full matches: **31% / 36%** of player-frames carry joints, but the **carrier has usable head pose at pass release in 87% / 90%** of passes (the camera follows the ball). 912 / 878 targeted passes. Head vs torso yaw differ by median 17°, p90 39° → head yaw adds real information.

## Q6. Do phase labels identify counters?
**Yes.** `team_in_possession_phase_type` across all 20 games: create 2880, chaotic 1802, finish 1623, build_up 1196, direct 677, set_play 360, **transition 185, quick_break 111**.
Label-based counters = 296 phases (quick_break + transition) → above the ~150 threshold. Per game: min 5, median 15.
**But the pose games are thin:** 1925299 = 11 phases (9 QB + 2 T), 1996435 = 10 (3 QB + 7 T) → only **67 decision points** (player_possession rows) in counters across the two pose games, vs 2,059 decisions in total there. All 20 games: 857 counter decisions / 18,710 total.

## Q7. Does kloppy load this format?
kloppy 3.19.1 installs; SkillCorner's own tutorial (`03_Basics_of_Tracking/..._Kloppy_Tutorial.ipynb`) uses kloppy for this tracking. Load test: `kloppy.skillcorner.load` reads match 1925299 (10 fps, pitch 105×68) → **use kloppy for tracking**. Pose is not a kloppy format → own streaming loader (`ijson`/line-by-line from the zip, constant memory).

## Sizes
Tracking ~20 × 97 MB ≈ 2 GB; pose 2 × ~620 MB zipped (3.3 GB JSON each, streamed from zip, never extracted). Data dir kept outside OneDrive.

## Event coordinates (found in Phase 1)
Event x/y are pitch metres centred on the pitch (no rescaling needed) but **rotated 180° into the attacking frame** of the event's team (`attacking_side`). See DECISIONS D-004.
