# DECISIONS

## D-000 (16 Nov 2026): data outside the repo
Data lives in `$SIGHTLINE_DATA_DIR` (here `C:\Users\arnol\sightline-data`), not in the OneDrive-synced repo. `./data` is still the default for a clean clone.

## D-001 (16 Nov 2026): pose sample for counters is too small → proposal, NOT yet agreed
Facts: 67 counter decision points with pose (2 games), 857 across 20 games, 2,059 open-play+set decisions in the pose games.
Proposal: apply the F1 widening up front for the *pose-based* analysis (all open-play decisions in the 2 pose games, ≈1,800) and use it for validation and the vision model. Counters stay the headline subset, made large enough by the tracking-only orientation model (Phase 4) on all 20 games. If G1b fails, counters become case studies and the headline moves to open play.
Status: adopted as working default (see update below).

## D-002 (16 Nov 2026): kloppy for tracking, own streaming loader for pose
kloppy 3.19.1 loads the tracking. Pose is streamed line by line from the HF zip (constant memory, ~40 s per match for a filtered pass).

## D-003 (16 Nov 2026): head yaw available
Head yaw = ears-midpoint → nose (all head joints present when pose exists). Torso facing kept as a fallback and for comparison (median head–torso difference 17°).

## D-001 update (16 Nov 2026): adopted as the working default
The data layer tags every decision with `is_open_play`, `rule_counter`, `label_counter`, `fast_attack`, so the analysis set is a filter, not a code path. Working default: pose analysis on all open-play decisions in the 2 pose games (1,899 decisions, 1,593 with synced head pose at release); counters are the highlighted subset. Reversible by config.

## D-004 (16 Nov 2026): event coordinates are in the attacking frame
SkillCorner event x/y are rotated 180° so the event's team attacks +x. `decision_points()` returns pitch-frame `x_start…y_end` (= event × attack_dir, from `attacking_side`) plus the native `*_att` columns. Check: carrier tracking XY vs event start, median 0.06 m, p95 0.64 m (5 games, n = 5,023). Before the fix: median 7.5 m.

## D-005 (16 Nov 2026): counter definition → AGREED (16 Nov)
All 20 games, 3,050 in-play regains (recovery / interception):

| Definition | Sequences | in pose games | Decisions | Fast-attack share |
|---|---|---|---|---|
| SkillCorner label (quick_break / transition) | 278 | 21 | 1,329 | 0.58 |
| PLAN rule (≥25 m in 10 s OR final third with ≤5 goal-side) | 1,077 | 107 | 4,377 | 0.34 |
| label OR (rule AND fast attack) | 480 | 36 | 2,169 | 0.76 |

Rule vs label: both 271, rule only 806, label only 7. The label is almost a strict subset of the rule, and the rule flags 35% of regains, which is too many to call counters (any long ball after a regain counts).
Proposal: **headline counter = label OR (rule AND fast attack)**: SkillCorner's expert label plus rule-based sequences that actually threatened (shot or box entry within 15 s). The PLAN rule stays as a sensitivity variant. Not tuned on results (no loss numbers exist yet).

Implemented as `counter` in `sightline.counters` (counters.parquet + decisions.parquet). Decisions inside the first 15 s of a headline counter: 1,394 (all games). The other flags stay available as sensitivity variants.
