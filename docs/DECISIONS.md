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

## D-006 (16 Nov 2026): pose error is persistent over the memory window
Independent per-frame joint noise implies head-yaw sd ≈ 16° per frame, but the observed frame-to-frame head-yaw noise is ≈ 2.3° (torso 1.5°; robust second differences, 1.4M frames, 1925299). The pose output is temporally smooth, so its error is mostly a slowly varying bias. P(seen) therefore uses **one noise draw per Monte Carlo sample, shared by all lags of a decision** (`sample_yaw(noise=...)`). Independent per-frame draws inflated "seen at least once" (other teammates 0.74 vs 0.64). The ≈16° spread is still an upper bound: head joints' errors are probably correlated with each other, which would shrink yaw error further.

## D-007 (16 Nov 2026): default cone ±60°, memory 1.5 s, chosen a priori, not tuned
Kept the PLAN defaults. On the receiver-sanity AUC the grid optimum is ±60° / 1.0 s (0.717 vs 0.684 at 1.5 s), but discriminating the receiver is a check, not the objective (we want what the carrier could know about, not a pass predictor), so the default is not moved towards it. The full 3×3 grid is carried to the Phase 5 sensitivity analysis. Options are teammates at any distance (no range limit). Decisions need pose on ≥ 50% of window samples (86.5% of decisions pass).

## D-008 (16 Nov 2026): option values, one currency (expected goals)
- **V(state) = E[xG of own shots in the next 10 s, before the opponent regains]**, LightGBM on ball-location state features (attacking frame). The PLAN target was "shot or box entry within 10 s"; changed so pass, carry and shot options share one unit (a shot option is worth its xG).
- **Horizon:** 10 s (PLAN). Tested 30 s / to possession end because shots looked over-valued. It barely changed that (shot best when available 0.66 vs 0.69) and made V worse (R² 0.036 vs 0.055, SkillCorner EPV ρ 0.72 vs 0.80), so it was **rejected**. Outputs kept under the `_h10` / default tags for the record.
- **xG** = logistic(distance, goal angle, n outfield defenders in the ball–post–post triangle, nearest defender), 526 shots. Blocking was added after shots dominated the option ranking; it improves Brier 0.0993 → 0.0978 and AUC 0.728 → 0.739.
- **P(pass)** = logistic on physics: time-to-intercept margins along the line (ball 15 m/s; defenders 0.3 s reaction, 7 m/s, 1 m reach, current velocity), length, forward gain, defenders within 3 m of the lane, nearest defender to target. Point-to-player at release.
- **P(carry)** = logistic on carry physics (carrier 5 m/s), trained on possessions flagged `carry`; outcome = possession not ended by `possession_loss`.
- **V_loss** = −V_opp at the target / end point (the opponent's value in their frame). Simplification: the true loss spot for an intercepted pass is on the lane.
- **Cross-fitting:** 5 game folds; every game, including both pose games, is valued by models that never saw it.
- **Decision set:** possessions ending in a pass (chosen = pass to the intended target) or a shot. Carry options are "keep carrying" alternatives at the release moment.

## D-009 (16 Nov 2026): shots out of the main loss split → proposal, NOT yet agreed
Shot options remain over-valued: within 30 m, "shoot" is the best option in 69% of decisions, while players shoot in 27%. This is selection bias. xG is learned only from shots players chose to take, so it cannot value shots they rightly passed up, and the "should shoot more" artefact would dominate any headline.
Proposal: the main Vision/Decision split uses **pass + carry options and pass decisions** (1,789 decisions in the pose games). Rationale beyond the artefact: the goal's location is always known, so "did he see the shot?" is not a perception question; Sightline is about seeing teammates and space. Shots are reported separately as a sensitivity analysis.
