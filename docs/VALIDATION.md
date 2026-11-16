# VALIDATION

## Gate G0 (16 Nov 2026), `scripts/audit_pose.py`, deterministic (no Monte Carlo), head yaw = ears-midpoint → nose
Rules: joints gated at p90 error ≤ 15 cm; "seen" = receiver within ±θ of head yaw at any 10 fps sample in the last T s, and moved ≤ 3 m since.
Results: `reports/audit_1925299.json`, `reports/audit_1996435.json` (gitignored, regenerate with the script).

| Check | Threshold | 1925299 | 1996435 | Status |
|---|---|---|---|---|
| Pose↔XY median offset | < 1 m | 0.22 m | 0.20 m | PASS |
| P(seen \| receiver), θ=60°, T=1.5 s | ≥ 0.75 | 0.885 | 0.913 | PASS |
| P(seen \| other teammate), same | clearly below | 0.646 | 0.631 | PASS (gap ≈ 0.25–0.28) |
| Placebo 180°: P(seen \| receiver) | collapses | 0.136 | 0.132 | PASS |
| Counters ≥ 150 (all games, labels) | ≥ 150 | 296 phases / 857 decisions | | PASS |
| Q2 SkillCorner option fields for pose games | answered | yes | yes | PASS |

Grid (receiver / other), head yaw:

| θ \ T | 1.0 s | 1.5 s | 2.5 s |
|---|---|---|---|
| 45° | .79/.51 · .84/.50 | .80/.56 · .85/.55 | .81/.60 · .86/.58 |
| 60° | .88/.60 · .91/.59 | .89/.65 · .91/.63 | .89/.68 · .91/.67 |
| 90° | .96/.77 · .98/.75 | .96/.80 · .98/.78 | .95/.82 · .97/.81 |

(first pair = 1925299, second = 1996435). Torso-only gives similar numbers (0.897 / 0.906 at 60°/1.5 s).

**Caveats for Phase 2 (skeptic):**
- The window includes the release frame itself, where a passer almost has to face the target. Re-run excluding the last 0.3 s before release; the gap must survive.
- "Other teammate" P(seen) ≈ 0.63 is high: with a ±60° cone + memory, most nearby options count as seen. Vision Loss will only bite on options behind/beside the carrier. Check that this is football-plausible, not a cone that's too wide.
- Not yet Monte Carlo over joint error.

**G0 verdict: PASS**, with one structural risk: only 67 counter decision points have pose (see DECISIONS.md D-001).

## Phase 1: data layer (16 Nov 2026), `scripts/reproduce.py` + `scripts/check_data_layer.py`

| Check | Result | Status |
|---|---|---|
| `reproduce.py --quick` (1925299 incl. pose) | 50 s (target < 5 min) | PASS |
| `reproduce.py --full` (20 games + 2 pose) | 212 s | PASS |
| Event start vs carrier tracking XY (after D-004) | median 0.06 m, p95 0.64 m | PASS |
| Pose sync, 10 fps aligned rows within 1 m (1925299) | 96.5%, median 0.23 m | PASS |
| Synthetic tests (facing angles, MC spread, frame mapping, sync flag, kinematics, counters, event rotation) | 24 passed | PASS |

Pose coverage at release (synced head yaw), 1925299 / 1996435: all 856/1054 · 865/1005; open play 790/971 · 803/928; label counters 27/34 · 24/31; fast attacks 40/52 · 55/68.

## Phase 2: perception (16 Nov 2026), `scripts/validate_vision.py` → `reports/vision_validation.json`
Both pose games, head yaw, 200 MC draws with persistent noise (D-006), ±60° / 1.5 s / 3 m (D-007). Passes with a target and window pose coverage ≥ 50%: n = 1,001 (971 open play, 73 counter).
AUC = within-pass probability that the receiver's P(seen) exceeds a random other teammate's.

| Check (open play unless stated) | Receiver | Other | AUC | Status |
|---|---|---|---|---|
| 1. Receiver sanity (default) | 0.887 | 0.643 | 0.684 | PASS (≥ 0.75, clearly above) |
| 1b. Excluding last 0.3 s before release | 0.847 | 0.625 | 0.663 | PASS, not just the kicking motion |
| 1c. Receivers behind/beside (> 90° from attack dir.), all passes, n = 555 | 0.868 | 0.598 | 0.702 | PASS (hard case) |
| 1d. Counters (n = 73) | 0.834 | 0.470 | 0.759 | PASS |
| 1e. Torso instead of head | 0.912 | 0.628 | 0.668 | head ≈ torso |
| 2. Placebo 180° | 0.162 | 0.406 | 0.293 | PASS (collapses, inverts) |
| 2b. Placebo + exclude last 0.3 s | 0.153 | 0.367 | 0.311 | PASS |

Grid, open play (receiver / other, AUC):

| θ \ T | 1.0 s | 1.5 s | 2.5 s |
|---|---|---|---|
| 45° | .79/.51, .713 | .80/.56, .677 | .81/.60, .643 |
| 60° | .88/.60, .717 | .89/.64, .684 | .90/.68, .660 |
| 90° | .96/.76, .699 | .97/.79, .669 | .97/.82, .644 |

P(seen) by option bearing from the attacking direction (all decisions): 0–45° 0.76, 45–90° 0.72, 90–135° 0.63, 135–180° 0.49. This is football-plausible: what is behind is seen least.

**3. Construct validity (scanning): INCONCLUSIVE.** Scans = head turns > 40° within 0.5 s (mean 2.2 per 5 s before receiving).
- Player level (40 players ≥ 15 decisions): scans vs P(seen behind) ρ = −0.36, but this is confounded by position (DMs scan most, 2.8 per 5 s, and face play; full-backs scan least, 1.5, and see behind easily from the touchline).
- Within position, decision level: scans before *receiving* vs P(seen behind) ρ = 0.03 (none); scans in the *release window* ρ = 0.13 (weak, partly mechanical).
- Pre-reception scanning does not show up in a 1.5 s pre-release memory. Proper test deferred to Phase 5 (scanning vs Vision Loss, controlled for position).

**Human check:** 30 random decisions rendered to `reports/vision_frames/` (index.csv has columns for reviewer verdicts). Spot-check by Claude of 2 frames: plausible (cone, memory-seen receiver behind-left). **Awaiting human review.**

**Phase 2 exit: PASS** (receiver sanity + placebo pass; defaults logged in D-007).
