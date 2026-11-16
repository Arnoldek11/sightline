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
