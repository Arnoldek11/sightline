import numpy as np
import pandas as pd
import pytest

from sightline.counters import CounterConfig, detect_counters, tag_decisions
from sightline.io.tracking import add_attack_dir, add_kinematics


def test_attack_dir_flips_by_period_and_team(meta):
    assert meta.attack_dir(100, 1) == 1 and meta.attack_dir(200, 1) == -1
    assert meta.attack_dir(100, 2) == -1 and meta.attack_dir(200, 2) == 1
    df = pd.DataFrame({"team_id": [100, 200, 100], "period": [1, 1, 2]})
    assert list(add_attack_dir(df, meta).attack_dir) == [1, -1, -1]


def test_kinematics_constant_velocity_and_gap():
    f = np.r_[np.arange(0, 30), np.arange(40, 45)]  # gap -> separate run
    df = pd.DataFrame({"frame": f, "player_id": 1, "x": 0.5 * f, "y": -0.2 * f})
    k = add_kinematics(df)
    assert np.allclose(k.vx, 5.0) and np.allclose(k.vy, -2.0)
    assert np.allclose(k.speed, np.hypot(5, 2))
    assert np.allclose(k.ax.iloc[:30], 0, atol=1e-9)


def _scenario(progress_m: float, n_goal_side: int, shot: bool = False):
    """Team 100 (attacks +x in period 1) regains at frame 100 at x=-20."""
    frames = np.arange(100, 260)
    bx = -20 + np.clip((frames - 100) / 80, 0, 1) * progress_m
    ball = pd.DataFrame({"frame": frames, "ball_x": bx, "ball_y": 0.0})
    opp_x = [-30.0] * (4 - n_goal_side) + [10.0] * n_goal_side  # 4 outfield + GK
    pl = [{"frame": 100, "player_id": 20 + i, "team_id": 200, "x": x}
          for i, x in enumerate(opp_x)]
    pl.append({"frame": 100, "player_id": 11, "team_id": 200, "x": 50.0})  # GK, ignored
    players = pd.DataFrame(pl)
    dec = pd.DataFrame({
        "event_id": ["a", "b", "c", "d"], "period": 1, "team_id": [200, 100, 100, 200],
        "frame_start": [50, 100, 140, 250], "frame_end": [95, 130, 170, 255],
        "is_regain": [False, True, False, True],
        "end_type": ["pass", "pass", "shot" if shot else "pass", "pass"],
        "is_counter_label": [False, False, True, False],
    })
    return dec, ball, players


def test_rule_counter_by_progress(meta):
    dec, ball, pl = _scenario(progress_m=30, n_goal_side=4)
    c = detect_counters(dec, ball, pl, meta).set_index("regain_event_id")
    r = c.loc["b"]
    assert r.progress_m == pytest.approx(30) and bool(r.rule_counter)
    assert r.defenders_goal_side == 4  # GK excluded
    assert r.frame_seq_end == 249 and bool(r.label_counter)


def test_rule_counter_by_few_defenders(meta):
    dec, ball, pl = _scenario(progress_m=20, n_goal_side=3)  # reaches x=0? no: -20+20=0
    c = detect_counters(dec, ball, pl, meta, CounterConfig(min_progress_m=25))
    assert not bool(c.set_index("regain_event_id").loc["b"].rule_counter)  # not final third
    dec, ball, pl = _scenario(progress_m=40, n_goal_side=3)
    cfg = CounterConfig(min_progress_m=50)
    assert bool(detect_counters(dec, ball, pl, meta, cfg).set_index(
        "regain_event_id").loc["b"].rule_counter)


def test_fast_attack_and_tagging(meta):
    dec, ball, pl = _scenario(progress_m=10, n_goal_side=4, shot=True)
    c = detect_counters(dec, ball, pl, meta)
    r = c.set_index("regain_event_id").loc["b"]
    assert bool(r.fast_attack) and not bool(r.rule_counter)
    assert bool(r.label_counter) and bool(r.counter)  # labelled sequence -> headline counter
    dec.loc[:, "is_counter_label"] = False
    c2 = detect_counters(dec, ball, pl, meta).set_index("regain_event_id")
    assert not bool(c2.loc["b"].counter)  # fast but neither label nor rule
    dec, ball, pl = _scenario(progress_m=30, n_goal_side=4, shot=True)
    dec.loc[:, "is_counter_label"] = False
    c3 = detect_counters(dec, ball, pl, meta).set_index("regain_event_id")
    assert bool(c3.loc["b"].counter)  # rule AND fast
    t = tag_decisions(dec, c)
    assert list(t.regain_event_id.fillna("-")) == ["-", "b", "b", "d"]
