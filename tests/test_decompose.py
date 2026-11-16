import numpy as np
import pandas as pd
import pytest

from sightline.decompose import shrink, split_losses


def test_identity_and_known_split():
    v = np.array([0.10, 0.05, 0.02, 0.01])
    seen = np.array([[False, True], [True, True], [True, False], [False, False]])
    r = split_losses(v, seen, chosen=3)
    # draw 0: best seen 0.05; draw 1: best seen 0.10
    assert r["vision_loss"] == pytest.approx((0.05 + 0.0) / 2)
    assert r["decision_loss"] == pytest.approx((0.04 + 0.09) / 2)
    assert r["vision_loss"] + r["decision_loss"] == pytest.approx(r["total_loss"])
    assert r["p_best_seen"] == 0.5


def test_chosen_always_perceived_and_losses_nonnegative():
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = rng.normal(size=8)
        seen = rng.random((8, 30)) < 0.4
        c = int(rng.integers(8))
        r = split_losses(v, seen, c, rng=rng)
        assert r["vision_loss"] >= 0 and r["decision_loss"] >= -1e-12
        assert r["vision_loss"] + r["decision_loss"] == pytest.approx(r["total_loss"])
        assert r["vision_loss_null"] >= 0


def test_best_chosen_means_zero_loss():
    r = split_losses(np.array([0.3, 0.1]), np.zeros((2, 5), bool), chosen=0)
    assert r["total_loss"] == r["vision_loss"] == r["decision_loss"] == 0


def test_null_equals_actual_when_everything_seen():
    v = np.array([0.3, 0.2, 0.1])
    r = split_losses(v, np.ones((3, 10), bool), 2, rng=np.random.default_rng(1))
    assert r["vision_loss"] == 0 and r["vision_loss_null"] == 0


def test_shrinkage_pulls_noisy_players_more():
    m = pd.Series([0.0, 1.0, 0.9, 0.2, 0.8])
    se = pd.Series([0.05, 0.05, 0.3, 0.05, 0.05])
    s = shrink(m, se, pd.Series(["a"] * 5))
    assert 0 < s.weight[2] < s.weight[0] < 1
    assert s.weight[2] < s.weight[0]
    assert abs(s.post_mean[2] - s.group_mean[2]) < abs(m[2] - s.group_mean[2])
