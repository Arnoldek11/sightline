import numpy as np
import pytest

from sightline.vision import DecisionGeometry, count_scans, p_seen

L, D = 26, 50


def geom(yaw_by_lag, opts_xy, moving=None):
    """Carrier at origin. opts_xy: (M,2) static positions; moving: {m: (L,2)} trajectories."""
    yaw = np.array(yaw_by_lag, float)
    oxy = np.repeat(np.asarray(opts_xy, float)[None], L, axis=0)
    for m, traj in (moving or {}).items():
        oxy[:, m] = traj
    return DecisionGeometry("e", 1, np.arange(len(opts_xy)), np.zeros((L, 2)), oxy,
                            np.repeat(yaw[:, None], D, axis=1), yaw)


AHEAD, LEFT, BEHIND = (10, 0), (0, 10), (-10, 0)


def test_cone_membership():
    g = geom(np.zeros(L), [AHEAD, LEFT, BEHIND])
    p, cov = p_seen(g, cone_deg=60, memory_s=1.5)
    assert list(p) == [1, 0, 0] and cov == 1.0
    p90, _ = p_seen(g, cone_deg=91, memory_s=1.5)
    assert list(p90) == [1, 1, 0]


def test_placebo_flips_front_and_back():
    g = geom(np.zeros(L), [AHEAD, LEFT, BEHIND])
    p, _ = p_seen(g, cone_deg=60, memory_s=1.5, placebo=True)
    assert list(p) == [0, 0, 1]


def test_memory_window_and_min_lag():
    yaw = np.zeros(L)
    yaw[:12] = np.pi  # looked at BEHIND only during the last 1.1 s; AHEAD from 1.2 s back
    g = geom(yaw, [AHEAD, BEHIND])
    assert list(p_seen(g, 60, memory_s=1.0)[0]) == [0, 1]
    assert list(p_seen(g, 60, memory_s=1.5)[0]) == [1, 1]
    assert list(p_seen(g, 60, memory_s=1.5, min_lag_s=1.2)[0]) == [1, 0]


def test_stale_view_of_moving_option_does_not_count():
    yaw = np.full(L, np.pi / 2)
    yaw[10:] = 0.0  # looked ahead 1.0 s+ ago, then left
    traj = np.c_[np.linspace(10, 22, L)[::-1] * 0 + 10, np.zeros(L)]
    traj[:, 0] = 10 + 6 * (np.arange(L) == 0)  # option jumped 6 m at release (lag 0)
    g = geom(yaw, [AHEAD], moving={0: traj})
    assert p_seen(g, 60, memory_s=2.5)[0][0] == 0
    assert p_seen(g, 60, memory_s=2.5, max_move_m=10)[0][0] == 1


def test_missing_pose_means_not_seen_and_lowers_coverage():
    yaw = np.full(L, np.nan)
    yaw[20] = 0.0
    g = geom(yaw, [AHEAD])
    p, cov = p_seen(g, 60, memory_s=1.5)
    assert p[0] == 0 and cov == 0.0
    p, cov = p_seen(g, 60, memory_s=2.5)
    assert p[0] == 1 and cov == pytest.approx(1 / 26)


def test_probabilistic_draws():
    g = geom(np.zeros(L), [(10, 10)])  # bearing 45 deg
    rng = np.random.default_rng(0)
    g.yaw_draws = rng.normal(0, np.radians(20), (L, 2000))
    g.yaw_draws[1:] = np.nan  # one sample only -> P = P(|45 - yaw| <= 60)
    p = p_seen(g, 60, memory_s=1.5)[0][0]
    assert p == pytest.approx(0.773, abs=0.03)  # Phi(15/20)


def test_count_scans():
    fps = 25
    y = np.zeros(10 * fps)
    y[50:] = np.radians(90)  # one sharp turn
    y[150:] = np.radians(-10)  # a second turn back
    assert count_scans(y, fps) == 2
    assert count_scans(np.linspace(0, np.radians(30), 250), fps) == 0  # slow drift
    y2 = y.copy()
    y2[100:110] = np.nan  # gap does not create scans
    assert count_scans(y2, fps) == 2
    wrapped = np.angle(np.exp(1j * np.r_[np.full(50, 3.0), np.full(50, 3.0 + 0.3)]))
    assert count_scans(wrapped, fps) == 0  # crossing +/-pi is not a turn
