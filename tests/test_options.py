import numpy as np

from sightline.options.geometry import (
    Snapshot,
    carry_endpoints,
    pass_features,
    state_features,
)


def snap(opps, mates=((0, 0),), outfield=None):
    opps = np.array(opps, float).reshape(-1, 2)
    mates = np.array(mates, float).reshape(-1, 2)
    of = np.ones(len(opps), bool) if outfield is None else np.array(outfield)
    return Snapshot(np.arange(len(mates)), mates, np.arange(len(opps)) + 100, opps,
                    np.zeros_like(opps), of, 105.0, 68.0)


def test_state_features_geometry():
    s = snap([(40, 0), (40, 30), (0, 0)])
    f = state_features(np.array([[32.5, 0.0], [32.5, 20.0]]), s)
    assert f.dist_goal.tolist() == [20.0, np.hypot(20, 20)]
    assert f.goal_angle[0] > f.goal_angle[1] > 0
    assert f.n_def_goal_side.tolist() == [2, 2]
    assert f.n_block.tolist() == [1, 0]  # (40,0) blocks the central shot only
    assert f.in_box.tolist() == [0, 0]


def test_goalkeeper_not_counted_as_blocker_or_goal_side():
    s = snap([(50, 0)], outfield=[False])
    f = state_features(np.array([[40.0, 0.0]]), s)
    assert f.n_block[0] == 0 and f.n_def_goal_side[0] == 0 and f.nearest_def[0] == 10


def test_pass_margin_sign():
    open_lane = pass_features(np.zeros(2), np.array([[20.0, 0.0]]), snap([(10, 25)]))
    cut_lane = pass_features(np.zeros(2), np.array([[20.0, 0.0]]), snap([(10, 0.5)]))
    assert open_lane.min_margin[0] > 0 > cut_lane.min_margin[0]
    assert cut_lane.n_lane[0] == 1 and open_lane.n_lane[0] == 0
    assert open_lane.length[0] == 20 and open_lane.forward[0] == 20


def test_carry_endpoints_clipped():
    e = carry_endpoints(np.array([50.0, 30.0]))
    assert e.shape == (16, 2)
    assert e[:, 0].max() <= 52.5 and e[:, 1].max() <= 34
