import numpy as np
import pandas as pd
import pytest
from conftest import skeleton_row

from sightline import pose
from sightline.io.pose import records_to_table
from sightline.sync import align_to_tracking, pose_to_tracking_frame, tracking_to_pose_frame


@pytest.mark.parametrize("yaw", np.radians([-179, -135, -90, -30, 0, 45, 90, 170, 180]))
def test_head_and_torso_recover_known_angle(yaw):
    df = pd.DataFrame([skeleton_row(yaw)])
    f = pose.facing(df)
    assert abs(pose.wrap(f.head_yaw[0] - yaw)) < 1e-9
    assert abs(pose.wrap(f.torso_yaw[0] - yaw)) < 1e-9


def test_head_and_torso_independent():
    df = pd.DataFrame([skeleton_row(np.radians(60), torso_yaw=0.0)])
    f = pose.facing(df)
    assert np.degrees(f.head_yaw[0]) == pytest.approx(60)
    assert np.degrees(f.torso_yaw[0]) == pytest.approx(0, abs=1e-9)


def test_gating_on_error_and_missing_joint():
    rows = [skeleton_row(0.0, err=30.0), skeleton_row(0.0)]
    rows[1]["nose_x"] = np.nan
    rows[1]["nose_err"] = np.nan
    f = pose.facing(pd.DataFrame(rows))
    assert f.head_yaw.isna().all()
    assert f.torso_yaw.isna().iloc[0] and f.torso_yaw.notna().iloc[1]


def test_collapsed_shoulders_gated():
    r = skeleton_row(0.0)
    r["rShoulder_x"], r["rShoulder_y"] = r["lShoulder_x"], r["lShoulder_y"] - 0.05
    assert pose.facing(pd.DataFrame([r])).torso_yaw.isna().all()


def test_monte_carlo_spread_grows_with_error():
    df = pd.DataFrame([skeleton_row(1.0, err=2.0), skeleton_row(1.0, err=12.0)])
    d = pose.sample_yaw(df, "head", n_draws=4000, rng=np.random.default_rng(1))
    sd = pose.circ_std(d)
    assert sd[0] < sd[1]
    mean = np.arctan2(np.nanmean(np.sin(d), 1), np.nanmean(np.cos(d), 1))
    assert np.all(np.abs(pose.wrap(mean - 1.0)) < 0.05)


def test_frame_mapping_roundtrip():
    t = np.arange(0, 1000)
    f = tracking_to_pose_frame(t)
    assert np.array_equal(pose_to_tracking_frame(f), t)
    assert np.array_equal(f[::2], 5 * t[::2] // 2)  # even t: exact 2.5 t
    unmatched = np.setdiff1d(np.arange(f.max()), f)
    assert np.all(pose_to_tracking_frame(unmatched) == -1)
    assert len(f) / (f.max() + 1) == pytest.approx(0.4, abs=0.01)


def test_align_and_sync_flag():
    rows = []
    for pf in range(0, 26):
        r = skeleton_row(0.0, cx=1.0, cy=2.0)
        r |= {"pose_frame": pf, "period": 1, "player_id": 7}
        rows.append(r)
    p = pd.DataFrame(rows)
    tr = pd.DataFrame({"frame": range(11), "player_id": 7, "x": 1.0, "y": 2.0})
    tr.loc[tr.frame == 4, "x"] = 3.5  # 2.5 m off -> not synced
    a = align_to_tracking(p, tr)
    assert sorted(a.frame) == list(range(11))
    assert a.set_index("frame").sync_ok.to_dict() == {i: i != 4 for i in range(11)}


def test_records_to_table_skips_null_joints():
    recs = [{"frame": 5, "period": 1, "player_data": [
        {"player_id": 1, "joints": None},
        {"player_id": 2, "joints": {"nose": {"xyz": [1, 2, 1.7], "p90_mae_cm": 7.0}}},
    ]}]
    df = records_to_table(recs)
    assert list(df.player_id) == [2]
    assert df.nose_x.iloc[0] == 1 and np.isnan(df.lEar_x.iloc[0])


def test_shared_noise_is_persistent_across_rows():
    df = pd.DataFrame([skeleton_row(0.5, err=10.0)] * 3)
    z = np.random.default_rng(3).standard_normal((100, 3, 2))
    d = pose.sample_yaw(df, "head", noise=z)
    assert d.shape == (3, 100)
    assert np.allclose(d[0], d[1], equal_nan=True) and np.allclose(d[1], d[2], equal_nan=True)
    assert pose.circ_std(d[0]) > 0.05
