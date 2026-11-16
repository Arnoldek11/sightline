import pandas as pd

from sightline.io.events import decision_points


def test_event_coords_rotated_back_to_pitch_frame():
    ev = pd.DataFrame({
        "event_id": ["a", "b"], "match_id": 1, "period": 1, "frame_start": [10, 20],
        "frame_end": [15, 25], "player_id": [1, 2], "team_id": [100, 200],
        "player_position": "CM", "start_type": ["recovery", "pass_reception"],
        "end_type": "pass", "player_targeted_id": [3, None], "pass_outcome": "successful",
        "x_start": [10.0, 10.0], "y_start": [5.0, 5.0], "x_end": [12.0, 12.0],
        "y_end": [-3.0, -3.0], "phase_index": 0, "team_in_possession_phase_type": "create",
        "first_player_possession_in_team_possession": [True, False],
        "game_interruption_before": [None, "throw_in_for"],
        "attacking_side": ["left_to_right", "right_to_left"],
        "event_type": "player_possession", "lead_to_goal": False, "carry": [True, False],
        "possession_epv_for_start": 0.01,
    })
    d = decision_points(ev).set_index("event_id")
    assert (d.loc["a", ["x_start", "y_start"]] == [10.0, 5.0]).all()
    assert (d.loc["b", ["x_start", "y_start", "y_end"]] == [-10.0, -5.0, 3.0]).all()
    assert d.loc["b", "x_start_att"] == 10.0
    assert bool(d.loc["a", "is_regain"]) and not bool(d.loc["b", "is_regain"])
    assert bool(d.loc["a", "is_open_play"]) and not bool(d.loc["b", "is_open_play"])
