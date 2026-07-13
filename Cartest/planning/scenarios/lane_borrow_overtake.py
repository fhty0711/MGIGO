"""Lane-borrow overtake scenario: blocker at s=80, ego borrows adjacent lane."""

from Cartest.core.reference_path import StraightReference

COST_NAME = "lane_borrow_overtake"
COST_PARAMS = {
    "omega_s": 1.0,
    "omega_d": 4.0,
    "alpha": 0.0,
    "start_lane_d": 0.0,
    "borrow_lane_d": 3.7,
    "return_lane_d": 0.0,
    "blocker_s": 80.0,
    "approach_distance": 22.0,
    "return_distance": 25.0,
    "transition_width": 8.0,
    "acc_weight": 10.0,
    "jerk_weight": 1.0,
}

SCENARIO = {
    "ref_path":      StraightReference(),
    "road":          {"lane_hw": 4.0, "lane_bounds_d": (-4.0, 4.0)},
    "obstacles":     {"static": [{"x": 80.0, "y": 0.0, "r": 2.0}],
                      "dynamic": []},
    "safety":        {"obs_safe_dist": 0.1, "a_brake": 8.0,
                      "v_min": 2.0, "v_max": 35.0,
                      "acc_max": 5.0, "jerk_max": 2.0},
    "behavior":      {"v_target": 18.0},
    "cost":          {"name": COST_NAME, "params": COST_PARAMS},
    "ego":           {"s": 0.0, "s_dot": 12.0, "s_ddot": 0.0,
                      "d":  0.0, "d_dot":  0.0, "d_ddot": 0.0,
                      "psi": 0.0},
}
