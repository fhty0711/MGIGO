"""Three-agent Trackgame-inspired B-spline + RNE scenario."""

from Cartest.core.reference_path import StraightReference


COST_NAME = "three_agent_track"


SCENARIO = {
    "type": "multi_agent_game",
    "ref_path": StraightReference(),
    "road": {"lane_width": 3.5, "lane_centers_d": (0.0, 3.5),
             "lane_hw": 5.25, "lane_bounds_d": (-1.75, 5.25)},
    "agents": [
        {"name": "ego", "role": "ego", "s": 25.0, "s_dot": 15.0, "s_ddot": 0.0,
         "d": 0.0, "d_dot": 0.0, "d_ddot": 0.0, "psi": 0.0, "v_target": 17.5},
        {"name": "front", "role": "front", "s": 35.0, "s_dot": 10.0, "s_ddot": 0.0,
         "d": 3.5, "d_dot": 0.0, "d_ddot": 0.0, "psi": 0.0, "v_target": 20.0},
        {"name": "rear", "role": "rear", "s": 5.0, "s_dot": 10.0, "s_ddot": 0.0,
         "d": 3.5, "d_dot": 0.0, "d_ddot": 0.0, "psi": 0.0, "v_target": 17.5},
    ],
    "game": {
        "solver": "cartest_batched_rne_blocks",
        "default_steps": 25,
        "execute_index": 3,
        "block_layout": ("ego_s", "ego_d", "front_s", "front_d", "rear_s", "rear_d"),
        "block_to_agent": (0, 0, 1, 1, 2, 2),
        "K": 3,
        "B": 60,
        "B0": 25,
        "T": 300,
        "dt": 0.15,
        "T_0": 300,
        "M_inner": 30,
    },
    "safety": {"vehicle_length": 5.0, "vehicle_width": 2.0, "safe_gap": 3.0,
               "v_min": 2.0, "v_max": 35.0, "acc_max": 5.0, "jerk_max": 2.0},
    "behavior": {"ego_target_d": 3.5, "lower_lane_d": 0.0, "upper_lane_d": 3.5,
                 "speed_factors": (1.15, 1.0, 0.75)},
    "cost": {"name": COST_NAME, "params": {"omega_s": 1.0, "omega_d": 4.0}},
}
