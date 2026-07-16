"""Two-agent basic B-spline + RNE scenario."""

from Cartest.core.reference_path import StraightReference


COST_NAME = "game_2a_basic"


SCENARIO = {
    "type": "multi_agent_game",
    "ref_path": StraightReference(),
    "road": {"lane_hw": 4.0, "lane_bounds_d": (-4.0, 4.0)},
    "agents": [
        {"name": "agent0", "role": "ego", "s": 0.0, "s_dot": 12.0, "s_ddot": 0.0,
         "d": -3.0, "d_dot": 0.0, "d_ddot": 0.0, "psi": 0.0, "v_target": 18.0},
        {"name": "agent1", "role": "peer", "s": 0.0, "s_dot": 12.0, "s_ddot": 0.0,
         "d": 3.0, "d_dot": 0.0, "d_ddot": 0.0, "psi": 0.0, "v_target": 18.0},
    ],
    "game": {
        "solver": "rne_blocks",
        "block_layout": ("agent0_s", "agent0_d", "agent1_s", "agent1_d"),
        "block_to_agent": (0, 0, 1, 1),
        "K": 3,
        "B": 64,
        "B0": 20,
        "T": 200,
        "dt": 0.15,
        "T_0": 100,
        "M_inner": 30,
    },
    "safety": {"safe_dist": 3.0, "v_min": 2.0, "v_max": 35.0,
               "acc_max": 5.0, "jerk_max": 2.0},
    "behavior": {"target_d": 0.0, "speed_factors": (1.15, 1.0, 0.85)},
    "cost": {"name": COST_NAME, "params": {"omega_s": 1.0, "omega_d": 4.0}},
}
