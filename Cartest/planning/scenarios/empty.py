"""Empty straight-road scenario."""

from Cartest.core.reference_path import StraightReference

COST_NAME = "default_lyapunov"
COST_PARAMS = {"omega_s": 1.0, "omega_d": 4.0, "alpha": 0.0}

SCENARIO = {
    "ref_path":      StraightReference(),
    "road":          {"lane_hw": 4.0},
    "obstacles":     [],
    "safety":        {"obs_safe_dist": 0.1, "a_brake": 8.0,
                      "v_min": 2.0, "v_max": 35.0,
                      "acc_max": 5.0, "jerk_max": 2.0},
    "behavior":      {"v_target": 18.0},
    "cost":          {"name": COST_NAME, "params": COST_PARAMS},
    "ego":           {"s": 0.0, "s_dot": 12.0, "s_ddot": 0.0,
                      "d":  0.0, "d_dot":  0.0, "d_ddot": 0.0,
                      "psi": 0.0},
}
