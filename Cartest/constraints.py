"""Constraint builders for Frenet B-spline MPC.

All kinematics constraints (jerk, acc, speed, lane) operate directly
on Frenet quantities — linear in control points, no Cartesian projection.

Only obstacle checking goes through Frenet→Cartesian mapping.
"""

from __future__ import annotations

import jax.numpy as jnp
from Constraintdealer.Constran import Deterministic

from Cartest.cost import _eval_traj


# ═══════════════════════════════════════════════════════════════════════
# Constraint parameters
# ═══════════════════════════════════════════════════════════════════════

V_MIN, V_MAX = 2.0, 35.0
ACC_MAX = 5.0          # m/s²  longitudinal / lateral
JERK_MAX = 5.0         # m/s³  longitudinal / lateral
LANE_HW = 4.0          # m     half-width
OBS_SAFE_DIST = 2.0    # m     safety margin around obstacle


# ═══════════════════════════════════════════════════════════════════════
# Constraint factory
# ═══════════════════════════════════════════════════════════════════════

def make_constraints(gen):
    """Build constraint list for Frenet B-spline MPC.

    Self-similar nesting: 小priority=内层(被后续σ·m放大, 影响大), 大priority=外层(直接输出)
      P1 (内): obstacle  — 安全约束, 被放大4×, 天然最优先
      P2:     lane      — 车道约束
      P3:     speed     — 速度约束
      P4:     acc       — 加速度约束
      P5 (外): jerk      — 控制输入, 直接输出, 不放大
    安全(obs/lane)自然优先于舒适(jerk/acc) — 靠结构保证, 不靠参数调
    """

    def obs_g(theta, ctx):
        """Obstacle penetration → max. 碰撞检测."""
        s, d, _, _, _, _, _, _ = _eval_traj(theta, ctx, gen)
        x, y = gen.to_cartesian(s, d)
        dx = x[:, None] - ctx["obs_pos"][None, :, 0]
        dy = y[:, None] - ctx["obs_pos"][None, :, 1]
        dist = jnp.sqrt(dx ** 2 + dy ** 2)
        pen = jnp.maximum(0., OBS_SAFE_DIST + ctx["obs_rad"][None, :] - dist)
        return jnp.min(pen, axis=-1)

    def lane_g(theta, ctx):
        """Lane boundary |d| ≤ lane_hw. 直接用 d."""
        _, d, _, _, _, _, _, _ = _eval_traj(theta, ctx, gen)
        return jnp.maximum(0., jnp.abs(d) - ctx["lane_hw"])

    def speed_g(theta, ctx):
        """Speed V_MIN ≤ ḃ ≤ V_MAX. 直接用 s_dot."""
        _, _, s_dot, _, _, _, _, _ = _eval_traj(theta, ctx, gen)
        return jnp.maximum(0., jnp.maximum(V_MIN - s_dot, s_dot - V_MAX))

    def acc_g(theta, ctx):
        """Acceleration |s̈|, |d̈| ≤ ACC_MAX. 直接用 s_ddot, d_ddot."""
        _, _, _, _, s_ddot, d_ddot, _, _ = _eval_traj(theta, ctx, gen)
        am = jnp.sqrt(s_ddot ** 2 + d_ddot ** 2)
        return jnp.maximum(0., am - ACC_MAX)

    def jerk_g(theta, ctx):
        """Jerk |s⃛|, |d⃛| ≤ JERK_MAX. 直接用 s_dddot, d_dddot."""
        _, _, _, _, _, _, s_dddot, d_dddot = _eval_traj(theta, ctx, gen)
        jm = jnp.sqrt(s_dddot ** 2 + d_dddot ** 2)
        return jnp.maximum(0., jm - JERK_MAX)

    return [
        # P1 (最内层): 避障 — baseline=2.0, 安全底线
        Deterministic(obs_g,   mode='hard', priority=1, aggregate='q95',
                      transform='hard'),
        # P2-P5: comfort — mode='soft' → baseline=0
        # 无违规时 Φ=0, σ层透明, obj信号完全恢复
        Deterministic(lane_g,  mode='soft', priority=2, aggregate='q95',
                      transform='soft'),
        Deterministic(speed_g, mode='soft', priority=3, aggregate='max',
                      transform='soft'),
        Deterministic(acc_g,   mode='soft', priority=4, aggregate='max',
                      transform='soft'),
        Deterministic(jerk_g,  mode='soft', priority=5, aggregate='max',
                      transform='soft'),
    ]
