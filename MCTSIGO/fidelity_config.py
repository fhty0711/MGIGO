"""
FidelityConfig — 三层保真度的参数和预算配置.

对标 Constran 的 ``ConstraintSpec`` 和参数选择（ConstranUser_README.md §4）：
用户声明三个保真度等级的计算参数，auto_fidelity_config 根据问题规模
自动给出合理的预算分配。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ===========================================================================
# 1. FidelityConfig — 三层保真度参数
# ===========================================================================

@dataclass
class FidelityConfig:
    """
    三层保真度的评估参数和预算.

    Parameters
    ----------
    none_enabled : bool
        是否启用 none 保真度（Constran 快评，无 IGO）.
    light_enabled : bool
        是否启用 light 保真度（轻量 IGO）.
    full_enabled : bool
        是否启用 full 保真度（精评 IGO）.

    light_T : int — light IGO 的迭代步数 (默认 100).
    light_B : int — light IGO 的样本数 (默认 30).
    light_MC : int — light IGO 的 MC 样本数 (默认 30).
    light_K : int — light IGO 的 GMM 分量数 (默认 3).

    full_T : int — full IGO 的迭代步数 (默认 300).
    full_B : int — full IGO 的样本数 (默认 200).
    full_MC : int — full IGO 的 MC 样本数 (默认 100).
    full_K : int — full IGO 的 GMM 分量数 (默认 3).

    budget_none : int or None — none 评估的预算（None = 自动）.
    budget_light : int or None — light 评估的预算（None = 自动）.
    budget_full : int or None — full 评估的预算（None = 自动）.

    uct_c : float — UCT 探索系数 (默认 1/√2).

    P.S 实际上在Build 的时候，需要根据树的结构 选择IGO 样本数，因为IGO 的样本数大约需要3-4倍它搜索的决策变量数。
    """
    # 启用开关
    none_enabled: bool = True
    light_enabled: bool = True
    full_enabled: bool = True

    # Light IGO 参数
    light_T: int = 100
    light_B: int = 30
    light_MC: int = 30
    light_K: int = 3

    # Full IGO 参数
    full_T: int = 300
    full_B: int = 200
    full_MC: int = 100
    full_K: int = 3

    # 预算（None = 自动）
    budget_none: Optional[int] = None
    budget_light: Optional[int] = None
    budget_full: Optional[int] = None

    # UCT 参数
    uct_c: float = 0.7071067811865476  # 1/√2

    # Q 计算模式
    q_mode: str = 'elite'  # 'elite' | 'mean'
    elite_fraction_none: float = 0.5   # none 保真度的精英比例
    elite_fraction_light: float = 0.3  # light 保真度的精英比例
    elite_fraction_full: float = 0.2   # full 保真度的精英比例（最严）

    def __post_init__(self):
        if self.light_T < 1 or self.full_T < 1:
            raise ValueError("IGO iteration count must be >= 1")
        if self.light_B < 1 or self.full_B < 1:
            raise ValueError("IGO sample count must be >= 1")
        if self.uct_c <= 0:
            raise ValueError("UCT exploration coefficient must be > 0")


# ===========================================================================
# 2. auto_fidelity_config — 启发式预算分配
# ===========================================================================

def auto_fidelity_config(
    n_paths: int,
    n_branches: int,
    budget_ratio: tuple = (5.0, 1.0, 0.4),
    **overrides,
) -> FidelityConfig:
    """
    根据问题规模自动配置保真度参数.

    启发式:
      budget_none ≈ n_branches * budget_ratio[0]
      budget_light ≈ n_branches * budget_ratio[1]
      budget_full ≈ min(n_paths, max(3, n_branches * budget_ratio[2]))

    Parameters
    ----------
    n_paths : int
        索引树的总路径数.
    n_branches : int
        所有层的分支数之和.
    budget_ratio : tuple (none_ratio, light_ratio, full_ratio)
        预算倍数（相对于 n_branches）.
    **overrides
        覆盖 FidelityConfig 的任意字段.

    Returns
    -------
    FidelityConfig
    """
    import math

    budget_none = max(n_branches, int(n_branches * budget_ratio[0]))
    budget_light = max(n_branches, int(n_branches * budget_ratio[1]))
    budget_full = max(2, min(n_paths, int(n_branches * budget_ratio[2])))

    config = FidelityConfig(
        budget_none=budget_none,
        budget_light=budget_light,
        budget_full=budget_full,
    )

    # 应用 overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
