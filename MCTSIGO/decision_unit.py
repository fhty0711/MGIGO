"""
DecisionUnit, MacroAction, and ProblemSpec — 离散决策空间的模板化描述.

与 Constran 的 ``ConstraintSpec`` 对标：用户只需定义"有哪些离散决策"和"每个
决策有哪些合法取值"，GuideTreeBuilder 自动建索引树。

Quick Start
-----------
>>> from MCTSIGO.decision_unit import DecisionUnit, ProblemSpec
>>>
>>> units = [
...     DecisionUnit(name="Gen0", choices=["ON", "OFF"]),
...     DecisionUnit(name="Gen1", choices=["ON", "OFF"]),
...     DecisionUnit(name="Storage0", choices=["CHARGE", "DISCHARGE", "IDLE"]),
... ]
>>>
>>> problem = ProblemSpec(
...     decision_units=units,
...     cost_fn=my_constran_cost,
...     horizon=24,
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Union)


# ===========================================================================
# 1. DecisionUnit — 单个离散决策维度
# ===========================================================================

@dataclass
class DecisionUnit:
    """
    问题中的一个离散决策维度.

    每个 DecisionUnit 对应索引树的一层。该层的分支数 = len(choices)。

    Parameters
    ----------
    name : str
        决策变量的名称，如 "Gen3", "LaneAtIntersectionX".
    choices : List[str]
        该决策变量的合法取值列表.
        例: ["ON", "OFF"], ["LEFT", "STRAIGHT", "RIGHT"], ["CHARGE", "DISCHARGE", "IDLE"].
    macro_encoder : Callable or None
        将 choice 映射为时域约束的编码器（可选）.
        例: "ON" → ON_interval(start=0, end=24).
        若为 None，choice 直接作为拓扑标记传给 cost_fn 的 ctx.
    description : str
        可选的人类可读描述.
    """
    name: str
    choices: List[str]
    macro_encoder: Optional[Callable[[str], Any]] = None
    description: str = ""

    def __post_init__(self):
        if len(self.choices) < 2:
            raise ValueError(
                f"DecisionUnit '{self.name}' must have at least 2 choices, "
                f"got {len(self.choices)}"
            )
        if len(set(self.choices)) != len(self.choices):
            raise ValueError(
                f"DecisionUnit '{self.name}' has duplicate choices"
            )

    @property
    def n_choices(self) -> int:
        return len(self.choices)

    def encode(self, choice: str) -> Any:
        """将 choice 字符串编码为拓扑标记."""
        if choice not in self.choices:
            raise ValueError(
                f"Invalid choice '{choice}' for '{self.name}'. "
                f"Valid: {self.choices}"
            )
        if self.macro_encoder is not None:
            return self.macro_encoder(choice)
        return choice


# ===========================================================================
# 2. MacroAction — 时域宏决策（MPC 场景专用）
# ===========================================================================

@dataclass
class MacroAction:
    """
    时域内的一个宏决策——将多个时步的离散状态打包为一个决策单元.

    例:
      UC 启停区间:
        MacroAction(
            name="Gen3_commitment",
            unit="Gen3",
            event_type="ON_INTERVAL",
            feasible_times=[0, 1, ..., 23],  # 可选的开机时步
        )
      驾驶变道:
        MacroAction(
            name="LaneChange_t2_t4",
            unit="EgoVehicle",
            event_type="LANE_CHANGE",
            feasible_times=[2, 3, 4],
        )

    Parameters
    ----------
    name : str
    unit : str
        所属的决策单元标识.
    event_type : str
        变更类型: "ON_INTERVAL", "OFF_INTERVAL", "LANE_CHANGE", "TURN", "YIELD", etc.
    feasible_times : List[int]
        该事件可以发生的时步列表.
    extra_params : dict
        额外参数 (如 target_lane, direction).
    """
    name: str
    unit: str
    event_type: str
    feasible_times: List[int] = field(default_factory=list)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_decision_unit(self) -> DecisionUnit:
        """将 MacroAction 转化为 DecisionUnit.

        每个 feasible_time 成为一个 choice，加上 "NONE"（不执行该事件）.
        """
        choices = [f"{self.event_type}_at_{t}" for t in self.feasible_times]
        choices.append("NONE")  # 不执行该事件

        def encoder(choice: str) -> Dict[str, Any]:
            if choice == "NONE":
                return {"event": self.event_type, "active": False}
            t = int(choice.split("_at_")[-1])
            return {
                "event": self.event_type,
                "active": True,
                "time": t,
                "unit": self.unit,
                **self.extra_params,
            }

        return DecisionUnit(
            name=self.name,
            choices=choices,
            macro_encoder=encoder,
            description=f"MacroAction({self.event_type}) for {self.unit}",
        )


# ===========================================================================
# 3. ProblemSpec — 完整的问题描述
# ===========================================================================

@dataclass
class ProblemSpec:
    """
    组合优化问题的完整描述.

    对标 Constran 中用户提供的 objective + constraints 组合：
    - decision_units 定义离散拓扑空间
    - cost_fn 是 Constran 构建的 cost 函数
    - context 是传递给 cost_fn 的静态上下文（需求、地图、他车信息等）

    Parameters
    ----------
    decision_units : List[DecisionUnit]
        离散决策变量列表，每个对应索引树的一层.
    cost_fn : Callable
        Constran 构建的 cost 函数，签名: (strategy, ctx) -> float.
        strategy: 一个 Dict[str, str]，将每个 DecisionUnit.name 映射到 choice.
    context : dict or None
        传递给 cost_fn 的静态上下文.
    horizon : int or None
        MPC 预测时域长度. None 表示一次性优化.
    name : str
        问题名称，用于日志和调试.
    """
    decision_units: List[DecisionUnit]
    cost_fn: Optional[Callable[..., float]] = None
    context: Optional[Dict[str, Any]] = None
    horizon: Optional[int] = None
    name: str = ""

    def __post_init__(self):
        # 验证 decision_units 名字唯一
        names = [u.name for u in self.decision_units]
        if len(names) != len(set(names)):
            seen = set()
            dupes = [n for n in names if n in seen or seen.add(n)]
            raise ValueError(
                f"Duplicate DecisionUnit names: {dupes}"
            )

    @property
    def n_decision_units(self) -> int:
        return len(self.decision_units)

    @property
    def n_paths(self) -> int:
        """所有离散组合的总数（宏决策压缩前）."""
        n = 1
        for u in self.decision_units:
            n *= u.n_choices
        return n

    @property
    def n_branches(self) -> int:
        """所有层的分支数之和（用于 auto-budget 启发式）."""
        return sum(u.n_choices for u in self.decision_units)
