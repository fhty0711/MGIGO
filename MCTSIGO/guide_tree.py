"""
IndexTree + GuideTreeSolver — 引导树闭环求解器.

模板化的离散-连续联合优化求解器。对标 Constran 的 ``build()`` 模式：
用户定义 DecisionUnit 列表 → build_guide_tree_solver() → .solve()。

Quick Start
-----------
>>> from MCTSIGO.decision_unit import DecisionUnit, ProblemSpec
>>> from MCTSIGO.guide_tree import build_guide_tree_solver
>>>
>>> units = [DecisionUnit("Gen0", ["ON","OFF"]), DecisionUnit("Gen1", ["ON","OFF"])]
>>> problem = ProblemSpec(decision_units=units, cost_fn=my_cost)
>>> solver = build_guide_tree_solver(problem)
>>> result = solver.solve()

Algorithm
---------
实现 MCTSforMPC.md §11 的最小闭环:
  1. 从 DecisionUnit 构建静态索引树
  2. 顺序三阶段: none → light → full
  3. Selection: UCT with recursive σ-wrapped Q̃ (§8.2)
  4. Evaluation: IGO solver per fidelity mode
  5. Backprop: incremental mean update per mode
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Union)

from MCTSIGO.decision_unit import DecisionUnit, ProblemSpec
from MCTSIGO.fidelity_config import FidelityConfig, auto_fidelity_config


# ===========================================================================
# 0. σ-saturation function (与 Constran.sigma_k 同构)
# ===========================================================================

def sigma(x: float, k: float = 1.0) -> float:
    """σ_k(x) = k·x / √(1 + (k·x)²). Output ∈ (-1, 1)."""
    kx = k * x
    return kx / math.sqrt(1.0 + kx * kx)


# ===========================================================================
# 1. TreeNode — 索引树节点
# ===========================================================================

class TreeNode:
    """
    索引树的节点。

    每个节点维护三层保真度的独立 Q 统计量:
      Q: Dict[mode, float] — 经验平均 cost
      n: Dict[mode, int]   — 评估次数

    is_leaf=True 表示该节点对应一条完整路径（一个完整离散策略）。
    """

    __slots__ = (
        'name', 'choice', 'parent', 'children',
        'is_leaf', 'strategy', 'Q', 'n',
    )

    def __init__(
        self,
        name: str = "",
        choice: str = "",
        parent: Optional[TreeNode] = None,
        is_leaf: bool = False,
        strategy: Optional[Dict[str, str]] = None,
    ):
        self.name = name          # 对应 DecisionUnit.name（根节点为空）
        self.choice = choice      # 本节点对应的选择（根节点为空）
        self.parent = parent
        self.children: List[TreeNode] = []
        self.is_leaf = is_leaf
        self.strategy = strategy  # 叶节点: 完整离散策略 {unit_name: choice}

        # 三层保真度统计量
        self.Q: Dict[str, float] = {'none': 0.0, 'light': 0.0, 'full': 0.0}
        self.n: Dict[str, int]   = {'none': 0,    'light': 0,    'full': 0}

    # -- visit counts -------------------------------------------------------

    def total_n(self) -> int:
        """该节点的总访问次数（所有保真度之和）."""
        return self.n['none'] + self.n['light'] + self.n['full']

    # -- Q̃ computation (§8.2) ------------------------------------------------

    def compute_Q_tilde(self) -> float:
        """
        递归 σ-包裹 Q̃.

        Q̃ = σ( Q_none + σ( Q_light + σ( Q_full ) ) )

        每层 Q 天然有界（继承自 Constran cost 的 σ-饱和输出 ∈ (-1,1)），
        因此不需要额外 δ 权重。嵌套顺序本身就是优先级：
        full (最内层, 重型) → light (中层) → none (最外层, 轻型).
        """
        return sigma(
            self.Q['none'] + sigma(
                self.Q['light'] + sigma(self.Q['full'])
            )
        )

    def compute_Q_tilde_norm(self) -> float:
        """归一化 Q̃ 到 [0, 1]，用于 UCT."""
        return (self.compute_Q_tilde() + 1.0) / 2.0

    # -- backprop -----------------------------------------------------------

    def backprop(self, mode: str, reward: float) -> None:
        """
        增量均值更新 (§8.2):
          Q_mode ← (n_mode · Q_mode + reward) / (n_mode + 1)
        """
        self.Q[mode] = (self.n[mode] * self.Q[mode] + reward) / (self.n[mode] + 1)
        self.n[mode] += 1

    # -- tree building ------------------------------------------------------

    def add_child(self, child: TreeNode) -> None:
        self.children.append(child)

    # -- representation -----------------------------------------------------

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"Leaf({self.strategy})"
        return (f"Node('{self.name}'='{self.choice}', "
                f"children={len(self.children)}, "
                f"n={{{self.n}}})")


# ===========================================================================
# 2. IndexTree — 静态索引树
# ===========================================================================

class IndexTree:
    """
    静态预建的路径索引树。

    构建后树结构不变——Backprop 只更新 Q 和 n 统计量。

    通过递归遍历需要访问整棵树时，使用 self.all_nodes() 或 self.leaves。
    """

    def __init__(self, root: TreeNode, leaves: List[TreeNode]):
        self.root = root
        self.leaves = leaves
        self._all_nodes: Optional[List[TreeNode]] = None

    @property
    def n_paths(self) -> int:
        return len(self.leaves)

    @property
    def depth(self) -> int:
        """树的最大深度（从根到最远叶子的边数）."""
        return max(self._depth_from_root(leaf) for leaf in self.leaves)

    def _depth_from_root(self, node: TreeNode) -> int:
        d = 0
        current = node
        while current.parent is not None:
            d += 1
            current = current.parent
        return d

    def all_nodes(self) -> List[TreeNode]:
        """返回树中所有节点（惰性缓存）."""
        if self._all_nodes is None:
            nodes = []
            stack = [self.root]
            while stack:
                node = stack.pop()
                nodes.append(node)
                stack.extend(node.children)
            self._all_nodes = nodes
        return self._all_nodes

    def reset_statistics(self) -> None:
        """重置所有节点的 Q 和 n 统计量."""
        for node in self.all_nodes():
            node.Q = {'none': 0.0, 'light': 0.0, 'full': 0.0}
            node.n = {'none': 0,    'light': 0,    'full': 0}

    def describe(self) -> str:
        """返回树的结构摘要."""
        n_nodes = len(self.all_nodes())
        return (
            f"IndexTree(n_paths={self.n_paths}, depth={self.depth}, "
            f"nodes={n_nodes}, leaves={len(self.leaves)})"
        )

    # -- factory methods ----------------------------------------------------

    @staticmethod
    def from_decision_units(
        units: List[DecisionUnit],
        macro_compress: bool = True,
        max_paths: int = 10000,
    ) -> IndexTree:
        """
        从 DecisionUnit 列表构建静态索引树.

        构建过程:
          1. Cartesian 积枚举所有路径: product(choices[0], ..., choices[N-1])
          2. 逐层建树: 每层 = 一个 DecisionUnit
          3. 每个叶节点绑定一条完整路径的策略字典

        Parameters
        ----------
        units : List[DecisionUnit]
            离散决策变量列表.
        macro_compress : bool
            是否通过 macro_encoder 压缩（目前保留接口，压缩逻辑待实现）.
        max_paths : int
            路径数上限，超出时警告.

        Returns
        -------
        IndexTree
        """
        # 1. 枚举所有路径
        all_choices = [u.choices for u in units]
        n_paths = 1
        for c in all_choices:
            n_paths *= len(c)
        if n_paths > max_paths:
            import warnings
            warnings.warn(
                f"Large path count ({n_paths}) exceeds max_paths ({max_paths}). "
                f"Consider blockwise decomposition (MCTSforMPC.md §2.2)."
            )

        # 2. 建树
        root = TreeNode(name="root", choice="")
        leaves: List[TreeNode] = []

        # 逐路径插入
        for choices_tuple in product(*all_choices):
            strategy: Dict[str, str] = {}
            current = root

            for i, choice in enumerate(choices_tuple):
                unit_name = units[i].name
                strategy[unit_name] = choice

                # 查找是否已有对应子节点
                existing = _find_child(current, choice)
                if existing is not None:
                    current = existing
                else:
                    new_node = TreeNode(
                        name=unit_name,
                        choice=choice,
                        parent=current,
                    )
                    current.add_child(new_node)
                    current = new_node

            # 到达叶子 — 绑定完整策略
            current.is_leaf = True
            current.strategy = strategy
            leaves.append(current)

        tree = IndexTree(root, leaves)
        if n_paths <= 50:
            # 对于小树，预先缓存所有节点
            _ = tree.all_nodes()
        return tree

    @staticmethod
    def from_macro_actions(
        actions: List[Any],  # List[MacroAction]
        horizon: int,
    ) -> IndexTree:
        """
        从时域宏决策列表构建索引树（MPC 场景专用）.

        每个 MacroAction 先转为 DecisionUnit，再调用 from_decision_units.
        """
        from MCTSIGO.decision_unit import MacroAction
        units: List[DecisionUnit] = []
        for action in actions:
            if isinstance(action, MacroAction):
                units.append(action.to_decision_unit())
            elif isinstance(action, DecisionUnit):
                units.append(action)
            else:
                raise TypeError(
                    f"Expected MacroAction or DecisionUnit, got {type(action)}"
                )
        return IndexTree.from_decision_units(units)


def _find_child(parent: TreeNode, choice: str) -> Optional[TreeNode]:
    """在 parent 的子节点中查找 choice 匹配的节点."""
    for child in parent.children:
        if child.choice == choice:
            return child
    return None


# ===========================================================================
# 3. GuideTreeResult
# ===========================================================================

@dataclass
class GuideTreeResult:
    """引导树求解器的输出."""
    best_strategy: Optional[Dict[str, str]] = None
    best_cost: float = float('inf')
    best_continuous: Any = None          # 对应的连续参数 x*
    tree: Optional[IndexTree] = None     # 最终树（含 Q 统计量）
    phase_costs: List[float] = field(default_factory=list)
    n_evaluations: Dict[str, int] = field(default_factory=dict)
    final_full_cost: Optional[float] = None  # 最终 full IGO 精搜 cost

    def __repr__(self) -> str:
        return (
            f"GuideTreeResult(best_cost={self.best_cost:.4f}, "
            f"best_strategy={self.best_strategy}, "
            f"phases={self.phase_costs}, "
            f"n_evals={self.n_evaluations})"
        )


# ===========================================================================
# 4. GuideTreeSolver
# ===========================================================================

class GuideTreeSolver:
    """
    引导树 + IGO 闭环求解器.

    实现 MCTSforMPC.md §11 的最小闭环算法.

    Parameters
    ----------
    tree : IndexTree
        静态预建的索引树.
    igo_evaluate : Callable
        IGO 评估函数. 签名: (strategy: Dict, mode: str) -> float
        mode ∈ {'none', 'light', 'full'}.
        mode='none': Constran 快评, 无 IGO.
        mode='light': 轻量 IGO (T=100, B=30).
        mode='full': 精评 IGO (T=300, B=200, MC=100).
    fidelity_config : FidelityConfig
        保真度参数和预算配置.
    """

    def __init__(
        self,
        tree: IndexTree,
        igo_evaluate: Callable[[Dict[str, str], str], float],
        fidelity_config: Optional[FidelityConfig] = None,
    ):
        self.tree = tree
        self.igo_evaluate = igo_evaluate
        self.config = fidelity_config or FidelityConfig()
        self.c = self.config.uct_c  # UCT 探索系数

        # 内部状态
        self.best_strategy: Optional[Dict[str, str]] = None
        self.best_cost: float = float('inf')
        self.best_continuous: Any = None
        self.phase_costs: List[float] = []
        self.n_evaluations: Dict[str, int] = {'none': 0, 'light': 0, 'full': 0}

    # -- main entry point ---------------------------------------------------

    def solve(
        self,
        final_full_refine: bool = True,
    ) -> GuideTreeResult:
        """
        运行最小闭环.

        Parameters
        ----------
        final_full_refine : bool
            是否在最终对 best_s 强制跑一次完整 full IGO 精搜.

        Returns
        -------
        GuideTreeResult
        """
        self.tree.reset_statistics()
        self.best_strategy = None
        self.best_cost = float('inf')
        self.best_continuous = None
        self.phase_costs = []
        self.n_evaluations = {'none': 0, 'light': 0, 'full': 0}

        # 阶段 1: none
        if self.config.none_enabled:
            budget = self._resolve_budget('none')
            self._run_phase('none', budget)
            self.phase_costs.append(self.best_cost)
            self.n_evaluations['none'] = budget

        # 阶段 2: light
        if self.config.light_enabled:
            budget = self._resolve_budget('light')
            # Warm start: 用 none 阶段的 Q 统计量初始化 light
            # Q_light 目前为 0，保持为 0——让 UCT 探索项自然引导
            self._run_phase('light', budget)
            self.phase_costs.append(self.best_cost)
            self.n_evaluations['light'] = budget

        # 阶段 3: full
        if self.config.full_enabled:
            budget = self._resolve_budget('full')
            self._run_phase('full', budget)
            self.phase_costs.append(self.best_cost)
            self.n_evaluations['full'] = budget

        # 最终精搜
        final_full_cost = None
        if final_full_refine and self.best_strategy is not None:
            final_full_cost = self.igo_evaluate(self.best_strategy, 'full')
            if final_full_cost < self.best_cost:
                self.best_cost = final_full_cost

        return GuideTreeResult(
            best_strategy=self.best_strategy,
            best_cost=self.best_cost,
            best_continuous=self.best_continuous,
            tree=self.tree,
            phase_costs=self.phase_costs,
            n_evaluations=self.n_evaluations,
            final_full_cost=final_full_cost,
        )

    # -- single phase -------------------------------------------------------

    def _run_phase(self, mode: str, n_iter: int) -> None:
        """运行单个保真度阶段的评估循环."""
        for iteration in range(n_iter):
            # ---- Selection ----
            v = self.tree.root
            path = [v]

            while not v.is_leaf:
                v = self._select_child(v)
                path.append(v)

            # v 现在是叶节点: 对应完整策略 s_v
            strategy = v.strategy

            # ---- Evaluation ----
            R_raw = self.igo_evaluate(strategy, mode)
            R = sigma(R_raw)  # R_raw 已饱经 Constran σ, 双重 σ 无害

            # ---- Update best ----
            if R_raw < self.best_cost:
                self.best_cost = R_raw
                self.best_strategy = strategy

            # ---- Backprop ----
            for node in path:
                node.backprop(mode, R)

    # -- selection ----------------------------------------------------------

    def _select_child(self, node: TreeNode) -> TreeNode:
        """
        UCT Selection (§8.5).

        对 node 的每个子节点 child:
          n_child = child.total_n()
          if n_child == 0: 优先探索 (return child)
          Q_tilde_norm = (σ(Q_none + σ(Q_light + σ(Q_full))) + 1) / 2
          UCT = Q_tilde_norm + c · √(ln N_node / n_child)
        返回 UCT 最大的子节点.
        """
        N_node = sum(c.total_n() for c in node.children)

        best_uct = -float('inf')
        best_child = node.children[0]

        for child in node.children:
            n_child = child.total_n()

            # 未访问分支 → 立即优先探索
            if n_child == 0:
                return child

            Q_tilde_norm = child.compute_Q_tilde_norm()
            exploration = self.c * math.sqrt(math.log(N_node) / n_child)
            uct = Q_tilde_norm + exploration

            if uct > best_uct:
                best_uct = uct
                best_child = child

        return best_child

    # -- budget resolution --------------------------------------------------

    def _resolve_budget(self, mode: str) -> int:
        """解析某保真度的预算.

        优先级: 显式配置 > auto-budget > 默认 = 分支数之和 × 倍数.
        """
        budget_attr = f'budget_{mode}'
        explicit = getattr(self.config, budget_attr, None)
        if explicit is not None:
            return explicit

        # 默认: 基于分支数
        n_branches = sum(len(c.children) for c in self.tree.all_nodes() if c.children)
        defaults = {'none': 5, 'light': 1, 'full': 2}
        multiplier = defaults.get(mode, 1)
        return max(len(self.tree.root.children), int(n_branches * multiplier / max(1, self.tree.depth)))

    # -- representation -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GuideTreeSolver(tree={self.tree.describe()}, "
            f"c={self.c:.4f})"
        )


# ===========================================================================
# 5. build_guide_tree_solver — Constran 式 Builder
# ===========================================================================

def build_guide_tree_solver(
    problem: ProblemSpec,
    igo_solver: Optional[Callable] = None,
    fidelity_config: Optional[FidelityConfig] = None,
) -> GuideTreeSolver:
    """
    从 ProblemSpec 一行构建 GuideTreeSolver.

    对标 Constran 的 ``build(objective, constraints) → cost_fn`` 模式.

    Parameters
    ----------
    problem : ProblemSpec
        问题描述 (decision_units + cost_fn + context).
    igo_solver : Callable or None
        IGO 求解器. 若为 None，使用 problem.cost_fn 作为 IGO evaluate.
        签名: (strategy: Dict[str, str], mode: str) -> float.
    fidelity_config : FidelityConfig or None
        保真度配置. 若为 None，使用 auto_fidelity_config.

    Returns
    -------
    GuideTreeSolver
    """
    # ① 从 DecisionUnit 构建索引树
    tree = IndexTree.from_decision_units(
        problem.decision_units,
        macro_compress=True,
    )

    # ② 若未提供 IGO 求解器，从 cost_fn 构建默认 evaluate
    if igo_solver is None:
        _cost_fn = problem.cost_fn
        _context = problem.context or {}

        def default_evaluate(strategy: Dict[str, str], mode: str) -> float:
            """
            默认评估：将 strategy 编码为 cost_fn 的上下文。

            mode='none': 固定连续参数为全 0，仅评估离散拓扑 + 基础连续 cost.
            mode='light': 调用轻量 MGIGO 求解器 (需外部注入).
            mode='full':  调用精评 MGIGO 求解器 (需外部注入).

            当前默认实现仅支持 'none' 模式。
            对于 'light'/'full'，需要在外部注入 igo_solver。
            """
            if mode == 'none':
                # 将策略注入 context，调用 Constran cost
                ctx = {**_context, 'strategy': strategy}
                # 使用全 0 连续参数做单点评估
                import numpy as np
                dummy_x = np.zeros(_estimate_continuous_dim(problem))
                return float(_cost_fn(dummy_x, ctx))
            else:
                raise NotImplementedError(
                    f"Default evaluate only supports mode='none'. "
                    f"For mode='{mode}', provide an external igo_solver."
                )

        igo_solver = default_evaluate

    # ③ 若未提供保真度配置，使用启发式 auto-config
    if fidelity_config is None:
        fidelity_config = auto_fidelity_config(
            n_paths=tree.n_paths,
            n_branches=problem.n_branches,
        )

    return GuideTreeSolver(tree, igo_solver, fidelity_config)


def _estimate_continuous_dim(problem: ProblemSpec) -> int:
    """
    估计连续参数的维度.

    启发式: 每个 DecisionUnit 贡献 1 维（连续松弛的离散变量），
    加上 horizon 个时步的连续控制维度.
    """
    base_dim = problem.n_decision_units
    if problem.horizon is not None:
        base_dim *= problem.horizon
    return max(1, base_dim)
