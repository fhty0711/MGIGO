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
import bisect
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
    静态预建的索引森林，组织为分层结构。

    层宽而总深较浅：
      - 每层（layer）是一组独立 root，代表同一粒度级别的决策类别
      - 层与层之间是粗→细的渐进关系
      - 同层 root 之间无顺序依赖，通过联合 cost 隐式协调

    一条完整路径 = 跨所有层、所有 root 的叶节点组合。
    单 root 时退化为传统树（向后兼容）。
    """

    def __init__(self, roots: List[TreeNode], leaves: List[TreeNode],
                 layers: Optional[List[List[TreeNode]]] = None):
        self.roots = roots
        self.leaves = leaves
        # layers: 可选的分层组织。若为 None，所有 root 视为同一层
        self.layers: List[List[TreeNode]] = layers or [list(roots)]
        self._all_nodes: Optional[List[TreeNode]] = None

    @property
    def root(self) -> TreeNode:
        """向后兼容：单 root 时返回 roots[0]."""
        if len(self.roots) != 1:
            raise AttributeError(
                f"Forest has {len(self.roots)} roots. Use .roots instead of .root."
            )
        return self.roots[0]

    @property
    def n_paths(self) -> int:
        """完整路径总数 = 各 root 叶节点数的乘积."""
        n = 1
        for r in self.roots:
            n_leaves = sum(1 for leaf in self.leaves
                          if self._is_descendant_of(leaf, r))
            if n_leaves > 0:
                n *= n_leaves
        return n

    @property
    def n_leaves_total(self) -> int:
        """所有 root 的叶节点总数（用于 K 估算）."""
        return len(self.leaves)

    @property
    def depth(self) -> int:
        """森林的最大深度."""
        return max(self._depth_from_root(leaf) for leaf in self.leaves)

    def _depth_from_root(self, node: TreeNode) -> int:
        d = 0
        current = node
        while current.parent is not None:
            d += 1
            current = current.parent
        return d

    def _is_descendant_of(self, node: TreeNode, ancestor: TreeNode) -> bool:
        current = node
        while current is not None:
            if current is ancestor:
                return True
            current = current.parent
        return False

    def all_nodes(self) -> List[TreeNode]:
        """返回森林中所有节点（惰性缓存）."""
        if self._all_nodes is None:
            nodes = []
            for root in self.roots:
                stack = [root]
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
        """返回森林的结构摘要."""
        n_nodes = len(self.all_nodes())
        n_root_leaves = sum(1 for r in self.roots
                           for l in self.leaves
                           if self._is_descendant_of(l, r))
        n_layers = len(self.layers) if self.layers else 0
        layer_info = ""
        if self.layers:
            widths = [len(layer) for layer in self.layers]
            layer_info = f", layers={n_layers}(widths={widths})"
        return (
            f"IndexTree(n_roots={len(self.roots)}, n_paths={self.n_paths}"
            f"{layer_info}, nodes={n_nodes}, leaves={n_root_leaves})"
        )

    # -- factory methods ----------------------------------------------------

    @staticmethod
    def from_decision_units(
        units: List[DecisionUnit],
        macro_compress: bool = True,
        max_paths: int = 10000,
    ) -> IndexTree:
        """
        从 DecisionUnit 列表构建静态索引森林（单 root，向后兼容）。

        所有 DecisionUnit 放在同一个 root 下，按列表顺序作为树的层级。
        多 root 场景请用 from_forest()。
        """
        return IndexTree._build_single_root(units, "root", max_paths)

    @staticmethod
    def from_forest(
        groups: List[Tuple[str, List[DecisionUnit]]],
        layers: Optional[List[List[str]]] = None,
        max_paths: int = 100000,
    ) -> IndexTree:
        """
        从多组 DecisionUnit 构建分层索引森林。

        Parameters
        ----------
        groups : List[Tuple[str, List[DecisionUnit]]]
            每个元素为 (group_name, [DecisionUnit, ...]).
        layers : List[List[str]] or None
            分层组织。每层是一组 group_name。
            若为 None，所有 root 视为同一层。
            例: [["Strategy"], ["Baseload","Peaker"], ["Storage","TieLine"]]
        max_paths : int
            路径数上限.

        Returns
        -------
        IndexTree (layered forest)
        """
        name_to_root: Dict[str, TreeNode] = {}
        all_leaves = []
        total_paths = 1

        for group_name, units in groups:
            tree = IndexTree._build_single_root(units, group_name, max_paths)
            name_to_root[group_name] = tree.roots[0]
            all_leaves.extend(tree.leaves)
            total_paths *= len(tree.leaves)

        if total_paths > max_paths:
            import warnings
            warnings.warn(
                f"Large cross-root path count ({total_paths}) exceeds "
                f"max_paths ({max_paths}). Consider blockwise decomposition."
            )

        roots = [name_to_root[name] for name, _ in groups]

        # 构建分层结构
        layer_list = None
        if layers is not None:
            layer_list = []
            for layer_names in layers:
                layer_roots = [name_to_root[n] for n in layer_names
                              if n in name_to_root]
                if layer_roots:
                    layer_list.append(layer_roots)

        forest = IndexTree(roots, all_leaves, layers=layer_list)
        return forest

    @staticmethod
    def _build_single_root(
        units: List[DecisionUnit],
        root_name: str,
        max_paths: int,
    ) -> IndexTree:
        """构建单 root 的索引树（内部辅助）."""
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

        root = TreeNode(name=root_name, choice="")
        leaves: List[TreeNode] = []

        for choices_tuple in product(*all_choices):
            strategy: Dict[str, str] = {}
            current = root

            for i, choice in enumerate(choices_tuple):
                unit_name = units[i].name
                strategy[unit_name] = choice

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

            current.is_leaf = True
            current.strategy = strategy
            leaves.append(current)

        tree = IndexTree([root], leaves)
        if n_paths <= 50:
            _ = tree.all_nodes()
        return tree

    def add_root(self, root: TreeNode, new_leaves: List[TreeNode]) -> None:
        """动态添加一个新 root（新 agent 加入 or 决策粒度细化）。

        新 root 从零开始（Q=0, n=0），不影响已有 root 的统计量。
        已有 Q 统计量保持不变——新 root 通过联合 cost 逐渐学习。

        注意：n_paths 会变为原来的 n_paths × len(new_leaves)。
        """
        self.roots.append(root)
        self.leaves.extend(new_leaves)
        self._all_nodes = None  # 缓存失效

    def remove_root(self, root: TreeNode) -> None:
        """动态移除一个 root（决策维度过期）。

        移除该 root 及其所有子树的节点从 leaves 和 roots 中删除。
        已有 Q 统计量保留在 remaining roots 中。
        """
        self.roots = [r for r in self.roots if r is not root]
        # 移除该 root 下所有叶节点
        removed_leaves = {l for l in self.leaves
                         if self._is_descendant_of(l, root)}
        self.leaves = [l for l in self.leaves if l not in removed_leaves]
        self._all_nodes = None

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
# 4. EliteTracker — top-K 精英路径追踪
# ===========================================================================

class EliteTracker:
    """增量投票式精英追踪。

    每次评估 = 一票。若该评估进入 top-K（精英），路径上所有节点得票。
    Q_j = votes_j / total_evals — 节点 j 被投票为"精英必经之路"的频率。

    语义：我们不是"一次性认识"离散结构，而是逐渐认识——
    通过反复评估和投票，真正的好节点积累票数，噪声假精英被稀释。

    Parameters
    ----------
    K : int
        精英容量。top-K 最低 cost 的路径被视为精英。
    mode : str
        保真度标识 ('none'/'light'/'full')，用于索引 node.Q[mode]。
    """

    def __init__(self, K: int, mode: str, all_nodes_cb=None):
        self.K = K
        self.mode = mode
        self._all_nodes_cb = all_nodes_cb

        # 投票计数：node_id → 该节点被投为精英的次数
        self.votes: Dict[int, int] = {}
        self.total_evals: int = 0

        # 精英列表：按 cost 升序的 top-K 唯一路径 (cost, path_key)
        self.elite_costs: List[Tuple[float, tuple]] = []

    def update(self, cost: float, path: List[TreeNode]) -> None:
        """一次评估 = 一票。

        ① 更新 n 计数
        ② 更新精英列表（去重 + 插入 + 裁剪到 K）
        ③ 若此路径在精英列表中 → 路径上所有节点得票
        ④ Q_j = votes_j / total_evals
        """
        leaf = path[-1]
        path_key = self._path_key(leaf)

        # ① n++
        for node in path:
            node.n[self.mode] += 1
        self.total_evals += 1

        # ② 更新精英列表
        self.elite_costs = [(c, pk) for c, pk in self.elite_costs if pk != path_key]
        idx = bisect.bisect_left([c for c, _ in self.elite_costs], cost)
        self.elite_costs.insert(idx, (cost, path_key))
        if len(self.elite_costs) > self.K:
            self.elite_costs.pop()

        # ③ 投票：此路径是否在精英列表中
        is_elite = any(pk == path_key for _, pk in self.elite_costs)
        if is_elite:
            for node in path:
                nid = id(node)
                self.votes[nid] = self.votes.get(nid, 0) + 1

        # ④ Q_j = votes_j / total_evals
        self._update_all_Q()

    def _update_all_Q(self) -> None:
        """用当前投票状态更新所有节点的 Q。"""
        if self._all_nodes_cb is None or self.total_evals == 0:
            return
        for node in self._all_nodes_cb():
            node.Q[self.mode] = self.votes.get(id(node), 0) / self.total_evals

    def _path_key(self, leaf: TreeNode) -> tuple:
        return tuple(sorted(leaf.strategy.items()))

    def reset(self) -> None:
        """清零投票和精英列表."""
        if self._all_nodes_cb is not None:
            for node in self._all_nodes_cb():
                node.Q[self.mode] = 0.0
        self.votes.clear()
        self.total_evals = 0
        self.elite_costs.clear()

    def describe(self) -> str:
        return (f"EliteTracker(K={self.K}, "
                f"total_evals={self.total_evals}, "
                f"n_elite={len(self.elite_costs)})")


# ===========================================================================
# 5. GuideTreeSolver
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
        返回原始 Constran cost（越小越好），不需要取反或变换。
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

        # Q 计算模式: 'mean' 或 'elite'
        self.q_mode = getattr(self.config, 'q_mode', 'elite')

        # EliteTracker（每个保真度一个）
        # K 基于叶节点总数（非跨 root 乘积）
        n_leaves = tree.n_leaves_total
        K_none  = max(2, int(n_leaves * self.config.elite_fraction_none))
        K_light = max(2, int(n_leaves * self.config.elite_fraction_light))
        K_full  = max(2, int(n_leaves * self.config.elite_fraction_full))
        self.elite_trackers = {
            'none':  EliteTracker(K_none,  'none',  all_nodes_cb=tree.all_nodes),
            'light': EliteTracker(K_light, 'light', all_nodes_cb=tree.all_nodes),
            'full':  EliteTracker(K_full,  'full',  all_nodes_cb=tree.all_nodes),
        }

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
        for tracker in self.elite_trackers.values():
            tracker.reset()
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
        """运行单个保真度阶段的评估循环。

        Forest 模式: 每个 root 独立 UCT 选择，合并为完整策略，
        联合评估，cost 回传给所有 root。
        """
        tracker = self.elite_trackers.get(mode)
        use_elite = (self.q_mode == 'elite' and tracker is not None)
        roots = self.tree.roots

        for iteration in range(n_iter):
            # ---- Selection (per root) ----
            all_paths: List[List[TreeNode]] = []
            strategy: Dict[str, str] = {}

            for root in roots:
                v = root
                root_path = [v]
                while not v.is_leaf:
                    v = self._select_child(v)
                    root_path.append(v)
                all_paths.append(root_path)
                if v.strategy:
                    strategy.update(v.strategy)

            # ---- Evaluation (joint, single cost for all roots) ----
            cost_raw = self.igo_evaluate(strategy, mode)

            # ---- Update best ----
            if cost_raw < self.best_cost:
                self.best_cost = cost_raw
                self.best_strategy = strategy

            # ---- Backprop / Elite update (all roots) ----
            for root_path in all_paths:
                if use_elite:
                    tracker.update(cost_raw, root_path)
                else:
                    R = sigma(cost_raw)
                    for node in root_path:
                        node.backprop(mode, R)

    # -- selection ----------------------------------------------------------

    def _select_child(self, node: TreeNode) -> TreeNode:
        """
        UCT Selection.

        Two Q modes:
          'mean':  Q_j = mean(σ(cost)), lower = better.
                   utility = 1 - Q̃_norm (low cost → high utility).
          'elite': Q_j = elite fraction ∈ [0,1], higher = better.
                   utility = Q̃_norm directly.

        UCT = utility + c · √(ln N_node / n_child), argmax.
        Hoeffding: utility ∈ (0,1), range=1, c=1/√2.
        """
        N_node = sum(c.total_n() for c in node.children)

        best_uct = -float('inf')
        best_child = node.children[0]

        for child in node.children:
            n_child = child.total_n()

            # 未访问分支 → 立即优先探索
            if n_child == 0:
                return child

            Q_tilde = child.compute_Q_tilde()
            Q_tilde_norm = (Q_tilde + 1.0) / 2.0   # ∈ (0,1)

            if self.q_mode == 'elite':
                # Q ∈ [0,1], larger = better → utility = Q̃_norm
                utility = Q_tilde_norm
            else:
                # Q = mean(σ(cost)), lower = better → utility = 1 - Q̃_norm
                utility = 1.0 - Q_tilde_norm

            exploration = self.c * math.sqrt(math.log(N_node) / n_child)
            uct = utility + exploration

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
