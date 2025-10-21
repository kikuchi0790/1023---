"""
Causal Inference Analysis for Process Insight Modeler
因果推論分析

観測されたネットワークから因果関係を推定し、介入効果を予測する。
"""

from typing import List, Dict, Tuple, Any, Callable, Set
import numpy as np
import pandas as pd
import networkx as nx
import logging
from dataclasses import dataclass
import time
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class CausalInferenceResult:
    """因果推論の結果"""
    direct_effects: Dict[Tuple[str, str], float]  # (source, target) -> effect
    indirect_effects: Dict[Tuple[str, str], float]
    total_effects: Dict[Tuple[str, str], float]
    causal_paths: Dict[Tuple[str, str], List[List[str]]]  # 因果経路
    intervention_effects: Dict[str, Dict[str, float]]  # do(X) -> {Y: effect}
    confounders: List[Tuple[str, str, List[str]]]  # (X, Y, confounders)
    top_intervention_targets: List[Tuple[str, float]]  # 介入効果が高いノード
    interpretation: str
    computation_time: float
    max_path_length: int


class CausalInferenceAnalyzer:
    """
    因果推論分析クラス
    
    Pearl's Causal Inferenceの枠組みで因果効果を推定:
    - 直接効果（Direct Effect）
    - 間接効果（Indirect Effect）
    - 総効果（Total Effect）
    - Do-operator シミュレーション
    - 因果経路の分析
    - 交絡因子の検出
    """
    
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_names: List[str],
        max_path_length: int = 4,
        decay_factor: float = 0.8
    ):
        """
        Args:
            adjacency_matrix: 隣接行列（N×N）
            node_names: ノード名リスト
            max_path_length: 因果経路の最大長
            decay_factor: 経路減衰係数（0-1、長い経路ほど効果が減衰）
        """
        self.matrix = adjacency_matrix.copy()
        self.node_names = node_names
        self.n = len(node_names)
        self.max_path_length = max_path_length
        self.decay_factor = decay_factor
        
        # グラフ構築
        self.graph = nx.DiGraph()
        for i, source in enumerate(node_names):
            for j, target in enumerate(node_names):
                if i != j and abs(self.matrix[i, j]) > 1e-6:
                    self.graph.add_edge(source, target, weight=self.matrix[i, j])
        
        logger.info(f"CausalInferenceAnalyzer初期化: {self.n}ノード, 最大経路長{max_path_length}")
    
    def compute_causal_inference(
        self,
        intervention_node: str = None,
        intervention_strength: float = 1.5,
        progress_callback: Callable[[str, float], None] = None
    ) -> CausalInferenceResult:
        """
        包括的な因果推論分析を実行
        
        Args:
            intervention_node: 介入対象ノード（Noneの場合は全ノード）
            intervention_strength: 介入の強さ（1.0=現状、1.5=50%改善）
            progress_callback: 進捗コールバック(message, pct)
        
        Returns:
            CausalInferenceResult
        """
        start_time = time.time()
        
        # 1. 直接効果を計算
        if progress_callback:
            progress_callback("直接効果を計算中...", 0.0)
        
        direct_effects = self._compute_direct_effects()
        
        # 2. 間接効果を計算
        if progress_callback:
            progress_callback("間接効果を計算中...", 0.2)
        
        indirect_effects = self._compute_indirect_effects(progress_callback)
        
        # 3. 総効果を計算
        if progress_callback:
            progress_callback("総効果を計算中...", 0.5)
        
        total_effects = self._compute_total_effects(direct_effects, indirect_effects)
        
        # 4. 因果経路を探索
        if progress_callback:
            progress_callback("因果経路を探索中...", 0.6)
        
        causal_paths = self._find_all_causal_paths()
        
        # 5. 介入効果をシミュレーション
        if progress_callback:
            progress_callback("介入効果をシミュレーション中...", 0.7)
        
        if intervention_node:
            intervention_effects = {
                intervention_node: self._simulate_intervention(
                    intervention_node, intervention_strength
                )
            }
        else:
            # 全ノードに対して介入効果を計算
            intervention_effects = {}
            for i, node in enumerate(self.node_names):
                if progress_callback:
                    pct = 0.7 + 0.2 * (i / self.n)
                    progress_callback(f"介入効果: {node}...", pct)
                
                intervention_effects[node] = self._simulate_intervention(
                    node, intervention_strength
                )
        
        # 6. 交絡因子を検出
        if progress_callback:
            progress_callback("交絡因子を検出中...", 0.9)
        
        confounders = self._detect_confounders()
        
        # 7. 最も効果的な介入ターゲットを特定
        top_targets = self._identify_top_intervention_targets(intervention_effects)
        
        computation_time = time.time() - start_time
        logger.info(f"因果推論分析完了: {computation_time:.2f}秒")
        
        # 8. 解釈文を生成
        interpretation = self._generate_interpretation(
            direct_effects, indirect_effects, total_effects,
            intervention_effects, confounders, top_targets,
            intervention_node, intervention_strength
        )
        
        return CausalInferenceResult(
            direct_effects=direct_effects,
            indirect_effects=indirect_effects,
            total_effects=total_effects,
            causal_paths=causal_paths,
            intervention_effects=intervention_effects,
            confounders=confounders,
            top_intervention_targets=top_targets,
            interpretation=interpretation,
            computation_time=computation_time,
            max_path_length=self.max_path_length
        )
    
    def _compute_direct_effects(self) -> Dict[Tuple[str, str], float]:
        """
        直接効果を計算
        
        直接効果 = 隣接行列の値（1ホップの影響）
        
        Returns:
            {(source, target): effect}
        """
        direct_effects = {}
        
        for i, source in enumerate(self.node_names):
            for j, target in enumerate(self.node_names):
                if i != j and abs(self.matrix[i, j]) > 1e-6:
                    direct_effects[(source, target)] = self.matrix[i, j]
        
        return direct_effects
    
    def _compute_indirect_effects(
        self,
        progress_callback: Callable[[str, float], None] = None
    ) -> Dict[Tuple[str, str], float]:
        """
        間接効果を計算
        
        間接効果 = 全ての2ホップ以上の経路の効果の合計
        
        Returns:
            {(source, target): effect}
        """
        indirect_effects = {}
        
        # 全ノードペアに対して間接効果を計算
        for i, source in enumerate(self.node_names):
            if progress_callback and i % 5 == 0:
                pct = 0.2 + 0.3 * (i / self.n)
                progress_callback(f"間接効果: {source}...", pct)
            
            for target in self.node_names:
                if source == target:
                    continue
                
                # 2ホップ以上の全経路を探索
                indirect_effect = 0.0
                
                for path_length in range(2, self.max_path_length + 1):
                    paths = self._find_paths_with_length(source, target, path_length)
                    
                    for path in paths:
                        # 経路の効果 = 各エッジの重みの積 × 減衰係数^(経路長-1)
                        path_effect = 1.0
                        for k in range(len(path) - 1):
                            node_from = path[k]
                            node_to = path[k + 1]
                            idx_from = self.node_names.index(node_from)
                            idx_to = self.node_names.index(node_to)
                            path_effect *= self.matrix[idx_from, idx_to]
                        
                        # 減衰係数を適用
                        path_effect *= (self.decay_factor ** (len(path) - 2))
                        indirect_effect += path_effect
                
                if abs(indirect_effect) > 1e-6:
                    indirect_effects[(source, target)] = indirect_effect
        
        return indirect_effects
    
    def _compute_total_effects(
        self,
        direct_effects: Dict[Tuple[str, str], float],
        indirect_effects: Dict[Tuple[str, str], float]
    ) -> Dict[Tuple[str, str], float]:
        """
        総効果を計算
        
        総効果 = 直接効果 + 間接効果
        
        Returns:
            {(source, target): effect}
        """
        total_effects = {}
        
        # 全てのペアを収集
        all_pairs = set(direct_effects.keys()) | set(indirect_effects.keys())
        
        for pair in all_pairs:
            direct = direct_effects.get(pair, 0.0)
            indirect = indirect_effects.get(pair, 0.0)
            total_effects[pair] = direct + indirect
        
        return total_effects
    
    def _find_paths_with_length(
        self,
        source: str,
        target: str,
        length: int
    ) -> List[List[str]]:
        """
        指定された長さの経路を全て探索
        
        Args:
            source: 始点
            target: 終点
            length: 経路長
        
        Returns:
            経路のリスト
        """
        if length < 1:
            return []
        
        if length == 1:
            # 直接エッジがあるか確認
            if self.graph.has_edge(source, target):
                return [[source, target]]
            else:
                return []
        
        # DFSで経路を探索
        paths = []
        
        def dfs(current: str, path: List[str], remaining: int):
            if remaining == 0:
                if current == target:
                    paths.append(path[:])
                return
            
            if current in self.graph:
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in path:  # サイクルを避ける
                        path.append(neighbor)
                        dfs(neighbor, path, remaining - 1)
                        path.pop()
        
        dfs(source, [source], length)
        
        return paths
    
    def _find_all_causal_paths(self) -> Dict[Tuple[str, str], List[List[str]]]:
        """
        全ノードペア間の因果経路を探索
        
        Returns:
            {(source, target): [path1, path2, ...]}
        """
        causal_paths = {}
        
        for source in self.node_names:
            for target in self.node_names:
                if source == target:
                    continue
                
                # 最大長までの全経路を探索
                all_paths = []
                for length in range(1, self.max_path_length + 1):
                    paths = self._find_paths_with_length(source, target, length)
                    all_paths.extend(paths)
                
                if all_paths:
                    # 効果が大きい順にソート
                    all_paths.sort(key=lambda p: self._compute_path_effect(p), reverse=True)
                    causal_paths[(source, target)] = all_paths[:10]  # 上位10経路
        
        return causal_paths
    
    def _compute_path_effect(self, path: List[str]) -> float:
        """経路の効果を計算"""
        effect = 1.0
        for k in range(len(path) - 1):
            idx_from = self.node_names.index(path[k])
            idx_to = self.node_names.index(path[k + 1])
            effect *= self.matrix[idx_from, idx_to]
        
        effect *= (self.decay_factor ** (len(path) - 2))
        return abs(effect)
    
    def _simulate_intervention(
        self,
        target_node: str,
        intervention_strength: float
    ) -> Dict[str, float]:
        """
        do(X=x) による介入効果のシミュレーション
        
        Args:
            target_node: 介入対象ノード
            intervention_strength: 介入の強さ（1.0=現状、1.5=50%改善）
        
        Returns:
            各ノードへの因果効果 {node: effect}
        """
        effects = {node: 0.0 for node in self.node_names}
        
        # 介入ノード自身の効果
        effects[target_node] = intervention_strength - 1.0
        
        # 介入ノードから到達可能な全ノードへの効果を計算
        if target_node in self.graph:
            # BFSで波及効果を計算
            visited = {target_node}
            queue = deque([(target_node, intervention_strength - 1.0)])
            
            while queue:
                current, current_effect = queue.popleft()
                
                if current not in self.graph:
                    continue
                
                for neighbor in self.graph.neighbors(current):
                    idx_current = self.node_names.index(current)
                    idx_neighbor = self.node_names.index(neighbor)
                    edge_weight = self.matrix[idx_current, idx_neighbor]
                    
                    # 効果の伝播
                    propagated_effect = current_effect * edge_weight
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        effects[neighbor] += propagated_effect
                        
                        # 効果が十分大きい場合のみ継続
                        if abs(propagated_effect) > 0.01:
                            queue.append((neighbor, propagated_effect))
                    else:
                        # 既訪問でも効果を累積
                        effects[neighbor] += propagated_effect
        
        return effects
    
    def _detect_confounders(self) -> List[Tuple[str, str, List[str]]]:
        """
        交絡因子を検出
        
        Backdoor pathが存在する場合、共通原因（交絡因子）が存在する。
        
        Returns:
            [(source, target, [confounder1, confounder2, ...])]
        """
        confounders_list = []
        
        for source in self.node_names:
            for target in self.node_names:
                if source == target:
                    continue
                
                # Backdoor pathを探索（source <- ... -> target）
                confounders = self._find_confounders_for_pair(source, target)
                
                if confounders:
                    confounders_list.append((source, target, confounders))
        
        return confounders_list
    
    def _find_confounders_for_pair(self, source: str, target: str) -> List[str]:
        """特定のペアに対する交絡因子を探索"""
        confounders = []
        
        # sourceの親ノード（source に入ってくるエッジ）
        source_parents = set()
        if source in self.graph:
            source_parents = set(self.graph.predecessors(source))
        
        # targetの親ノード
        target_parents = set()
        if target in self.graph:
            target_parents = set(self.graph.predecessors(target))
        
        # 共通の親 = 交絡因子
        common_parents = source_parents & target_parents
        
        if common_parents:
            confounders = list(common_parents)
        
        return confounders
    
    def _identify_top_intervention_targets(
        self,
        intervention_effects: Dict[str, Dict[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        最も効果的な介入ターゲットを特定
        
        Args:
            intervention_effects: {intervention_node: {affected_node: effect}}
        
        Returns:
            [(node, total_impact), ...] 総影響力が大きい順
        """
        impact_scores = {}
        
        for node, effects in intervention_effects.items():
            # 総影響力 = 全ノードへの効果の絶対値の合計
            total_impact = sum(abs(effect) for effect in effects.values())
            impact_scores[node] = total_impact
        
        # 降順ソート
        top_targets = sorted(impact_scores.items(), key=lambda x: x[1], reverse=True)
        
        return top_targets
    
    def _generate_interpretation(
        self,
        direct_effects: Dict,
        indirect_effects: Dict,
        total_effects: Dict,
        intervention_effects: Dict,
        confounders: List,
        top_targets: List,
        intervention_node: str,
        intervention_strength: float
    ) -> str:
        """平易な日本語の解釈文を生成"""
        
        n_direct = len(direct_effects)
        n_indirect = len(indirect_effects)
        n_total = len(total_effects)
        n_confounders = len(confounders)
        
        interpretation = f"""
## 🔍 因果推論分析結果の解釈

### 因果効果のサマリー

- **直接効果のペア数**: {n_direct}
- **間接効果のペア数**: {n_indirect}
- **総効果のペア数**: {n_total}
- **検出された交絡因子**: {n_confounders}
- **最大経路長**: {self.max_path_length}

### 💡 直接効果 vs 間接効果

**直接効果**は、ノード間の直接的な影響（1ホップ）です。
**間接効果**は、他のノードを経由した影響（2ホップ以上）です。

"""
        
        if n_indirect > 0:
            ratio = n_indirect / n_direct if n_direct > 0 else 0
            interpretation += f"""
間接効果のペア数が直接効果の**{ratio:.1f}倍**あります。
これは、多くの影響が間接的に伝播していることを示しており、
システム全体の複雑な相互依存関係を表しています。
"""
        
        interpretation += "\n### 🎯 介入効果のシミュレーション\n\n"
        
        if intervention_node:
            effects = intervention_effects.get(intervention_node, {})
            if effects:
                sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)
                top_affected = sorted_effects[:5]
                
                interpretation += f"""
**do({intervention_node})** を実行した場合（介入強度: {intervention_strength:.1f}）:

影響を最も受けるノード（上位5件）:
"""
                for node, effect in top_affected:
                    direction = "↑ 改善" if effect > 0 else "↓ 悪化"
                    interpretation += f"- **{node}**: {effect:+.4f} ({direction})\n"
        
        interpretation += "\n### 🏆 最も効果的な介入ターゲット\n\n"
        
        if top_targets:
            interpretation += """
以下のノードに介入すると、システム全体への影響が最も大きくなります:

"""
            for i, (node, impact) in enumerate(top_targets[:10], 1):
                interpretation += f"{i}. **{node}**: 総影響力 {impact:.4f}\n"
        
        if n_confounders > 0:
            interpretation += f"""
### ⚠️ 交絡因子の検出

{n_confounders}件の交絡因子が検出されました。

**交絡因子**とは、2つのノードに共通して影響を与える第三の要因です。
これらの存在により、見かけの因果関係が実際より強く（または弱く）見える可能性があります。

**対処法**:
- 交絡因子を制御（一定に保つ）して分析を行う
- 交絡因子自体を改善ターゲットとする
"""
        
        interpretation += """
### 📖 活用方法

1. **改善施策の優先順位決定**: 総影響力が高いノードから改善
2. **効果予測**: do(X)シミュレーションで改善効果を事前予測
3. **因果経路の分析**: どの経路で影響が伝播するかを特定
4. **交絡の排除**: 正確な因果関係の把握

### 注意事項

- 因果推論は観測データからの推定であり、真の因果関係を保証するものではありません
- 実際の改善施策を実施する際は、小規模なパイロット実験で効果を検証することを推奨します
- 交絡因子が存在する場合、因果効果が過大評価されている可能性があります
"""
        
        return interpretation.strip()
