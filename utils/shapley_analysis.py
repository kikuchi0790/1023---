"""
Shapley Value Analysis for Process Insight Modeler
協力貢献度分析（Shapley Value）

各ノードの真の限界貢献度を公平に評価する。
協力ゲーム理論に基づき、「このノードを削除したら全体性能がどれだけ下がるか」を数値化。
"""

from typing import List, Dict, Tuple, Any, Callable
import numpy as np
import networkx as nx
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ShapleyResult:
    """Shapley Value分析の結果"""
    shapley_values: Dict[str, float]  # ノード名 → Shapley値
    top_contributors: List[Tuple[str, float]]  # (ノード名, Shapley値)のリスト（降順）
    cumulative_contribution: List[Tuple[int, float]]  # (上位N, 累積貢献率%)
    category_contributions: Dict[str, float]  # カテゴリ → 平均Shapley値
    total_value: float  # V(全ノード)
    computation_time: float  # 計算時間（秒）
    n_samples: int  # サンプル数
    interpretation: str  # 平易な解釈文


class ShapleyAnalyzer:
    """
    Shapley Value分析クラス
    
    協力ゲーム理論に基づく公平な貢献度評価:
    - Monte Carlo近似で計算効率化（正確解は2^N通り）
    - 価値関数V: ネットワークの部分集合の性能
    - 限界貢献: ノードを追加したときの性能向上
    """
    
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_names: List[str],
        node_categories: Dict[str, str] = None,
        value_function: str = "pagerank_sum"
    ):
        """
        Args:
            adjacency_matrix: 隣接行列（N×N）
            node_names: ノード名リスト
            node_categories: ノード名 → カテゴリ名の辞書（オプション）
            value_function: 価値関数タイプ（"pagerank_sum", "efficiency", "connectivity"）
        """
        self.matrix = adjacency_matrix.copy()
        self.node_names = node_names
        self.node_categories = node_categories or {}
        self.n = len(node_names)
        
        # 価値関数の選択
        if value_function == "pagerank_sum":
            self.value_func = self._value_pagerank_sum
        elif value_function == "efficiency":
            self.value_func = self._value_network_efficiency
        elif value_function == "connectivity":
            self.value_func = self._value_connectivity
        else:
            raise ValueError(f"未知の価値関数: {value_function}")
        
        logger.info(f"ShapleyAnalyzer初期化: {self.n}ノード, 価値関数={value_function}")
    
    def compute_shapley_values(
        self,
        n_samples: int = 1000,
        random_seed: int = None,
        progress_callback: Callable[[int, int], None] = None
    ) -> ShapleyResult:
        """
        Monte Carlo近似でShapley Valueを計算
        
        Args:
            n_samples: サンプル数（多いほど精度向上、計算時間増加）
            random_seed: 乱数シード（再現性のため）
            progress_callback: 進捗コールバック関数(current, total)
        
        Returns:
            ShapleyResult
        """
        start_time = time.time()
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Shapley値を初期化
        shapley_values = {name: 0.0 for name in self.node_names}
        
        logger.info(f"Shapley Value計算開始: {n_samples}サンプル")
        
        # Monte Carloサンプリング
        for sample in range(n_samples):
            # ランダムな順列を生成
            permutation = np.random.permutation(self.n)
            
            # 各ノードの限界貢献を計算
            for i, node_idx in enumerate(permutation):
                node_name = self.node_names[node_idx]
                
                # このノードより前のノード集合
                S_before_indices = permutation[:i]
                S_after_indices = permutation[:i+1]
                
                # 価値関数を評価
                V_before = self._evaluate_coalition(S_before_indices)
                V_after = self._evaluate_coalition(S_after_indices)
                
                # 限界貢献
                marginal_contribution = V_after - V_before
                
                # Shapley値に加算
                shapley_values[node_name] += marginal_contribution
            
            # 進捗報告
            if progress_callback and (sample + 1) % 50 == 0:
                progress_callback(sample + 1, n_samples)
        
        # 平均化
        for name in shapley_values:
            shapley_values[name] /= n_samples
        
        # 全体の価値
        total_value = self._evaluate_coalition(np.arange(self.n))
        
        computation_time = time.time() - start_time
        logger.info(f"Shapley Value計算完了: {computation_time:.2f}秒")
        
        # 結果を整形
        result = self._format_result(shapley_values, total_value, n_samples, computation_time)
        
        return result
    
    def _evaluate_coalition(self, coalition_indices: np.ndarray) -> float:
        """
        連携集合（部分グラフ）の価値を評価
        
        Args:
            coalition_indices: 連携に含まれるノードのインデックス
        
        Returns:
            価値スコア
        """
        if len(coalition_indices) == 0:
            return 0.0
        
        # 部分グラフを抽出
        submatrix = self.matrix[np.ix_(coalition_indices, coalition_indices)]
        
        # 価値関数を適用
        value = self.value_func(submatrix, len(coalition_indices))
        
        return value
    
    def _value_pagerank_sum(self, submatrix: np.ndarray, n_nodes: int) -> float:
        """
        価値関数: PageRankの合計
        
        ネットワークの影響力の総和を評価
        """
        if n_nodes == 0:
            return 0.0
        
        try:
            # NetworkXグラフに変換
            G = nx.from_numpy_array(submatrix, create_using=nx.DiGraph)
            
            # PageRank計算
            if G.number_of_edges() == 0:
                # エッジがない場合は均等
                return 1.0
            
            pagerank = nx.pagerank(G, weight='weight')
            
            # 合計（正規化されているので、平均的には1/n_nodes）
            return sum(pagerank.values())
        
        except Exception as e:
            logger.warning(f"PageRank計算エラー: {e}")
            return 0.0
    
    def _value_network_efficiency(self, submatrix: np.ndarray, n_nodes: int) -> float:
        """
        価値関数: ネットワーク効率性
        
        平均最短経路長の逆数（効率性）
        """
        if n_nodes <= 1:
            return 0.0
        
        try:
            G = nx.from_numpy_array(submatrix, create_using=nx.DiGraph)
            
            # グローバル効率性（disconnectedでも計算可能）
            efficiency = nx.global_efficiency(G)
            
            return efficiency
        
        except Exception as e:
            logger.warning(f"効率性計算エラー: {e}")
            return 0.0
    
    def _value_connectivity(self, submatrix: np.ndarray, n_nodes: int) -> float:
        """
        価値関数: 接続性
        
        エッジ数の正規化値
        """
        if n_nodes <= 1:
            return 0.0
        
        # 非ゼロ要素数（エッジ数）
        n_edges = np.count_nonzero(submatrix)
        
        # 可能な最大エッジ数で正規化
        max_edges = n_nodes * (n_nodes - 1)
        
        if max_edges == 0:
            return 0.0
        
        return n_edges / max_edges
    
    def _format_result(
        self,
        shapley_values: Dict[str, float],
        total_value: float,
        n_samples: int,
        computation_time: float
    ) -> ShapleyResult:
        """
        結果を整形してShapleyResultオブジェクトを生成
        """
        # 降順ソート
        sorted_items = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
        top_contributors = sorted_items
        
        # 累積貢献度
        cumulative_contribution = []
        cumulative_sum = 0.0
        total_shapley_sum = sum(shapley_values.values())
        
        for i, (name, value) in enumerate(sorted_items, 1):
            cumulative_sum += value
            if total_shapley_sum > 0:
                cumulative_pct = (cumulative_sum / total_shapley_sum) * 100
            else:
                cumulative_pct = 0.0
            cumulative_contribution.append((i, cumulative_pct))
        
        # カテゴリ別平均貢献度
        category_contributions = {}
        if self.node_categories:
            category_sums = {}
            category_counts = {}
            
            for name, value in shapley_values.items():
                category = self.node_categories.get(name, "Unknown")
                category_sums[category] = category_sums.get(category, 0.0) + value
                category_counts[category] = category_counts.get(category, 0) + 1
            
            for category, total in category_sums.items():
                count = category_counts[category]
                category_contributions[category] = total / count if count > 0 else 0.0
        
        # 平易な解釈文を生成
        interpretation = self._generate_interpretation(
            top_contributors, cumulative_contribution, total_value
        )
        
        return ShapleyResult(
            shapley_values=shapley_values,
            top_contributors=top_contributors,
            cumulative_contribution=cumulative_contribution,
            category_contributions=category_contributions,
            total_value=total_value,
            computation_time=computation_time,
            n_samples=n_samples,
            interpretation=interpretation
        )
    
    def _generate_interpretation(
        self,
        top_contributors: List[Tuple[str, float]],
        cumulative_contribution: List[Tuple[int, float]],
        total_value: float
    ) -> str:
        """
        平易な日本語の解釈文を生成
        """
        if len(top_contributors) == 0:
            return "分析結果が得られませんでした。"
        
        # 最上位ノード
        top_node, top_value = top_contributors[0]
        top_pct = (top_value / total_value * 100) if total_value > 0 else 0
        
        # 80%達成するノード数
        n_for_80_pct = next(
            (n for n, pct in cumulative_contribution if pct >= 80.0),
            len(top_contributors)
        )
        
        # 負の値を持つノード
        negative_nodes = [name for name, value in top_contributors if value < 0]
        
        interpretation = f"""
## 📊 Shapley Value分析結果の解釈

### 最重要ノード
**「{top_node}」**が最も高い貢献度を示しており、全体性能の約**{top_pct:.1f}%**を担っています。
このプロセスへの投資が最も効果的です。

### 重点管理対象
上位**{n_for_80_pct}ノード**で全体の**80%**の貢献を説明できます。
これらのノードを重点的に管理することで、効率的な改善が可能です。

### 貢献度分布
- 総ノード数: {len(top_contributors)}
- 全体価値: {total_value:.4f}
- 平均貢献度: {total_value/len(top_contributors):.4f}
"""
        
        if negative_nodes:
            interpretation += f"""
### ⚠️ 要再検討ノード
以下の{len(negative_nodes)}ノードは負の貢献度を示しています:
{', '.join(f'「{name}」' for name in negative_nodes[:3])}{'...' if len(negative_nodes) > 3 else ''}

これらは削除または再設計により、全体性能が向上する可能性があります。
"""
        
        interpretation += """
### 💡 活用方法
1. **投資優先順位**: Shapley値が高いノードから改善
2. **リソース配分**: 貢献度に応じた予算配分
3. **ボトルネック発見**: 「縁の下の力持ち」の可視化
4. **プロセス簡素化**: 負の値ノードの削減検討
"""
        
        return interpretation.strip()


def compute_shapley_coalition_stability(
    shapley_values: Dict[str, float],
    adjacency_matrix: np.ndarray,
    node_names: List[str]
) -> Dict[str, Any]:
    """
    連携の安定性を分析
    
    Shapley値が高いノード同士は協力すべきか？
    
    Returns:
        stable_coalitions: 安定した連携候補
    """
    n = len(node_names)
    
    # Shapley値の上位25%
    sorted_nodes = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
    top_25_pct_count = max(1, n // 4)
    top_nodes = [name for name, _ in sorted_nodes[:top_25_pct_count]]
    
    # 上位ノード間の接続強度
    top_indices = [node_names.index(name) for name in top_nodes]
    top_submatrix = adjacency_matrix[np.ix_(top_indices, top_indices)]
    
    # 密結合ペア
    dense_pairs = []
    for i, name_i in enumerate(top_nodes):
        for j, name_j in enumerate(top_nodes):
            if i < j and top_submatrix[i, j] != 0:
                strength = abs(top_submatrix[i, j])
                dense_pairs.append((name_i, name_j, strength))
    
    dense_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return {
        "top_contributors": top_nodes,
        "dense_connections": dense_pairs[:10],
        "recommendation": f"上位{len(top_nodes)}ノードの連携を強化することで、相乗効果が期待できます。"
    }
