"""
Bootstrap Statistical Testing for Process Insight Modeler
統計的検定（Bootstrap法）

既存の分析結果に信頼区間と有意性検定を付与し、統計的信頼性を担保する。
"""

from typing import List, Dict, Tuple, Any, Callable
import numpy as np
import pandas as pd
import networkx as nx
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Bootstrap統計検定の結果"""
    metric_name: str  # "PageRank", "Shapley", "TE"等
    node_ci: Dict[str, Tuple[float, float, float]]  # ノード名 → (値, 下限, 上限)
    group_comparison: pd.DataFrame  # グループ間比較（p値付き）
    stable_findings: List[str]  # 統計的に安定した知見
    unstable_findings: List[str]  # 不安定（再評価推奨）
    interpretation: str  # 平易な解釈文
    computation_time: float
    n_bootstrap: int  # リサンプル回数
    alpha: float  # 有意水準


class BootstrapTester:
    """
    Bootstrap統計検定クラス
    
    リサンプリング法で信頼区間を計算し、Permutation検定で有意性を評価。
    """
    
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_names: List[str],
        node_groups: Dict[str, str] = None,
        n_bootstrap: int = 1000,
        alpha: float = 0.05
    ):
        """
        Args:
            adjacency_matrix: 隣接行列（N×N）
            node_names: ノード名リスト
            node_groups: ノード名 → グループ名の辞書（オプション）
            n_bootstrap: リサンプル回数
            alpha: 有意水準（0.05 = 95%信頼区間）
        """
        self.matrix = adjacency_matrix.copy()
        self.node_names = node_names
        self.node_groups = node_groups or {}
        self.n = len(node_names)
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        
        logger.info(f"BootstrapTester初期化: {self.n}ノード, {n_bootstrap}リサンプル")
    
    def bootstrap_confidence_interval(
        self,
        metric_func: Callable[[np.ndarray], Dict[str, float]],
        progress_callback: Callable[[int, int], None] = None
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Bootstrap法で信頼区間を計算
        
        Args:
            metric_func: 隣接行列を受け取り、{ノード名: スコア}を返す関数
            progress_callback: 進捗コールバック(current, total)
        
        Returns:
            {ノード名: (元の値, 下限, 上限)}
        """
        # 元の値を計算
        original_scores = metric_func(self.matrix)
        
        # 非ゼロ要素のインデックス
        nonzero_i, nonzero_j = np.where(self.matrix != 0)
        n_edges = len(nonzero_i)
        
        if n_edges == 0:
            logger.warning("非ゼロ要素がありません")
            return {node: (0.0, 0.0, 0.0) for node in self.node_names}
        
        # Bootstrap サンプリング
        bootstrap_samples = []
        
        for b in range(self.n_bootstrap):
            # 復元抽出でリサンプル
            resampled_indices = np.random.choice(n_edges, size=n_edges, replace=True)
            
            # リサンプル行列を構築
            resampled_matrix = np.zeros_like(self.matrix)
            
            for idx in resampled_indices:
                i, j = nonzero_i[idx], nonzero_j[idx]
                resampled_matrix[i, j] += self.matrix[i, j]
            
            # 正規化（期待値を元と同じに）
            resampled_matrix = resampled_matrix / n_edges * n_edges
            
            try:
                # メトリック計算
                resampled_scores = metric_func(resampled_matrix)
                bootstrap_samples.append(resampled_scores)
            except Exception as e:
                logger.warning(f"Bootstrapサンプル{b}でエラー: {e}")
                continue
            
            if progress_callback and (b + 1) % 50 == 0:
                progress_callback(b + 1, self.n_bootstrap)
        
        # 信頼区間を計算
        ci_results = {}
        
        for node_name in original_scores:
            # このノードのBootstrapサンプル
            node_samples = [
                sample.get(node_name, 0) for sample in bootstrap_samples
            ]
            
            if len(node_samples) == 0:
                ci_results[node_name] = (original_scores[node_name], 0, 0)
                continue
            
            # パーセンタイル法
            lower = np.percentile(node_samples, self.alpha/2 * 100)
            upper = np.percentile(node_samples, (1 - self.alpha/2) * 100)
            
            ci_results[node_name] = (
                original_scores[node_name],
                lower,
                upper
            )
        
        return ci_results
    
    def permutation_test(
        self,
        metric_func: Callable,
        group_a_nodes: List[str],
        group_b_nodes: List[str],
        n_permutations: int = 1000
    ) -> Dict[str, Any]:
        """
        2群間の差のPermutation検定
        
        Args:
            metric_func: メトリック計算関数
            group_a_nodes: グループAのノード名リスト
            group_b_nodes: グループBのノード名リスト
            n_permutations: Permutation回数
        
        Returns:
            検定結果の辞書
        """
        # 元のスコアを計算
        original_scores = metric_func(self.matrix)
        
        # 各グループの平均値
        group_a_scores = [original_scores.get(node, 0) for node in group_a_nodes]
        group_b_scores = [original_scores.get(node, 0) for node in group_b_nodes]
        
        if len(group_a_scores) == 0 or len(group_b_scores) == 0:
            return {
                "observed_diff": 0,
                "p_value": 1.0,
                "significant": False,
                "null_distribution": []
            }
        
        observed_diff = np.mean(group_a_scores) - np.mean(group_b_scores)
        
        # Null分布を生成
        null_distribution = []
        
        pooled_nodes = group_a_nodes + group_b_nodes
        n_a = len(group_a_nodes)
        
        for _ in range(n_permutations):
            # ラベルをランダムに入れ替え
            perm_indices = np.random.permutation(len(pooled_nodes))
            perm_nodes = [pooled_nodes[i] for i in perm_indices]
            
            perm_group_a = perm_nodes[:n_a]
            perm_group_b = perm_nodes[n_a:]
            
            perm_a_scores = [original_scores.get(node, 0) for node in perm_group_a]
            perm_b_scores = [original_scores.get(node, 0) for node in perm_group_b]
            
            perm_diff = np.mean(perm_a_scores) - np.mean(perm_b_scores)
            null_distribution.append(perm_diff)
        
        # p値計算（両側検定）
        p_value = (np.abs(null_distribution) >= np.abs(observed_diff)).mean()
        
        return {
            "observed_diff": observed_diff,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "null_distribution": null_distribution,
            "group_a_mean": np.mean(group_a_scores),
            "group_b_mean": np.mean(group_b_scores)
        }
    
    def run_comprehensive_bootstrap_analysis(
        self,
        metric_name: str = "PageRank",
        metric_func: Callable = None,
        progress_callback: Callable[[str, float], None] = None
    ) -> BootstrapResult:
        """
        包括的なBootstrap分析を実行
        
        Args:
            metric_name: メトリック名
            metric_func: メトリック計算関数（Noneの場合はPageRank）
            progress_callback: 進捗コールバック(message, progress_pct)
        
        Returns:
            BootstrapResult
        """
        start_time = time.time()
        
        # デフォルトはPageRank
        if metric_func is None:
            def pagerank_func(matrix):
                G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
                try:
                    pr = nx.pagerank(G, weight='weight')
                except:
                    pr = nx.pagerank(G)
                return pr
            
            metric_func = pagerank_func
        
        # 1. 信頼区間計算
        if progress_callback:
            progress_callback("Bootstrap信頼区間計算中...", 0.0)
        
        def bootstrap_progress(current, total):
            if progress_callback:
                pct = 0.0 + 0.7 * (current / total)
                progress_callback(f"Bootstrap {current}/{total}...", pct)
        
        ci_results = self.bootstrap_confidence_interval(
            metric_func,
            progress_callback=bootstrap_progress
        )
        
        # 2. 安定性評価
        stable_findings = []
        unstable_findings = []
        
        for node, (value, lower, upper) in ci_results.items():
            # 相対誤差
            if abs(value) > 1e-6:
                rel_error = (upper - lower) / (2 * abs(value))
                
                if rel_error < 0.2:  # 20%以内
                    stable_findings.append(
                        f"{node}: {value:.4f} [{lower:.4f}, {upper:.4f}]"
                    )
                else:
                    unstable_findings.append(
                        f"{node}: {value:.4f} [{lower:.4f}, {upper:.4f}] (相対誤差{rel_error*100:.1f}%)"
                    )
        
        # 3. グループ間比較
        if progress_callback:
            progress_callback("グループ間比較実行中...", 0.7)
        
        group_comparison = self._compare_groups(metric_func)
        
        computation_time = time.time() - start_time
        logger.info(f"Bootstrap分析完了: {computation_time:.2f}秒")
        
        # 4. 解釈文生成
        interpretation = self._generate_interpretation(
            metric_name, ci_results, stable_findings, unstable_findings, group_comparison
        )
        
        return BootstrapResult(
            metric_name=metric_name,
            node_ci=ci_results,
            group_comparison=group_comparison,
            stable_findings=stable_findings,
            unstable_findings=unstable_findings,
            interpretation=interpretation,
            computation_time=computation_time,
            n_bootstrap=self.n_bootstrap,
            alpha=self.alpha
        )
    
    def _compare_groups(self, metric_func: Callable) -> pd.DataFrame:
        """グループ間比較（Permutation検定）"""
        
        if not self.node_groups:
            return pd.DataFrame()
        
        group_names = list(set(self.node_groups.values()))
        
        if len(group_names) < 2:
            return pd.DataFrame()
        
        comparison_results = []
        
        for i, group_a in enumerate(group_names):
            for group_b in group_names[i+1:]:
                nodes_a = [n for n, g in self.node_groups.items() if g == group_a]
                nodes_b = [n for n, g in self.node_groups.items() if g == group_b]
                
                if len(nodes_a) == 0 or len(nodes_b) == 0:
                    continue
                
                perm_result = self.permutation_test(
                    metric_func, nodes_a, nodes_b, n_permutations=500
                )
                
                comparison_results.append({
                    "グループA": group_a,
                    "グループB": group_b,
                    "平均値A": perm_result["group_a_mean"],
                    "平均値B": perm_result["group_b_mean"],
                    "平均値の差": perm_result["observed_diff"],
                    "p値": perm_result["p_value"],
                    "有意性": "✅ 有意" if perm_result["significant"] else "❌ 非有意"
                })
        
        if len(comparison_results) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_results)
        df = df.sort_values(by="p値")
        
        return df
    
    def _generate_interpretation(
        self,
        metric_name: str,
        ci_results: Dict,
        stable_findings: List[str],
        unstable_findings: List[str],
        group_comparison: pd.DataFrame
    ) -> str:
        """平易な日本語の解釈文を生成"""
        
        n_stable = len(stable_findings)
        n_unstable = len(unstable_findings)
        total = len(ci_results)
        
        interpretation = f"""
## 📊 Bootstrap統計検定結果の解釈

### {metric_name}の信頼性評価

**安定性サマリー:**
- 統計的に安定: {n_stable}/{total}ノード ({n_stable/total*100:.1f}%)
- 不安定（再評価推奨）: {n_unstable}/{total}ノード ({n_unstable/total*100:.1f}%)
- リサンプル回数: {self.n_bootstrap}
- 信頼水準: {(1-self.alpha)*100:.0f}%

### ✅ 安定した知見（信頼できる）
以下の{metric_name}値は統計的に安定しており、信頼できます（相対誤差<20%）:

"""
        
        # 上位5件の安定した知見
        for finding in stable_findings[:5]:
            interpretation += f"- {finding}\n"
        
        if len(stable_findings) > 5:
            interpretation += f"... 他{len(stable_findings) - 5}件\n"
        
        if len(unstable_findings) > 0:
            interpretation += f"""
### ⚠️ 不安定な知見（再評価推奨）
以下のノードは信頼区間が広く、再評価が推奨されます:

"""
            for finding in unstable_findings[:3]:
                interpretation += f"- {finding}\n"
        
        # グループ間比較
        if len(group_comparison) > 0:
            interpretation += """
### グループ間比較（Permutation検定）

"""
            significant_comparisons = group_comparison[
                group_comparison['有意性'] == '✅ 有意'
            ]
            
            if len(significant_comparisons) > 0:
                interpretation += f"以下のグループ間には統計的に有意な差があります (p<{self.alpha}):\n\n"
                for _, row in significant_comparisons.iterrows():
                    interpretation += (
                        f"- **{row['グループA']}** vs **{row['グループB']}**: "
                        f"差={row['平均値の差']:.4f}, p={row['p値']:.4f}\n"
                    )
            else:
                interpretation += "グループ間に統計的に有意な差は検出されませんでした。\n"
        
        interpretation += """
### 💡 活用方法

1. **信頼できる知見**: 安定した上位ノードを重点管理対象とする
2. **再評価箇所**: 不安定なノードは追加の評価データを収集
3. **グループ戦略**: 有意差があるグループ間で異なるアプローチを採用
4. **報告資料**: 信頼区間付きで経営層に説得力ある説明が可能

### 📖 信頼区間の見方

- **狭い区間**: データが安定している、信頼性が高い
- **広い区間**: データのばらつきが大きい、追加評価が必要
- **ゼロを含む**: 統計的に有意でない可能性
"""
        
        return interpretation.strip()


def compute_stability_score(ci_results: Dict[str, Tuple[float, float, float]]) -> pd.DataFrame:
    """
    安定性スコアを計算
    
    Args:
        ci_results: {ノード名: (値, 下限, 上限)}
    
    Returns:
        安定性スコアのDataFrame
    """
    stability_data = []
    
    for node, (value, lower, upper) in ci_results.items():
        if abs(value) > 1e-6:
            rel_error = (upper - lower) / (2 * abs(value))
            stability_score = 1 / (1 + rel_error)  # 0-1スケール、高いほど安定
            
            stability_data.append({
                "ノード名": node,
                "値": value,
                "下限": lower,
                "上限": upper,
                "相対誤差": rel_error,
                "安定性スコア": stability_score,
                "判定": "✅ 安定" if rel_error < 0.2 else "⚠️ やや不安定" if rel_error < 0.5 else "❌ 不安定"
            })
    
    df = pd.DataFrame(stability_data)
    df = df.sort_values(by="安定性スコア", ascending=False)
    
    return df
