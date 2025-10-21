"""
Transfer Entropy Analysis for Process Insight Modeler
情報フロー分析（Transfer Entropy）

因果的な情報の流れを定量化する。
単なる相関ではなく、「XがYに何bits情報を伝えているか」を測定。
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
class TransferEntropyResult:
    """Transfer Entropy分析の結果"""
    te_matrix: np.ndarray  # Transfer Entropy行列 (N×N)
    significant_flows: List[Tuple[str, str, float]]  # (source, target, TE値)
    bottleneck_nodes: List[str]  # 情報ボトルネックノード
    info_inflow: Dict[str, float]  # ノードごとの情報流入量
    info_outflow: Dict[str, float]  # ノードごとの情報流出量
    comparison_with_original: pd.DataFrame  # 隣接行列との比較
    interpretation: str  # 平易な解釈文
    computation_time: float
    n_walks: int
    walk_length: int


class TransferEntropyAnalyzer:
    """
    Transfer Entropy分析クラス
    
    ランダムウォークシミュレーションで擬似時系列を生成し、
    Transfer Entropyを計算して因果的な情報フローを可視化。
    """
    
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_names: List[str],
        n_walks: int = 1000,
        walk_length: int = 50,
        n_bins: int = 3
    ):
        """
        Args:
            adjacency_matrix: 隣接行列（N×N）
            node_names: ノード名リスト
            n_walks: ランダムウォーク回数
            walk_length: 各ウォークの長さ（時系列の長さ）
            n_bins: 離散化ビン数（2=低/高、3=低/中/高）
        """
        self.matrix = adjacency_matrix.copy()
        self.node_names = node_names
        self.n = len(node_names)
        self.n_walks = n_walks
        self.walk_length = walk_length
        self.n_bins = n_bins
        
        logger.info(f"TransferEntropyAnalyzer初期化: {self.n}ノード, {n_walks}ウォーク×{walk_length}長")
    
    def compute_transfer_entropy(
        self,
        progress_callback: Callable[[str, float], None] = None
    ) -> TransferEntropyResult:
        """
        Transfer Entropyを計算
        
        Args:
            progress_callback: 進捗コールバック(message, progress_pct)
        
        Returns:
            TransferEntropyResult
        """
        start_time = time.time()
        
        # 1. ランダムウォークシミュレーション
        if progress_callback:
            progress_callback("ランダムウォークシミュレーション実行中...", 0.0)
        
        time_series = self._simulate_random_walks()
        logger.info(f"時系列生成完了: shape={time_series.shape}")
        
        # 2. 離散化
        if progress_callback:
            progress_callback("時系列データを離散化中...", 0.3)
        
        discrete_series = self._discretize(time_series)
        
        # 3. Transfer Entropy計算
        if progress_callback:
            progress_callback("Transfer Entropy計算中...", 0.5)
        
        te_matrix = self._compute_te_matrix(discrete_series, progress_callback)
        logger.info(f"TE行列計算完了: 平均TE={te_matrix.mean():.4f} bits")
        
        # 4. 統計的有意性フィルタリング
        if progress_callback:
            progress_callback("有意な情報フローを抽出中...", 0.8)
        
        significant_flows = self._filter_significant(te_matrix)
        
        # 5. ボトルネック検出
        bottleneck_nodes = self._detect_bottlenecks(te_matrix)
        
        # 6. 情報流入/流出量
        info_inflow = {
            self.node_names[j]: te_matrix[:, j].sum()
            for j in range(self.n)
        }
        info_outflow = {
            self.node_names[i]: te_matrix[i, :].sum()
            for i in range(self.n)
        }
        
        # 7. 元の隣接行列と比較
        if progress_callback:
            progress_callback("元の隣接行列と比較中...", 0.9)
        
        comparison = self._compare_with_original(te_matrix)
        
        computation_time = time.time() - start_time
        logger.info(f"Transfer Entropy計算完了: {computation_time:.2f}秒")
        
        # 解釈文生成
        interpretation = self._generate_interpretation(
            significant_flows, bottleneck_nodes, comparison, te_matrix
        )
        
        return TransferEntropyResult(
            te_matrix=te_matrix,
            significant_flows=significant_flows,
            bottleneck_nodes=bottleneck_nodes,
            info_inflow=info_inflow,
            info_outflow=info_outflow,
            comparison_with_original=comparison,
            interpretation=interpretation,
            computation_time=computation_time,
            n_walks=self.n_walks,
            walk_length=self.walk_length
        )
    
    def _simulate_random_walks(self) -> np.ndarray:
        """
        重み付きランダムウォークでノード活性化時系列を生成
        
        Returns:
            time_series: (T, N) - 時刻T×ノードNの活性化度行列
        """
        activation_counts = np.zeros((self.walk_length, self.n))
        
        # 絶対値で重み付け（負のスコアも影響として扱う）
        abs_matrix = np.abs(self.matrix)
        
        for walk_idx in range(self.n_walks):
            # ランダムな開始ノード
            current_node = np.random.randint(0, self.n)
            
            for t in range(self.walk_length):
                # 時刻tでcurrent_nodeが活性化
                activation_counts[t, current_node] += 1
                
                # 次のノードへ遷移
                out_weights = abs_matrix[current_node, :]
                
                if out_weights.sum() > 0:
                    # 重みに基づく確率的遷移
                    probs = out_weights / out_weights.sum()
                    current_node = np.random.choice(self.n, p=probs)
                else:
                    # 出次数がない場合はランダムテレポート
                    current_node = np.random.randint(0, self.n)
        
        # 正規化
        time_series = activation_counts / self.n_walks
        
        return time_series
    
    def _discretize(self, time_series: np.ndarray) -> np.ndarray:
        """
        連続値を離散カテゴリに変換
        
        n_bins=3の場合: 低(0)、中(1)、高(2)
        """
        discrete = np.zeros_like(time_series, dtype=int)
        
        for node_idx in range(self.n):
            series = time_series[:, node_idx]
            
            # パーセンタイルで区切る
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            thresholds = [np.percentile(series, p) for p in percentiles[1:-1]]
            
            discrete[:, node_idx] = np.digitize(series, thresholds)
        
        return discrete
    
    def _compute_te_matrix(
        self, 
        discrete_series: np.ndarray,
        progress_callback: Callable = None
    ) -> np.ndarray:
        """
        離散時系列からTransfer Entropy行列を計算
        
        TE(X→Y) = I(Y_future; X_past | Y_past)
        """
        te_matrix = np.zeros((self.n, self.n))
        
        total_pairs = self.n * (self.n - 1)
        computed = 0
        
        for i in range(self.n):  # source
            for j in range(self.n):  # target
                if i == j:
                    continue
                
                # 時系列データ
                X_past = discrete_series[:-1, i]
                Y_past = discrete_series[:-1, j]
                Y_future = discrete_series[1:, j]
                
                # 条件付き相互情報量
                te_value = self._conditional_mutual_info(Y_future, X_past, Y_past)
                te_matrix[i, j] = max(0, te_value)  # 負の値は0に
                
                computed += 1
                if progress_callback and computed % 50 == 0:
                    pct = 0.5 + 0.3 * (computed / total_pairs)
                    progress_callback(f"TE計算中... {computed}/{total_pairs}", pct)
        
        return te_matrix
    
    def _conditional_mutual_info(
        self,
        Y_future: np.ndarray,
        X_past: np.ndarray,
        Y_past: np.ndarray
    ) -> float:
        """
        条件付き相互情報量 I(Y_future; X_past | Y_past)
        
        = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
        """
        # 同時出現カウント
        joint_counts = {}
        total = len(Y_future)
        
        for yf, x, yp in zip(Y_future, X_past, Y_past):
            key = (int(yf), int(x), int(yp))
            joint_counts[key] = joint_counts.get(key, 0) + 1
        
        # 周辺分布
        p_yf_yp_x = {}  # P(Y_future, Y_past, X_past)
        p_yf_yp = {}    # P(Y_future, Y_past)
        p_yp = {}       # P(Y_past)
        p_yp_x = {}     # P(Y_past, X_past)
        
        for (yf, x, yp), count in joint_counts.items():
            p = count / total
            p_yf_yp_x[(yf, yp, x)] = p
            
            key_yf_yp = (yf, yp)
            p_yf_yp[key_yf_yp] = p_yf_yp.get(key_yf_yp, 0) + p
            
            p_yp[yp] = p_yp.get(yp, 0) + p
            
            key_yp_x = (yp, x)
            p_yp_x[key_yp_x] = p_yp_x.get(key_yp_x, 0) + p
        
        # I(Y_future; X_past | Y_past)
        mi = 0.0
        epsilon = 1e-10
        
        for (yf, yp, x), p_joint in p_yf_yp_x.items():
            if p_joint > 0:
                p_yp_x_val = p_yp_x.get((yp, x), epsilon)
                p_yf_yp_val = p_yf_yp.get((yf, yp), epsilon)
                p_yp_val = p_yp.get(yp, epsilon)
                
                # P(Y_f | Y_p, X_p)
                p_yf_given_yp_x = p_joint / p_yp_x_val
                
                # P(Y_f | Y_p)
                p_yf_given_yp = p_yf_yp_val / p_yp_val
                
                if p_yf_given_yp_x > epsilon and p_yf_given_yp > epsilon:
                    mi += p_joint * np.log2(p_yf_given_yp_x / p_yf_given_yp)
        
        return mi
    
    def _filter_significant(
        self,
        te_matrix: np.ndarray,
        threshold_percentile: float = 75
    ) -> List[Tuple[str, str, float]]:
        """
        統計的に有意な情報フローのみ抽出（上位25%）
        """
        nonzero_te = te_matrix[te_matrix > 0]
        
        if len(nonzero_te) == 0:
            return []
        
        threshold = np.percentile(nonzero_te, threshold_percentile)
        
        significant = []
        for i in range(self.n):
            for j in range(self.n):
                if te_matrix[i, j] >= threshold:
                    significant.append((
                        self.node_names[i],
                        self.node_names[j],
                        te_matrix[i, j]
                    ))
        
        significant.sort(key=lambda x: x[2], reverse=True)
        
        return significant
    
    def _detect_bottlenecks(self, te_matrix: np.ndarray) -> List[str]:
        """
        情報ボトルネックノードの検出
        
        多くの情報が集中・経由するノード
        """
        inflow = te_matrix.sum(axis=0)   # 列の合計（流入）
        outflow = te_matrix.sum(axis=1)  # 行の合計（流出）
        
        # ボトルネックスコア = 流入 × 流出
        bottleneck_scores = inflow * outflow
        
        if bottleneck_scores.sum() == 0:
            return []
        
        # 上位25%
        threshold = np.percentile(bottleneck_scores, 75)
        
        bottleneck_indices = np.where(bottleneck_scores >= threshold)[0]
        bottleneck_nodes = [self.node_names[i] for i in bottleneck_indices]
        
        return bottleneck_nodes
    
    def _compare_with_original(self, te_matrix: np.ndarray) -> pd.DataFrame:
        """
        Transfer Entropy vs 元の隣接行列スコアの比較
        """
        comparison_data = []
        
        # 正規化のための最大値
        orig_max = np.abs(self.matrix).max()
        te_max = te_matrix.max()
        
        if orig_max == 0:
            orig_max = 1.0
        if te_max == 0:
            te_max = 1.0
        
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                
                original_score = self.matrix[i, j]
                te_score = te_matrix[i, j]
                
                # どちらかが非ゼロなら記録
                if abs(original_score) > 0 or te_score > 0:
                    orig_norm = abs(original_score) / orig_max
                    te_norm = te_score / te_max
                    
                    diff = te_norm - orig_norm
                    
                    comparison_data.append({
                        "From": self.node_names[i],
                        "To": self.node_names[j],
                        "元のスコア": original_score,
                        "TE (bits)": te_score,
                        "元のスコア正規化": orig_norm,
                        "TE正規化": te_norm,
                        "差分": diff,
                        "判定": self._classify_discrepancy(orig_norm, te_norm)
                    })
        
        df = pd.DataFrame(comparison_data)
        if len(df) > 0:
            df = df.sort_values(by="差分", ascending=False)
        
        return df
    
    def _classify_discrepancy(self, orig_norm: float, te_norm: float) -> str:
        """差分の分類"""
        diff = te_norm - orig_norm
        
        if diff > 0.2:
            return "⬆️ 隠れた影響（因果 > 評価）"
        elif diff < -0.2:
            return "⬇️ 見かけの相関（因果 < 評価）"
        else:
            return "✅ 一致"
    
    def _generate_interpretation(
        self,
        significant_flows: List[Tuple[str, str, float]],
        bottleneck_nodes: List[str],
        comparison: pd.DataFrame,
        te_matrix: np.ndarray
    ) -> str:
        """平易な日本語の解釈文を生成"""
        
        if len(significant_flows) == 0:
            return "有意な情報フローが検出されませんでした。ネットワークの接続性が低い可能性があります。"
        
        # 最強の情報フロー
        top_flow = significant_flows[0]
        
        # 見かけの相関（上位3件）
        overestimated = comparison[comparison["判定"] == "⬇️ 見かけの相関（因果 < 評価）"].head(3)
        overestimated_text = "\n".join([
            f"- 「{row['From']}」→「{row['To']}」: 評価スコア{row['元のスコア']:.1f} だが TE {row['TE (bits)']:.3f} bits"
            for _, row in overestimated.iterrows()
        ]) if len(overestimated) > 0 else "該当なし"
        
        # 隠れた影響（上位3件）
        underestimated = comparison[comparison["判定"] == "⬆️ 隠れた影響（因果 > 評価）"].head(3)
        underestimated_text = "\n".join([
            f"- 「{row['From']}」→「{row['To']}」: 評価スコア{row['元のスコア']:.1f} だが TE {row['TE (bits)']:.3f} bits"
            for _, row in underestimated.iterrows()
        ]) if len(underestimated) > 0 else "該当なし"
        
        # 平均TE
        avg_te = te_matrix[te_matrix > 0].mean() if (te_matrix > 0).any() else 0
        
        interpretation = f"""
## 📡 Transfer Entropy分析結果の解釈

### 最強の情報フロー
**「{top_flow[0]}」から「{top_flow[1]}」**へ **{top_flow[2]:.3f} bits**の情報が流れています。
これは因果的な影響を示しており、単なる相関ではありません。

### 情報ボトルネック
以下のノードは多くの情報が集中・経由する重要な中継点です:
{', '.join(f'「{node}」' for node in bottleneck_nodes[:5]) if bottleneck_nodes else '検出されませんでした'}

これらのノードが停止すると、情報の流れが遮断されます。

### 相関 vs 因果の乖離

#### ⬇️ 見かけの相関（評価過大、実際の因果影響は低い）
{overestimated_text}

#### ⬆️ 隠れた影響（評価過小、実際の因果影響は高い）
{underestimated_text}

### 統計情報
- 有意な情報フロー数: {len(significant_flows)}
- 平均Transfer Entropy: {avg_te:.3f} bits
- ボトルネックノード数: {len(bottleneck_nodes)}

### 💡 活用方法
1. **真のボトルネック特定**: TE上位のエッジとノードを優先改善
2. **コミュニケーション設計**: 情報フローが弱い箇所に報連相の仕組みを構築
3. **評価の見直し**: 「見かけの相関」ペアは実際の因果影響が弱い可能性
4. **隠れた依存関係**: 「隠れた影響」ペアは見落としていた重要な依存関係
"""
        
        return interpretation.strip()
