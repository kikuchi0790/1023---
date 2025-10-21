"""
Fisher Information Analysis for Process Insight Modeler
感度分析（Fisher Information Matrix）

どのスコアが不正確だと全体が大きく歪むかを特定し、推定精度の理論限界を計算する。
"""

from typing import List, Dict, Tuple, Any, Callable
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class FisherInformationResult:
    """Fisher Information分析の結果"""
    fisher_matrix: np.ndarray  # Fisher情報行列
    sensitivity_scores: Dict[Tuple[str, str], float]  # エッジごとの感度スコア
    top_sensitive_edges: List[Tuple[str, str, float]]  # 感度が高いエッジ（上位）
    cramer_rao_bounds: Dict[Tuple[str, str], float]  # Cramér-Rao下限
    condition_number: float  # 条件数（数値安定性）
    effective_rank: int  # 実効ランク
    eigenvalues: np.ndarray  # 固有値
    computation_time: float
    n_edges: int
    interpretation: str


class FisherInformationAnalyzer:
    """
    Fisher Information分析クラス
    
    Fisher情報行列を計算し、パラメータ推定の感度と精度限界を評価:
    - Fisher情報 I(θ) = E[(∂logL/∂θ)²]
    - Cramér-Rao下限: Var(θ̂) ≥ [I(θ)]⁻¹
    - 感度分析: どのパラメータが全体に大きく影響するか
    """
    
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_names: List[str],
        noise_variance: float = 1.0
    ):
        """
        Args:
            adjacency_matrix: 隣接行列（N×N）
            node_names: ノード名リスト
            noise_variance: 観測ノイズの分散（σ²）
        """
        self.matrix = adjacency_matrix.copy()
        self.node_names = node_names
        self.n = len(node_names)
        self.noise_variance = noise_variance
        
        # 非ゼロエッジのリスト
        self.edges = []
        for i in range(self.n):
            for j in range(self.n):
                if i != j and abs(self.matrix[i, j]) > 1e-6:
                    self.edges.append((self.node_names[i], self.node_names[j]))
        
        self.n_edges = len(self.edges)
        
        logger.info(f"FisherInformationAnalyzer初期化: {self.n}ノード, {self.n_edges}エッジ")
    
    def compute_fisher_information(
        self,
        progress_callback: Callable[[str, float], None] = None
    ) -> FisherInformationResult:
        """
        包括的なFisher Information分析を実行
        
        Args:
            progress_callback: 進捗コールバック(message, pct)
        
        Returns:
            FisherInformationResult
        """
        start_time = time.time()
        
        if self.n_edges == 0:
            # エッジがない場合
            return self._create_empty_result(start_time)
        
        # 1. Fisher情報行列を計算
        if progress_callback:
            progress_callback("Fisher情報行列を計算中...", 0.0)
        
        fisher_matrix = self._compute_fisher_matrix()
        
        # 2. 感度スコアを計算
        if progress_callback:
            progress_callback("感度スコアを計算中...", 0.3)
        
        sensitivity_scores = self._compute_sensitivity_scores(fisher_matrix)
        top_sensitive_edges = self._identify_top_sensitive_edges(sensitivity_scores)
        
        # 3. Cramér-Rao下限を計算
        if progress_callback:
            progress_callback("Cramér-Rao下限を計算中...", 0.6)
        
        cramer_rao_bounds = self._compute_cramer_rao_bounds(fisher_matrix)
        
        # 4. 固有値分析
        if progress_callback:
            progress_callback("固有値分析中...", 0.8)
        
        eigenvalues, condition_number, effective_rank = self._analyze_eigenstructure(fisher_matrix)
        
        computation_time = time.time() - start_time
        logger.info(f"Fisher Information分析完了: {computation_time:.2f}秒")
        
        # 5. 解釈文生成
        interpretation = self._generate_interpretation(
            sensitivity_scores, top_sensitive_edges, cramer_rao_bounds,
            condition_number, effective_rank
        )
        
        return FisherInformationResult(
            fisher_matrix=fisher_matrix,
            sensitivity_scores=sensitivity_scores,
            top_sensitive_edges=top_sensitive_edges,
            cramer_rao_bounds=cramer_rao_bounds,
            condition_number=condition_number,
            effective_rank=effective_rank,
            eigenvalues=eigenvalues,
            computation_time=computation_time,
            n_edges=self.n_edges,
            interpretation=interpretation
        )
    
    def _compute_fisher_matrix(self) -> np.ndarray:
        """
        Fisher情報行列を計算
        
        ガウシアン尤度を仮定:
        I(θ) = (1/σ²) A^T A
        
        ここで A は設計行列（エッジの影響を表現）
        
        Returns:
            Fisher情報行列（n_edges × n_edges）
        """
        # 設計行列 A の構築
        # 各エッジがシステム全体に与える影響を表現
        A = np.zeros((self.n * self.n, self.n_edges))
        
        for k, (source, target) in enumerate(self.edges):
            i = self.node_names.index(source)
            j = self.node_names.index(target)
            
            # エッジ k の影響は位置 (i, j) に現れる
            idx = i * self.n + j
            A[idx, k] = 1.0
        
        # Fisher情報行列: I(θ) = (1/σ²) A^T A
        fisher_matrix = (1.0 / self.noise_variance) * (A.T @ A)
        
        return fisher_matrix
    
    def _compute_sensitivity_scores(
        self,
        fisher_matrix: np.ndarray
    ) -> Dict[Tuple[str, str], float]:
        """
        各エッジの感度スコアを計算
        
        感度スコア = Fisher情報行列の対角要素
        大きいほど、そのパラメータの推定が全体に大きく影響
        
        Returns:
            {(source, target): sensitivity}
        """
        sensitivity_scores = {}
        
        for k, edge in enumerate(self.edges):
            # 対角要素 = そのパラメータの Fisher 情報量
            sensitivity = fisher_matrix[k, k]
            sensitivity_scores[edge] = sensitivity
        
        return sensitivity_scores
    
    def _identify_top_sensitive_edges(
        self,
        sensitivity_scores: Dict[Tuple[str, str], float],
        top_k: int = 20
    ) -> List[Tuple[str, str, float]]:
        """
        感度が最も高いエッジを特定
        
        Returns:
            [(source, target, sensitivity), ...] 降順
        """
        sorted_edges = sorted(
            [(edge[0], edge[1], score) for edge, score in sensitivity_scores.items()],
            key=lambda x: x[2],
            reverse=True
        )
        
        return sorted_edges[:top_k]
    
    def _compute_cramer_rao_bounds(
        self,
        fisher_matrix: np.ndarray
    ) -> Dict[Tuple[str, str], float]:
        """
        Cramér-Rao下限を計算
        
        CR下限 = [I(θ)]⁻¹ の対角要素
        これは推定精度の理論限界を表す
        
        Returns:
            {(source, target): cr_bound}
        """
        cramer_rao_bounds = {}
        
        try:
            # Fisher情報行列の逆行列
            fisher_inv = np.linalg.inv(fisher_matrix)
            
            for k, edge in enumerate(self.edges):
                # CR下限 = 逆行列の対角要素
                cr_bound = fisher_inv[k, k]
                cramer_rao_bounds[edge] = cr_bound
        
        except np.linalg.LinAlgError:
            # 特異行列の場合は疑似逆行列を使用
            logger.warning("Fisher行列が特異です。疑似逆行列を使用します。")
            fisher_pinv = np.linalg.pinv(fisher_matrix)
            
            for k, edge in enumerate(self.edges):
                cr_bound = fisher_pinv[k, k]
                cramer_rao_bounds[edge] = cr_bound
        
        return cramer_rao_bounds
    
    def _analyze_eigenstructure(
        self,
        fisher_matrix: np.ndarray
    ) -> Tuple[np.ndarray, float, int]:
        """
        Fisher情報行列の固有値構造を分析
        
        Returns:
            (eigenvalues, condition_number, effective_rank)
        """
        # 固有値を計算（対称行列なので実数）
        eigenvalues = np.linalg.eigvalsh(fisher_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 降順
        
        # 条件数 = 最大固有値 / 最小固有値（正の固有値のみ）
        positive_eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        if len(positive_eigenvalues) > 0:
            condition_number = positive_eigenvalues[0] / positive_eigenvalues[-1]
        else:
            condition_number = np.inf
        
        # 実効ランク（有意な正の固有値の数）
        effective_rank = np.sum(eigenvalues > 1e-6)
        
        return eigenvalues, condition_number, effective_rank
    
    def _create_empty_result(self, start_time: float) -> FisherInformationResult:
        """エッジがない場合の空の結果を作成"""
        return FisherInformationResult(
            fisher_matrix=np.array([[]]),
            sensitivity_scores={},
            top_sensitive_edges=[],
            cramer_rao_bounds={},
            condition_number=0.0,
            effective_rank=0,
            eigenvalues=np.array([]),
            computation_time=time.time() - start_time,
            n_edges=0,
            interpretation="エッジが存在しないため、Fisher情報分析を実行できません。"
        )
    
    def _generate_interpretation(
        self,
        sensitivity_scores: Dict,
        top_sensitive_edges: List,
        cramer_rao_bounds: Dict,
        condition_number: float,
        effective_rank: int
    ) -> str:
        """平易な日本語の解釈文を生成"""
        
        n_edges_total = len(sensitivity_scores)
        
        # 平均感度
        avg_sensitivity = np.mean(list(sensitivity_scores.values())) if sensitivity_scores else 0
        
        # CR下限の統計
        cr_values = list(cramer_rao_bounds.values()) if cramer_rao_bounds else []
        avg_cr_bound = np.mean(cr_values) if cr_values else 0
        
        interpretation = f"""
## 🔍 Fisher Information分析結果の解釈

### 感度分析のサマリー

- **総エッジ数**: {n_edges_total}
- **実効ランク**: {effective_rank}（独立な情報の数）
- **条件数**: {condition_number:.2f}

### 💡 条件数の意味

"""
        
        if condition_number < 10:
            interpretation += """
✅ **条件数 < 10: 数値的に安定**

推定問題は良好な条件にあります。すべてのエッジの推定が安定しています。
"""
        elif condition_number < 100:
            interpretation += """
⚠️ **条件数 10-100: やや不安定**

一部のエッジ間に強い相関があります。推定には注意が必要です。
"""
        else:
            interpretation += """
❌ **条件数 > 100: 数値的に不安定**

推定問題が病的です。一部のエッジは正確に推定できない可能性があります。
多重共線性が存在する可能性が高いです。
"""
        
        interpretation += f"""
### 🎯 最も感度が高いエッジ（上位5組）

これらのエッジが不正確だと、全体の推定が大きく歪みます:

"""
        
        for i, (source, target, sensitivity) in enumerate(top_sensitive_edges[:5], 1):
            interpretation += f"{i}. **{source} → {target}**: 感度 {sensitivity:.4f}\n"
        
        interpretation += f"""
### 📊 Cramér-Rao下限とは

**CR下限**は、どんなに優れた推定手法を使っても達成できない精度の理論限界です。

- 平均CR下限: {avg_cr_bound:.4f}
- CR下限が小さい = より正確に推定可能
- CR下限が大きい = 推定が本質的に困難

### 💡 活用方法

1. **再評価の優先順位決定**
   - 感度が高いエッジを優先的に再評価
   - CR下限が大きいエッジは慎重に評価

2. **データ収集計画**
   - 感度が高いエッジについて追加データを収集
   - CR下限を小さくするための実験設計

3. **信頼性評価**
   - 条件数が大きい場合は、推定結果の解釈に注意
   - 多重共線性がある場合は、モデルの簡素化を検討

### 注意事項

- Fisher情報はガウシアン尤度を仮定しています
- 真のノイズ分散が仮定値（σ² = {self.noise_variance}）と異なる場合、結果は調整が必要です
- 条件数が非常に大きい場合、正則化手法の適用を検討してください
"""
        
        return interpretation.strip()
