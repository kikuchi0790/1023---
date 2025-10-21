"""
Bayesian Inference Analysis for Process Insight Modeler
不確実性定量化（Bayesian Inference）

Bootstrap-based Bayesian Approximation（簡易版）:
- 共役事前分布を使用して解析的に事後分布を計算
- Bootstrapリサンプリングで経験的事後分布を近似
- MCMCなしで高速に信用区間を算出
"""

from typing import List, Dict, Tuple, Any, Callable
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
import time
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class BayesianInferenceResult:
    posterior_mean: Dict[Tuple[str, str], float]
    posterior_std: Dict[Tuple[str, str], float]
    credible_intervals: Dict[Tuple[str, str], Tuple[float, float, float]]
    high_uncertainty_edges: List[Tuple[str, str, float]]
    uncertainty_scores: Dict[Tuple[str, str], float]
    prior_params: Dict[str, float]
    computation_time: float
    n_edges: int
    n_bootstrap: int
    credible_level: float
    interpretation: str


class BayesianAnalyzer:
    """
    Bayesian Inference分析クラス
    
    Bootstrap-based Bayesian Approximation:
    - 共役事前分布（正規-逆ガンマ）を使用
    - Bootstrapリサンプリングで事後分布を近似
    - 信用区間の計算と高不確実性エッジの特定
    """
    
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_names: List[str],
        n_bootstrap: int = 1000,
        credible_level: float = 0.95,
        prior_type: str = 'weak_informative'
    ):
        """
        Args:
            adjacency_matrix: 隣接行列（N×N）
            node_names: ノード名リスト
            n_bootstrap: Bootstrapサンプル数
            credible_level: 信用区間レベル（0.90, 0.95, 0.99）
            prior_type: 事前分布タイプ ('weak_informative', 'uninformative')
        """
        self.matrix = adjacency_matrix.copy()
        self.node_names = node_names
        self.n = len(node_names)
        self.n_bootstrap = n_bootstrap
        self.credible_level = credible_level
        self.prior_type = prior_type
        
        self.edges = []
        for i in range(self.n):
            for j in range(self.n):
                if i != j and abs(self.matrix[i, j]) > 1e-6:
                    self.edges.append((self.node_names[i], self.node_names[j]))
        
        self.n_edges = len(self.edges)
        
        self.prior_params = self._initialize_prior()
        
        logger.info(f"BayesianAnalyzer初期化: {self.n}ノード, {self.n_edges}エッジ, Bootstrap={n_bootstrap}")
    
    def _initialize_prior(self) -> Dict[str, float]:
        if self.prior_type == 'weak_informative':
            return {
                'mu_0': 0.0,
                'sigma_0': 5.0,
                'alpha': 2.0,
                'beta': 2.0
            }
        else:
            return {
                'mu_0': 0.0,
                'sigma_0': 100.0,
                'alpha': 0.1,
                'beta': 0.1
            }
    
    def compute_bayesian_inference(
        self,
        progress_callback: Callable[[str, float], None] = None
    ) -> BayesianInferenceResult:
        start_time = time.time()
        
        if self.n_edges == 0:
            return self._create_empty_result(start_time)
        
        if progress_callback:
            progress_callback("Bootstrapリサンプリング中...", 0.0)
        
        bootstrap_samples = self._bootstrap_approximation(progress_callback)
        
        if progress_callback:
            progress_callback("事後分布を計算中...", 0.5)
        
        posterior_mean, posterior_std = self._compute_posterior_distribution(bootstrap_samples)
        
        if progress_callback:
            progress_callback("信用区間を計算中...", 0.7)
        
        credible_intervals = self._compute_credible_intervals(bootstrap_samples)
        
        if progress_callback:
            progress_callback("不確実性スコアを計算中...", 0.85)
        
        uncertainty_scores = self._compute_uncertainty_scores(credible_intervals, posterior_mean)
        high_uncertainty_edges = self._identify_high_uncertainty_edges(uncertainty_scores)
        
        computation_time = time.time() - start_time
        logger.info(f"Bayesian Inference分析完了: {computation_time:.2f}秒")
        
        interpretation = self._generate_interpretation(
            posterior_mean, posterior_std, credible_intervals,
            high_uncertainty_edges, uncertainty_scores
        )
        
        return BayesianInferenceResult(
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            credible_intervals=credible_intervals,
            high_uncertainty_edges=high_uncertainty_edges,
            uncertainty_scores=uncertainty_scores,
            prior_params=self.prior_params,
            computation_time=computation_time,
            n_edges=self.n_edges,
            n_bootstrap=self.n_bootstrap,
            credible_level=self.credible_level,
            interpretation=interpretation
        )
    
    def _bootstrap_approximation(
        self,
        progress_callback: Callable[[str, float], None] = None
    ) -> Dict[Tuple[str, str], np.ndarray]:
        bootstrap_samples = {edge: [] for edge in self.edges}
        
        for b in range(self.n_bootstrap):
            if progress_callback and b % 100 == 0:
                pct = 0.0 + (b / self.n_bootstrap) * 0.5
                progress_callback(f"Bootstrap {b}/{self.n_bootstrap}", pct)
            
            indices = np.random.choice(self.n, size=self.n, replace=True)
            resampled_matrix = self.matrix[np.ix_(indices, indices)]
            
            for edge in self.edges:
                i = self.node_names.index(edge[0])
                j = self.node_names.index(edge[1])
                
                i_resample = np.where(indices == i)[0]
                j_resample = np.where(indices == j)[0]
                
                if len(i_resample) > 0 and len(j_resample) > 0:
                    value = resampled_matrix[i_resample[0], j_resample[0]]
                    bootstrap_samples[edge].append(value)
                else:
                    bootstrap_samples[edge].append(0.0)
        
        for edge in self.edges:
            bootstrap_samples[edge] = np.array(bootstrap_samples[edge])
        
        return bootstrap_samples
    
    def _compute_posterior_distribution(
        self,
        bootstrap_samples: Dict[Tuple[str, str], np.ndarray]
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
        posterior_mean = {}
        posterior_std = {}
        
        mu_0 = self.prior_params['mu_0']
        sigma_0 = self.prior_params['sigma_0']
        alpha = self.prior_params['alpha']
        beta = self.prior_params['beta']
        
        for edge, samples in bootstrap_samples.items():
            n = len(samples)
            sample_mean = np.mean(samples)
            sample_var = np.var(samples, ddof=1) if n > 1 else 1.0
            
            precision_0 = 1.0 / (sigma_0 ** 2)
            precision_data = n / sample_var if sample_var > 1e-6 else n
            
            posterior_precision = precision_0 + precision_data
            posterior_mu = (precision_0 * mu_0 + precision_data * sample_mean) / posterior_precision
            
            posterior_alpha = alpha + n / 2.0
            posterior_beta = beta + 0.5 * np.sum((samples - sample_mean) ** 2)
            
            posterior_sigma = np.sqrt(posterior_beta / (posterior_alpha - 1)) if posterior_alpha > 1 else 1.0
            
            posterior_mean[edge] = posterior_mu
            posterior_std[edge] = posterior_sigma / np.sqrt(posterior_precision)
        
        return posterior_mean, posterior_std
    
    def _compute_credible_intervals(
        self,
        bootstrap_samples: Dict[Tuple[str, str], np.ndarray]
    ) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
        credible_intervals = {}
        
        alpha = (1.0 - self.credible_level) / 2.0
        lower_quantile = alpha
        upper_quantile = 1.0 - alpha
        
        for edge, samples in bootstrap_samples.items():
            mean_val = np.mean(samples)
            lower = np.quantile(samples, lower_quantile)
            upper = np.quantile(samples, upper_quantile)
            
            credible_intervals[edge] = (mean_val, lower, upper)
        
        return credible_intervals
    
    def _compute_uncertainty_scores(
        self,
        credible_intervals: Dict[Tuple[str, str], Tuple[float, float, float]],
        posterior_mean: Dict[Tuple[str, str], float]
    ) -> Dict[Tuple[str, str], float]:
        uncertainty_scores = {}
        
        for edge, (mean_val, lower, upper) in credible_intervals.items():
            interval_width = upper - lower
            
            if abs(mean_val) > 1e-6:
                uncertainty_score = interval_width / abs(mean_val)
            else:
                uncertainty_score = interval_width
            
            uncertainty_scores[edge] = uncertainty_score
        
        return uncertainty_scores
    
    def _identify_high_uncertainty_edges(
        self,
        uncertainty_scores: Dict[Tuple[str, str], float],
        top_k: int = 20
    ) -> List[Tuple[str, str, float]]:
        sorted_edges = sorted(
            [(edge[0], edge[1], score) for edge, score in uncertainty_scores.items()],
            key=lambda x: x[2],
            reverse=True
        )
        
        return sorted_edges[:top_k]
    
    def _create_empty_result(self, start_time: float) -> BayesianInferenceResult:
        return BayesianInferenceResult(
            posterior_mean={},
            posterior_std={},
            credible_intervals={},
            high_uncertainty_edges=[],
            uncertainty_scores={},
            prior_params=self.prior_params,
            computation_time=time.time() - start_time,
            n_edges=0,
            n_bootstrap=self.n_bootstrap,
            credible_level=self.credible_level,
            interpretation="エッジが存在しないため、Bayesian推論を実行できません。"
        )
    
    def _generate_interpretation(
        self,
        posterior_mean: Dict,
        posterior_std: Dict,
        credible_intervals: Dict,
        high_uncertainty_edges: List,
        uncertainty_scores: Dict
    ) -> str:
        n_edges_total = len(posterior_mean)
        
        avg_uncertainty = np.mean(list(uncertainty_scores.values())) if uncertainty_scores else 0
        
        n_high_uncertainty = sum(1 for score in uncertainty_scores.values() if score > 0.5)
        
        credible_pct = int(self.credible_level * 100)
        
        interpretation = f"""
## 🔍 Bayesian Inference分析結果の解釈

### 不確実性定量化のサマリー

- **総エッジ数**: {n_edges_total}
- **平均不確実性スコア**: {avg_uncertainty:.3f}
- **高不確実性エッジ数**: {n_high_uncertainty}（スコア > 0.5）
- **信用区間レベル**: {credible_pct}%

### 💡 不確実性スコアの意味

"""
        
        if avg_uncertainty < 0.3:
            interpretation += """
✅ **平均不確実性 < 0.3: 評価が安定**

ほとんどのエッジの評価は信頼性が高く、再評価の必要性は低いです。
"""
        elif avg_uncertainty < 0.5:
            interpretation += """
⚠️ **平均不確実性 0.3-0.5: やや不安定**

一部のエッジで評価の不確実性が見られます。高スコアのエッジを優先的に確認してください。
"""
        else:
            interpretation += """
❌ **平均不確実性 > 0.5: 不安定**

多くのエッジで評価の不確実性が高いです。再評価または追加データの収集を推奨します。
"""
        
        interpretation += f"""
### 🎯 最も不確実性が高いエッジ（上位5組）

これらのエッジは再評価を推奨します:

"""
        
        for i, (source, target, score) in enumerate(high_uncertainty_edges[:5], 1):
            ci = credible_intervals.get((source, target), (0, 0, 0))
            interpretation += f"{i}. **{source} → {target}**: 不確実性 {score:.3f}, 信用区間 [{ci[1]:.2f}, {ci[2]:.2f}]\n"
        
        interpretation += f"""
### 📊 信用区間とは

**{credible_pct}%信用区間**は、真のスコアがその範囲に含まれる確率が{credible_pct}%であることを意味します。

- 信用区間が狭い = 評価が安定、信頼できる
- 信用区間が広い = 評価が不安定、再評価推奨

### 💡 活用方法

1. **再評価の優先順位決定**
   - 不確実性スコアが高いエッジを優先的に再評価
   - 信用区間が広いエッジは追加の専門家意見を収集

2. **リスク評価**
   - 重要な意思決定に関わるエッジの不確実性を確認
   - 信用区間の下限・上限を考慮したシナリオ分析

3. **データ収集計画**
   - 不確実性が高い箇所について追加データを収集
   - Bootstrapサンプル数を増やして精度向上

### 注意事項

- Bayesian推論は事前分布（{self.prior_type}）を仮定しています
- Bootstrapサンプル数が少ない場合、推定精度が低下します
- エッジ数が非常に多い場合、計算時間が増加する可能性があります
"""
        
        return interpretation.strip()
