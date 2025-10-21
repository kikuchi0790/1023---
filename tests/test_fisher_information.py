"""
Test cases for Fisher Information Analysis
"""

import pytest
import numpy as np
from utils.fisher_information import FisherInformationAnalyzer


class TestFisherInformationAnalyzer:
    """FisherInformationAnalyzerのテスト"""
    
    def test_initialization(self):
        """初期化テスト"""
        matrix = np.array([
            [0, 3, 0],
            [2, 0, 4],
            [0, 1, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = FisherInformationAnalyzer(matrix, nodes)
        
        assert analyzer.n == 3
        assert len(analyzer.node_names) == 3
        assert analyzer.n_edges == 4  # 非ゼロエッジ: A→B, B→A, B→C, C→B
        assert analyzer.noise_variance == 1.0
    
    def test_fisher_matrix_structure(self):
        """Fisher情報行列の構造テスト"""
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [2, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = FisherInformationAnalyzer(matrix, nodes, noise_variance=1.0)
        fisher_matrix = analyzer._compute_fisher_matrix()
        
        # Fisher行列は正方行列
        assert fisher_matrix.shape[0] == fisher_matrix.shape[1]
        assert fisher_matrix.shape[0] == analyzer.n_edges
        
        # Fisher行列は対称行列
        assert np.allclose(fisher_matrix, fisher_matrix.T)
        
        # Fisher行列は半正定値（固有値が非負）
        eigenvalues = np.linalg.eigvalsh(fisher_matrix)
        assert np.all(eigenvalues >= -1e-10)
    
    def test_sensitivity_scores(self):
        """感度スコア計算テスト"""
        matrix = np.array([
            [0, 3, 0],
            [0, 0, 2],
            [1, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = FisherInformationAnalyzer(matrix, nodes)
        fisher_matrix = analyzer._compute_fisher_matrix()
        sensitivity_scores = analyzer._compute_sensitivity_scores(fisher_matrix)
        
        # 全エッジの感度スコアが存在
        assert len(sensitivity_scores) == analyzer.n_edges
        
        # 感度スコアは非負
        for edge, score in sensitivity_scores.items():
            assert score >= 0
            assert isinstance(edge, tuple)
            assert len(edge) == 2
    
    def test_top_sensitive_edges(self):
        """感度上位エッジ抽出テスト"""
        sensitivity_scores = {
            ("A", "B"): 10.0,
            ("B", "C"): 5.0,
            ("C", "A"): 2.0,
            ("A", "C"): 15.0
        }
        
        matrix = np.zeros((3, 3))
        nodes = ["A", "B", "C"]
        analyzer = FisherInformationAnalyzer(matrix, nodes)
        
        top_edges = analyzer._identify_top_sensitive_edges(sensitivity_scores, top_k=2)
        
        # 上位2つが抽出される
        assert len(top_edges) == 2
        
        # 降順でソートされている
        assert top_edges[0][2] == 15.0  # A→C
        assert top_edges[1][2] == 10.0  # A→B
    
    def test_cramer_rao_bounds(self):
        """Cramér-Rao下限計算テスト"""
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [2, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = FisherInformationAnalyzer(matrix, nodes, noise_variance=1.0)
        fisher_matrix = analyzer._compute_fisher_matrix()
        cr_bounds = analyzer._compute_cramer_rao_bounds(fisher_matrix)
        
        # 全エッジのCR下限が存在
        assert len(cr_bounds) == analyzer.n_edges
        
        # CR下限は正値
        for edge, bound in cr_bounds.items():
            assert bound > 0
    
    def test_eigenstructure_analysis(self):
        """固有値分析テスト"""
        matrix = np.array([
            [0, 3, 2],
            [1, 0, 3],
            [2, 1, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = FisherInformationAnalyzer(matrix, nodes)
        fisher_matrix = analyzer._compute_fisher_matrix()
        eigenvalues, condition_number, effective_rank = analyzer._analyze_eigenstructure(fisher_matrix)
        
        # 固有値の数
        assert len(eigenvalues) == analyzer.n_edges
        
        # 固有値は降順
        assert np.all(eigenvalues[:-1] >= eigenvalues[1:])
        
        # 条件数は正値
        assert condition_number > 0
        
        # 実効ランクは0以上
        assert effective_rank >= 0
        assert effective_rank <= analyzer.n_edges
    
    def test_comprehensive_analysis(self):
        """包括的分析のテスト"""
        # 実際のネットワーク
        matrix = np.array([
            [0, 5, 3, 0],
            [2, 0, 4, 1],
            [1, 3, 0, 2],
            [0, 0, 1, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        analyzer = FisherInformationAnalyzer(matrix, nodes, noise_variance=2.0)
        result = analyzer.compute_fisher_information()
        
        # 結果の構造確認
        assert isinstance(result.fisher_matrix, np.ndarray)
        assert isinstance(result.sensitivity_scores, dict)
        assert isinstance(result.top_sensitive_edges, list)
        assert isinstance(result.cramer_rao_bounds, dict)
        assert isinstance(result.condition_number, float)
        assert isinstance(result.effective_rank, (int, np.integer))
        assert isinstance(result.eigenvalues, np.ndarray)
        assert isinstance(result.interpretation, str)
        assert result.computation_time > 0
        
        # エッジ数確認
        assert result.n_edges > 0
        
        # Fisher行列のサイズ
        assert result.fisher_matrix.shape == (result.n_edges, result.n_edges)
        
        # 感度スコア数
        assert len(result.sensitivity_scores) == result.n_edges
        
        # CR下限数
        assert len(result.cramer_rao_bounds) == result.n_edges
    
    def test_empty_graph(self):
        """空グラフのテスト（エッジなし）"""
        matrix = np.zeros((3, 3))
        nodes = ["A", "B", "C"]
        
        analyzer = FisherInformationAnalyzer(matrix, nodes)
        result = analyzer.compute_fisher_information()
        
        # エラーなく完了
        assert result is not None
        
        # エッジ数0
        assert result.n_edges == 0
        
        # 空の結果
        assert len(result.sensitivity_scores) == 0
        assert len(result.top_sensitive_edges) == 0
        assert len(result.cramer_rao_bounds) == 0
    
    def test_noise_variance_effect(self):
        """ノイズ分散の影響テスト"""
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [2, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        # σ² = 1.0
        analyzer1 = FisherInformationAnalyzer(matrix, nodes, noise_variance=1.0)
        fisher1 = analyzer1._compute_fisher_matrix()
        
        # σ² = 4.0
        analyzer2 = FisherInformationAnalyzer(matrix, nodes, noise_variance=4.0)
        fisher2 = analyzer2._compute_fisher_matrix()
        
        # Fisher情報はσ²に反比例
        # I(θ; σ²=1) = 4 × I(θ; σ²=4)
        # 対角要素のみ比較（非対角要素は0の可能性がある）
        for i in range(fisher1.shape[0]):
            ratio = fisher1[i, i] / fisher2[i, i]
            assert abs(ratio - 4.0) < 0.01
    
    def test_singular_matrix_handling(self):
        """特異行列の処理テスト"""
        # 同じ行に複数の非ゼロエッジ（相関あり）
        matrix = np.array([
            [0, 5, 3, 2],
            [5, 0, 3, 2],
            [5, 3, 0, 2],
            [5, 3, 2, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        analyzer = FisherInformationAnalyzer(matrix, nodes)
        result = analyzer.compute_fisher_information()
        
        # エラーなく完了
        assert result is not None
        
        # 条件数が計算される（有限または無限大）
        assert result.condition_number > 0 or np.isinf(result.condition_number)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
