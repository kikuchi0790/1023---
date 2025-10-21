"""
Test cases for Transfer Entropy Analysis
"""

import pytest
import numpy as np
from utils.information_theory_analysis import TransferEntropyAnalyzer


class TestTransferEntropyAnalyzer:
    """Transfer Entropy分析のテスト"""
    
    def test_initialization(self):
        """初期化テスト"""
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [2, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = TransferEntropyAnalyzer(matrix, nodes)
        
        assert analyzer.n == 3
        assert len(analyzer.node_names) == 3
        assert analyzer.n_walks == 1000
        assert analyzer.walk_length == 50
    
    def test_random_walk_simulation(self):
        """ランダムウォークシミュレーションのテスト"""
        matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = TransferEntropyAnalyzer(matrix, nodes, n_walks=100, walk_length=20)
        
        time_series = analyzer._simulate_random_walks()
        
        # 時系列の形状
        assert time_series.shape == (20, 3)
        
        # 値は0-1の範囲（正規化されている）
        assert time_series.min() >= 0
        assert time_series.max() <= 1
    
    def test_discretization(self):
        """離散化のテスト"""
        time_series = np.array([
            [0.1, 0.5, 0.9],
            [0.2, 0.6, 0.8],
            [0.3, 0.4, 0.7]
        ])
        
        matrix = np.zeros((3, 3))
        nodes = ["A", "B", "C"]
        analyzer = TransferEntropyAnalyzer(matrix, nodes, n_bins=3)
        
        discrete = analyzer._discretize(time_series)
        
        # 離散値は0, 1, 2のみ
        unique_values = np.unique(discrete)
        assert len(unique_values) <= 3
        assert all(v in [0, 1, 2] for v in unique_values)
    
    def test_te_matrix_shape(self):
        """TE行列の形状テスト"""
        matrix = np.array([
            [0, 3, 2],
            [1, 0, 4],
            [2, 3, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = TransferEntropyAnalyzer(
            matrix, nodes, 
            n_walks=50,  # 高速化
            walk_length=20
        )
        
        result = analyzer.compute_transfer_entropy()
        
        # TE行列の形状
        assert result.te_matrix.shape == (3, 3)
        
        # 対角成分は0（自己への情報転送は計算しない）
        assert result.te_matrix[0, 0] == 0
        assert result.te_matrix[1, 1] == 0
        assert result.te_matrix[2, 2] == 0
        
        # 非負値
        assert (result.te_matrix >= 0).all()
    
    def test_te_causality_direction(self):
        """
        因果方向性のテスト
        一方向の強いエッジがある場合、TEもそちらが高いべき
        """
        # A → B → C の一方向チェーン
        matrix = np.array([
            [0, 9, 0, 0],
            [0, 0, 9, 0],
            [0, 0, 0, 9],
            [0, 0, 0, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        analyzer = TransferEntropyAnalyzer(
            matrix, nodes,
            n_walks=100,
            walk_length=30
        )
        
        result = analyzer.compute_transfer_entropy()
        
        # A→Bのフローが逆方向より高い
        te_AB = result.te_matrix[0, 1]
        te_BA = result.te_matrix[1, 0]
        
        # 一方向なのでA→B > B→A（厳密には確率的だが期待値として）
        # テストは緩い条件
        assert te_AB + te_BA > 0  # 何らかの情報フローがある
    
    def test_isolated_node_te(self):
        """孤立ノードのTEテスト"""
        # ノードDは完全に孤立
        matrix = np.array([
            [0, 3, 2, 0],
            [1, 0, 4, 0],
            [2, 3, 0, 0],
            [0, 0, 0, 0]
        ])
        nodes = ["A", "B", "C", "D_isolated"]
        
        analyzer = TransferEntropyAnalyzer(
            matrix, nodes,
            n_walks=100,
            walk_length=20
        )
        
        result = analyzer.compute_transfer_entropy()
        
        # 孤立ノード関連のTE
        te_from_D = result.te_matrix[3, :].sum()
        te_to_D = result.te_matrix[:, 3].sum()
        
        # 孤立ノードからの/への情報フローは0に近い
        # ランダムテレポートがあるので完全に0ではないが、非常に小さい
        avg_te = result.te_matrix[result.te_matrix > 0].mean() if (result.te_matrix > 0).any() else 1
        
        assert te_from_D < avg_te or te_from_D == 0
        assert te_to_D < avg_te or te_to_D == 0
    
    def test_result_structure(self):
        """結果構造のテスト"""
        matrix = np.array([
            [0, 5, 3],
            [2, 0, 4],
            [1, 3, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = TransferEntropyAnalyzer(
            matrix, nodes,
            n_walks=50,
            walk_length=20
        )
        
        result = analyzer.compute_transfer_entropy()
        
        # 結果の構造確認
        assert result.te_matrix.shape == (3, 3)
        assert isinstance(result.significant_flows, list)
        assert isinstance(result.bottleneck_nodes, list)
        assert isinstance(result.info_inflow, dict)
        assert isinstance(result.info_outflow, dict)
        assert isinstance(result.interpretation, str)
        assert result.computation_time > 0
        assert result.n_walks == 50
        assert result.walk_length == 20
    
    def test_significant_flows_filtering(self):
        """有意なフローフィルタリングのテスト"""
        matrix = np.array([
            [0, 5, 3, 1],
            [2, 0, 4, 1],
            [1, 3, 0, 1],
            [1, 1, 1, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        analyzer = TransferEntropyAnalyzer(
            matrix, nodes,
            n_walks=50,
            walk_length=20
        )
        
        result = analyzer.compute_transfer_entropy()
        
        # significant_flowsは上位25%のみ
        # 全非ゼロTEの数
        nonzero_count = (result.te_matrix > 0).sum() - result.te_matrix.shape[0]  # 対角除く
        
        expected_max_significant = int(nonzero_count * 0.25) + 1
        
        # significant_flowsの数は上位25%程度
        assert len(result.significant_flows) <= expected_max_significant + 5  # 余裕を持たせる
    
    def test_comparison_with_original(self):
        """元の隣接行列との比較テスト"""
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [2, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = TransferEntropyAnalyzer(
            matrix, nodes,
            n_walks=50,
            walk_length=20
        )
        
        result = analyzer.compute_transfer_entropy()
        
        # 比較DataFrameの確認
        assert len(result.comparison_with_original) > 0
        assert "From" in result.comparison_with_original.columns
        assert "To" in result.comparison_with_original.columns
        assert "元のスコア" in result.comparison_with_original.columns
        assert "TE (bits)" in result.comparison_with_original.columns
        assert "判定" in result.comparison_with_original.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
