"""
Test cases for Shapley Value Analysis
"""

import pytest
import numpy as np
import networkx as nx
from utils.shapley_analysis import ShapleyAnalyzer, compute_shapley_coalition_stability


class TestShapleyAnalyzer:
    """Shapley Value分析のテスト"""
    
    def test_initialization(self):
        """初期化テスト"""
        matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = ShapleyAnalyzer(matrix, nodes)
        
        assert analyzer.n == 3
        assert len(analyzer.node_names) == 3
        assert analyzer.value_func is not None
    
    def test_value_functions(self):
        """価値関数のテスト"""
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [2, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = ShapleyAnalyzer(matrix, nodes, value_function="pagerank_sum")
        
        # 全ノードの価値
        coalition = np.array([0, 1, 2])
        value = analyzer._evaluate_coalition(coalition)
        
        # PageRankの合計は1.0付近
        assert 0.5 <= value <= 1.5
        
        # 空集合の価値は0
        empty_coalition = np.array([])
        empty_value = analyzer._evaluate_coalition(empty_coalition)
        assert empty_value == 0.0
    
    def test_shapley_additivity(self):
        """
        Shapley値の加法性テスト
        sum(Shapley値) ≈ V(全体)
        """
        matrix = np.array([
            [0, 3, 0, 2],
            [0, 0, 4, 0],
            [1, 0, 0, 3],
            [2, 1, 0, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        analyzer = ShapleyAnalyzer(matrix, nodes, value_function="pagerank_sum")
        
        # 少ないサンプル数でテスト（高速化）
        result = analyzer.compute_shapley_values(n_samples=100, random_seed=42)
        
        # Shapley値の合計
        shapley_sum = sum(result.shapley_values.values())
        
        # 全体の価値
        total_value = result.total_value
        
        # 誤差10%以内
        assert abs(shapley_sum - total_value) / total_value < 0.1
    
    def test_shapley_symmetry(self):
        """
        Shapley値の対称性テスト
        同じ構造のノードは同じShapley値を持つべき
        """
        # 対称な構造
        matrix = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = ShapleyAnalyzer(matrix, nodes, value_function="connectivity")
        
        result = analyzer.compute_shapley_values(n_samples=200, random_seed=42)
        
        # 全ノードのShapley値が近似的に等しい
        values = list(result.shapley_values.values())
        mean_value = np.mean(values)
        
        for value in values:
            # 平均から20%以内
            assert abs(value - mean_value) / mean_value < 0.2
    
    def test_null_player(self):
        """
        Null Playerテスト
        孤立ノード（接続なし）のShapley値は0に近い
        """
        # ノードDは完全に孤立
        matrix = np.array([
            [0, 3, 2, 0],
            [0, 0, 4, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        nodes = ["A", "B", "C", "D_isolated"]
        
        analyzer = ShapleyAnalyzer(matrix, nodes, value_function="pagerank_sum")
        
        result = analyzer.compute_shapley_values(n_samples=200, random_seed=42)
        
        # 孤立ノードのShapley値は他より著しく低い
        isolated_value = result.shapley_values["D_isolated"]
        other_values = [v for k, v in result.shapley_values.items() if k != "D_isolated"]
        min_other = min(other_values)
        
        assert isolated_value < min_other * 0.5
    
    def test_result_structure(self):
        """結果構造のテスト"""
        matrix = np.array([
            [0, 5, 3],
            [2, 0, 4],
            [1, 3, 0]
        ])
        nodes = ["A", "B", "C"]
        categories = {"A": "Cat1", "B": "Cat1", "C": "Cat2"}
        
        analyzer = ShapleyAnalyzer(
            matrix, nodes, 
            node_categories=categories,
            value_function="pagerank_sum"
        )
        
        result = analyzer.compute_shapley_values(n_samples=100, random_seed=42)
        
        # 結果の構造確認
        assert len(result.shapley_values) == 3
        assert len(result.top_contributors) == 3
        assert len(result.cumulative_contribution) == 3
        assert len(result.category_contributions) == 2  # Cat1, Cat2
        assert result.total_value > 0
        assert result.computation_time > 0
        assert result.n_samples == 100
        assert len(result.interpretation) > 0
    
    def test_cumulative_contribution(self):
        """累積貢献度のテスト"""
        matrix = np.array([
            [0, 8, 7, 6],
            [0, 0, 5, 4],
            [0, 0, 0, 3],
            [0, 0, 0, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        analyzer = ShapleyAnalyzer(matrix, nodes)
        result = analyzer.compute_shapley_values(n_samples=200, random_seed=42)
        
        # 累積貢献度は単調増加
        cumulative_values = [pct for _, pct in result.cumulative_contribution]
        
        for i in range(len(cumulative_values) - 1):
            assert cumulative_values[i] <= cumulative_values[i+1]
        
        # 最後は100%
        assert cumulative_values[-1] == 100.0
    
    def test_interpretation_generation(self):
        """解釈文生成のテスト"""
        matrix = np.array([
            [0, 5, 3],
            [2, 0, 4],
            [1, 3, 0]
        ])
        nodes = ["Process_A", "Process_B", "Process_C"]
        
        analyzer = ShapleyAnalyzer(matrix, nodes)
        result = analyzer.compute_shapley_values(n_samples=100, random_seed=42)
        
        # 解釈文に重要な情報が含まれているか
        assert "最重要ノード" in result.interpretation
        assert "重点管理対象" in result.interpretation
        assert "80%" in result.interpretation
        
        # トップノード名が含まれている
        top_node_name = result.top_contributors[0][0]
        assert top_node_name in result.interpretation


class TestCoalitionStability:
    """連携安定性分析のテスト"""
    
    def test_coalition_stability(self):
        """連携安定性のテスト"""
        matrix = np.array([
            [0, 9, 8, 1],
            [0, 0, 7, 2],
            [0, 0, 0, 3],
            [0, 0, 0, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        analyzer = ShapleyAnalyzer(matrix, nodes)
        result = analyzer.compute_shapley_values(n_samples=100, random_seed=42)
        
        stability = compute_shapley_coalition_stability(
            result.shapley_values,
            matrix,
            nodes
        )
        
        # 結果の構造確認
        assert "top_contributors" in stability
        assert "dense_connections" in stability
        assert "recommendation" in stability
        assert len(stability["top_contributors"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
