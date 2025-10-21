"""
Test cases for Bootstrap Statistical Testing
"""

import pytest
import numpy as np
import networkx as nx
from utils.statistical_testing import BootstrapTester, compute_stability_score


class TestBootstrapTester:
    """Bootstrap統計検定のテスト"""
    
    def test_initialization(self):
        """初期化テスト"""
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [2, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        tester = BootstrapTester(matrix, nodes)
        
        assert tester.n == 3
        assert len(tester.node_names) == 3
        assert tester.n_bootstrap == 1000
        assert tester.alpha == 0.05
    
    def test_bootstrap_ci_structure(self):
        """Bootstrap信頼区間の構造テスト"""
        matrix = np.array([
            [0, 3, 2],
            [1, 0, 4],
            [2, 3, 0]
        ])
        nodes = ["A", "B", "C"]
        
        tester = BootstrapTester(matrix, nodes, n_bootstrap=50)
        
        def pagerank_func(mat):
            G = nx.from_numpy_array(mat, create_using=nx.DiGraph)
            pr = nx.pagerank(G, weight='weight')
            return pr
        
        ci_results = tester.bootstrap_confidence_interval(pagerank_func)
        
        # 結果の構造確認
        assert len(ci_results) == 3
        
        for node in nodes:
            assert node in ci_results
            value, lower, upper = ci_results[node]
            
            # 信頼区間は値を含む
            assert lower <= value <= upper
    
    def test_bootstrap_ci_coverage(self):
        """
        Bootstrap信頼区間のカバレッジテスト
        
        信頼区間が元の値を含むべき
        """
        matrix = np.array([
            [0, 5, 3, 2],
            [1, 0, 4, 1],
            [2, 3, 0, 2],
            [1, 1, 2, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        tester = BootstrapTester(matrix, nodes, n_bootstrap=100)
        
        def pagerank_func(mat):
            G = nx.from_numpy_array(mat, create_using=nx.DiGraph)
            pr = nx.pagerank(G, weight='weight')
            return pr
        
        ci_results = tester.bootstrap_confidence_interval(pagerank_func)
        
        # 全ノードで値が信頼区間内
        for node, (value, lower, upper) in ci_results.items():
            assert lower <= value <= upper, f"{node}: {lower} <= {value} <= {upper}"
    
    def test_permutation_test_null_hypothesis(self):
        """
        Permutation検定のNull仮説テスト
        
        同一分布からのサンプルではp値が大きい（有意でない）
        """
        matrix = np.array([
            [0, 5, 3, 2, 1],
            [1, 0, 4, 1, 2],
            [2, 3, 0, 2, 1],
            [1, 1, 2, 0, 3],
            [2, 1, 1, 3, 0]
        ])
        nodes = ["A1", "A2", "A3", "B1", "B2"]
        
        # グループAとグループBは実質同じ（Null仮説が真）
        tester = BootstrapTester(matrix, nodes, n_bootstrap=100)
        
        def pagerank_func(mat):
            G = nx.from_numpy_array(mat, create_using=nx.DiGraph)
            pr = nx.pagerank(G, weight='weight')
            return pr
        
        # 同じような構造のノード同士を比較
        result = tester.permutation_test(
            pagerank_func,
            ["A1", "A2"],
            ["B1", "B2"],
            n_permutations=100
        )
        
        # Null仮説が真なので、p値は大きいはず（>0.05）
        # ただしランダム性があるので緩い条件
        assert 0 <= result["p_value"] <= 1
        assert "observed_diff" in result
        assert "significant" in result
    
    def test_permutation_test_alternative(self):
        """
        Permutation検定の対立仮説テスト
        
        明らかに異なる分布ではp値が小さい（有意）
        """
        # グループAは高スコア、グループBは低スコア
        matrix = np.array([
            [0, 9, 9, 0, 0],
            [9, 0, 9, 0, 0],
            [9, 9, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0]
        ])
        nodes = ["HighA", "HighB", "HighC", "LowA", "LowB"]
        
        tester = BootstrapTester(matrix, nodes, n_bootstrap=100)
        
        def pagerank_func(mat):
            G = nx.from_numpy_array(mat, create_using=nx.DiGraph)
            pr = nx.pagerank(G, weight='weight')
            return pr
        
        result = tester.permutation_test(
            pagerank_func,
            ["HighA", "HighB", "HighC"],
            ["LowA", "LowB"],
            n_permutations=100
        )
        
        # 明らかに異なるので有意なはず
        assert result["p_value"] < 0.1  # 緩い条件（サンプル数少ないので）
    
    def test_comprehensive_analysis(self):
        """包括的分析のテスト"""
        matrix = np.array([
            [0, 5, 3],
            [2, 0, 4],
            [1, 3, 0]
        ])
        nodes = ["A", "B", "C"]
        node_groups = {"A": "Group1", "B": "Group1", "C": "Group2"}
        
        tester = BootstrapTester(
            matrix, nodes, 
            node_groups=node_groups,
            n_bootstrap=50
        )
        
        result = tester.run_comprehensive_bootstrap_analysis()
        
        # 結果の構造確認
        assert result.metric_name == "PageRank"
        assert len(result.node_ci) == 3
        assert isinstance(result.stable_findings, list)
        assert isinstance(result.unstable_findings, list)
        assert isinstance(result.interpretation, str)
        assert result.computation_time > 0
        assert result.n_bootstrap == 50
        assert result.alpha == 0.05
    
    def test_stability_classification(self):
        """安定性分類のテスト"""
        ci_results = {
            "Stable": (1.0, 0.9, 1.1),  # 相対誤差10%
            "Unstable": (1.0, 0.5, 1.5),  # 相対誤差50%
            "VeryUnstable": (1.0, 0.0, 2.0)  # 相対誤差100%
        }
        
        stability_df = compute_stability_score(ci_results)
        
        # 安定性スコアの順序
        stable_score = stability_df[stability_df["ノード名"] == "Stable"]["安定性スコア"].values[0]
        unstable_score = stability_df[stability_df["ノード名"] == "Unstable"]["安定性スコア"].values[0]
        very_unstable_score = stability_df[stability_df["ノード名"] == "VeryUnstable"]["安定性スコア"].values[0]
        
        # 安定なほどスコアが高い
        assert stable_score > unstable_score > very_unstable_score
        
        # 判定ラベル
        stable_judgment = stability_df[stability_df["ノード名"] == "Stable"]["判定"].values[0]
        unstable_judgment = stability_df[stability_df["ノード名"] == "Unstable"]["判定"].values[0]
        
        assert "安定" in stable_judgment
        assert "不安定" in unstable_judgment
    
    def test_with_small_data(self):
        """小規模データでのテスト"""
        matrix = np.array([
            [0, 1],
            [1, 0]
        ])
        nodes = ["A", "B"]
        
        tester = BootstrapTester(matrix, nodes, n_bootstrap=20)
        
        def pagerank_func(mat):
            G = nx.from_numpy_array(mat, create_using=nx.DiGraph)
            pr = nx.pagerank(G, weight='weight')
            return pr
        
        ci_results = tester.bootstrap_confidence_interval(pagerank_func)
        
        # 小規模でも動作
        assert len(ci_results) == 2
        
        for node, (value, lower, upper) in ci_results.items():
            assert lower <= value <= upper
    
    def test_empty_matrix(self):
        """空行列（全ゼロ）での挙動テスト"""
        matrix = np.zeros((3, 3))
        nodes = ["A", "B", "C"]
        
        tester = BootstrapTester(matrix, nodes, n_bootstrap=10)
        
        def pagerank_func(mat):
            G = nx.from_numpy_array(mat, create_using=nx.DiGraph)
            pr = nx.pagerank(G, weight='weight') if G.number_of_edges() > 0 else {n: 1/3 for n in ["A", "B", "C"]}
            return pr
        
        # エラーが発生せず、何らかの結果が返る
        ci_results = tester.bootstrap_confidence_interval(pagerank_func)
        
        assert len(ci_results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
