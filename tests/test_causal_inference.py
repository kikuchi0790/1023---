"""
Test cases for Causal Inference Analysis
"""

import pytest
import numpy as np
import networkx as nx
from utils.causal_inference import CausalInferenceAnalyzer


class TestCausalInferenceAnalyzer:
    """CausalInferenceAnalyzerのテスト"""
    
    def test_initialization(self):
        """初期化テスト"""
        matrix = np.array([
            [0, 3, 0],
            [0, 0, 2],
            [1, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = CausalInferenceAnalyzer(matrix, nodes)
        
        assert analyzer.n == 3
        assert len(analyzer.node_names) == 3
        assert analyzer.max_path_length == 4
        assert analyzer.graph.number_of_nodes() == 3
        assert analyzer.graph.number_of_edges() == 3
    
    def test_direct_effects(self):
        """直接効果の計算テスト"""
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [2, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = CausalInferenceAnalyzer(matrix, nodes)
        direct_effects = analyzer._compute_direct_effects()
        
        # 直接効果は隣接行列の値と一致
        assert direct_effects[("A", "B")] == 5
        assert direct_effects[("B", "C")] == 3
        assert direct_effects[("C", "A")] == 2
        
        # 非ゼロのエッジのみ
        assert len(direct_effects) == 3
    
    def test_indirect_effects(self):
        """間接効果の計算テスト"""
        # A → B → C の連鎖
        matrix = np.array([
            [0, 2, 0],
            [0, 0, 3],
            [0, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = CausalInferenceAnalyzer(matrix, nodes, max_path_length=3)
        indirect_effects = analyzer._compute_indirect_effects()
        
        # A → B → C の間接効果が存在するべき
        assert ("A", "C") in indirect_effects
        
        # 間接効果 = 2 * 3 * decay_factor^1
        expected = 2 * 3 * (0.8 ** 1)
        assert abs(indirect_effects[("A", "C")] - expected) < 0.1
    
    def test_total_effects(self):
        """総効果の計算テスト"""
        direct = {
            ("A", "B"): 5.0,
            ("B", "C"): 3.0
        }
        
        indirect = {
            ("A", "C"): 2.0
        }
        
        matrix = np.zeros((3, 3))
        nodes = ["A", "B", "C"]
        analyzer = CausalInferenceAnalyzer(matrix, nodes)
        
        total = analyzer._compute_total_effects(direct, indirect)
        
        # 総効果 = 直接 + 間接
        assert total[("A", "B")] == 5.0
        assert total[("B", "C")] == 3.0
        assert total[("A", "C")] == 2.0
    
    def test_find_paths_with_length(self):
        """経路探索テスト"""
        matrix = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        analyzer = CausalInferenceAnalyzer(matrix, nodes)
        
        # 長さ1の経路（直接エッジ）
        paths_1 = analyzer._find_paths_with_length("A", "B", 1)
        assert len(paths_1) == 1
        assert paths_1[0] == ["A", "B"]
        
        # 長さ2の経路
        paths_2 = analyzer._find_paths_with_length("A", "C", 2)
        assert len(paths_2) == 1
        assert paths_2[0] == ["A", "B", "C"]
        
        # 長さ3の経路
        paths_3 = analyzer._find_paths_with_length("A", "D", 3)
        assert len(paths_3) == 1
        assert paths_3[0] == ["A", "B", "C", "D"]
    
    def test_simulate_intervention(self):
        """介入シミュレーションテスト"""
        # A → B → C
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [0, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = CausalInferenceAnalyzer(matrix, nodes)
        
        # Aに介入（50%改善）
        effects = analyzer._simulate_intervention("A", 1.5)
        
        # A自身の効果
        assert abs(effects["A"] - 0.5) < 0.01
        
        # Bへの波及効果（A → B）
        assert effects["B"] > 0
        
        # Cへの波及効果（A → B → C）
        assert effects["C"] > 0
    
    def test_detect_confounders(self):
        """交絡因子検出テスト"""
        # Z → A, Z → B（Zは交絡因子）
        matrix = np.array([
            [0, 0, 0],  # A
            [0, 0, 0],  # B
            [1, 1, 0]   # Z
        ])
        nodes = ["A", "B", "Z"]
        
        analyzer = CausalInferenceAnalyzer(matrix, nodes)
        confounders = analyzer._detect_confounders()
        
        # AとBの間にZが交絡因子として検出されるべき
        found = False
        for source, target, conf_list in confounders:
            if (source == "A" and target == "B") or (source == "B" and target == "A"):
                if "Z" in conf_list:
                    found = True
        
        assert found
    
    def test_identify_top_intervention_targets(self):
        """介入ターゲット特定テスト"""
        intervention_effects = {
            "A": {"B": 0.5, "C": 0.3},
            "B": {"C": 0.2},
            "C": {}
        }
        
        matrix = np.zeros((3, 3))
        nodes = ["A", "B", "C"]
        analyzer = CausalInferenceAnalyzer(matrix, nodes)
        
        top_targets = analyzer._identify_top_intervention_targets(intervention_effects)
        
        # Aが最も影響力が大きい
        assert top_targets[0][0] == "A"
        assert top_targets[0][1] == 0.8  # 0.5 + 0.3
    
    def test_comprehensive_analysis(self):
        """包括的分析のテスト"""
        # 複雑なネットワーク
        matrix = np.array([
            [0, 5, 2, 0],
            [0, 0, 3, 1],
            [0, 0, 0, 4],
            [0, 0, 0, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        analyzer = CausalInferenceAnalyzer(matrix, nodes, max_path_length=4)
        
        result = analyzer.compute_causal_inference(
            intervention_node="A",
            intervention_strength=1.5
        )
        
        # 結果の構造確認
        assert isinstance(result.direct_effects, dict)
        assert isinstance(result.indirect_effects, dict)
        assert isinstance(result.total_effects, dict)
        assert isinstance(result.causal_paths, dict)
        assert isinstance(result.intervention_effects, dict)
        assert isinstance(result.confounders, list)
        assert isinstance(result.top_intervention_targets, list)
        assert isinstance(result.interpretation, str)
        assert result.computation_time > 0
        
        # 直接効果の存在確認
        assert len(result.direct_effects) > 0
        
        # 介入効果の存在確認
        assert "A" in result.intervention_effects
        assert len(result.intervention_effects["A"]) > 0
    
    def test_causal_paths(self):
        """因果経路探索のテスト"""
        # A → B → C → D
        matrix = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        analyzer = CausalInferenceAnalyzer(matrix, nodes, max_path_length=4)
        causal_paths = analyzer._find_all_causal_paths()
        
        # A → D への経路が存在
        assert ("A", "D") in causal_paths
        
        # 経路の内容確認
        paths = causal_paths[("A", "D")]
        assert len(paths) > 0
        
        # 最短経路は ["A", "B", "C", "D"]
        assert ["A", "B", "C", "D"] in paths
    
    def test_with_cycle(self):
        """サイクルを含むグラフのテスト"""
        # A → B → C → A（サイクル）
        matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = CausalInferenceAnalyzer(matrix, nodes)
        
        # サイクルがあっても分析が完了する
        result = analyzer.compute_causal_inference()
        
        assert result is not None
        assert len(result.direct_effects) == 3
    
    def test_disconnected_graph(self):
        """非連結グラフのテスト"""
        # A → B, C → D（2つの連結成分）
        matrix = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        analyzer = CausalInferenceAnalyzer(matrix, nodes)
        result = analyzer.compute_causal_inference()
        
        # 直接効果は2つ
        assert len(result.direct_effects) == 2
        
        # A → D の効果は存在しない（非連結）
        assert ("A", "D") not in result.total_effects


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
