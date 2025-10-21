"""
Test cases for Advanced Analytics Export
"""

import pytest
import numpy as np
import pandas as pd
from io import BytesIO
import json
from unittest.mock import MagicMock
from dataclasses import dataclass

from utils.analytics_export import AdvancedAnalyticsExporter


@dataclass
class MockShapleyResult:
    """Mock Shapley Value result"""
    shapley_values: dict
    top_contributors: list
    cumulative_contribution: list
    category_contributions: dict
    total_value: float
    computation_time: float
    n_samples: int
    interpretation: str


@dataclass
class MockTEResult:
    """Mock Transfer Entropy result"""
    te_matrix: np.ndarray
    significant_flows: list
    bottleneck_nodes: list
    info_inflow: dict
    info_outflow: dict
    comparison_with_original: pd.DataFrame
    interpretation: str
    computation_time: float
    n_walks: int
    walk_length: int


@dataclass
class MockBootstrapResult:
    """Mock Bootstrap result"""
    metric_name: str
    node_ci: dict
    group_comparison: pd.DataFrame
    stable_findings: list
    unstable_findings: list
    interpretation: str
    computation_time: float
    n_bootstrap: int
    alpha: float


class TestAdvancedAnalyticsExporter:
    """AdvancedAnalyticsExporterのテスト"""
    
    def test_initialization(self):
        """初期化テスト"""
        analytics_results = {
            "shapley": {
                "result": MagicMock(),
                "parameters": {},
                "timestamp": "2025-01-01 00:00:00"
            }
        }
        
        exporter = AdvancedAnalyticsExporter(analytics_results)
        
        assert exporter.results == analytics_results
    
    def test_empty_results(self):
        """空の分析結果のテスト"""
        analytics_results = {}
        
        exporter = AdvancedAnalyticsExporter(analytics_results)
        
        # Excelエクスポートが失敗しない
        excel_buffer = exporter.export_to_excel()
        assert isinstance(excel_buffer, BytesIO)
        
        # JSONエクスポートが失敗しない
        json_str = exporter.export_to_json()
        json_data = json.loads(json_str)
        assert json_data["export_version"] == "1.0.0"
        assert json_data["analyses"] == {}
    
    def test_shapley_export_to_excel(self):
        """Shapley ValueのExcelエクスポートテスト（連携安定性含む）"""
        shapley_result = MockShapleyResult(
            shapley_values={"A": 0.5, "B": 0.3, "C": 0.2},
            top_contributors=[("A", 0.5), ("B", 0.3), ("C", 0.2)],
            cumulative_contribution=[(1, 50), (2, 80), (3, 100)],
            category_contributions={"Cat1": 0.4, "Cat2": 0.6},
            total_value=1.0,
            computation_time=10.5,
            n_samples=1000,
            interpretation="Test interpretation"
        )
        
        # 連携安定性データ（NEW）
        stability_data = {
            "top_contributors": ["A", "B"],
            "dense_connections": [("A", "B", 5.0)],
            "recommendation": "上位2ノードの連携を強化することで、相乗効果が期待できます。"
        }
        
        analytics_results = {
            "shapley": {
                "result": shapley_result,
                "stability": stability_data,  # NEW
                "parameters": {"n_samples": 1000, "value_function": "pagerank_sum"},
                "timestamp": "2025-01-01 00:00:00"
            }
        }
        
        exporter = AdvancedAnalyticsExporter(analytics_results)
        excel_buffer = exporter.export_to_excel()
        
        # バッファが生成されている
        assert isinstance(excel_buffer, BytesIO)
        assert excel_buffer.getbuffer().nbytes > 0
        
        # Excelファイルとして読み込める
        excel_file = pd.ExcelFile(excel_buffer)
        assert "サマリー" in excel_file.sheet_names
        assert "Shapley_Values" in excel_file.sheet_names
        assert "Shapley_Cumulative" in excel_file.sheet_names
        assert "Shapley_Categories" in excel_file.sheet_names
        assert "Shapley_TopContributors" in excel_file.sheet_names  # NEW
        assert "Shapley_DensePairs" in excel_file.sheet_names  # NEW
    
    def test_te_export_to_excel(self):
        """Transfer EntropyのExcelエクスポートテスト"""
        te_matrix = np.array([
            [0, 0.1, 0.2],
            [0.15, 0, 0.05],
            [0.1, 0.2, 0]
        ])
        
        comparison_df = pd.DataFrame({
            "From": ["A", "B"],
            "To": ["B", "C"],
            "元のスコア": [5, 3],
            "TE (bits)": [0.1, 0.05],
            "判定": ["✅ 一致", "⬆️ 隠れた影響"]
        })
        
        te_result = MockTEResult(
            te_matrix=te_matrix,
            significant_flows=[("A", "C", 0.2), ("B", "C", 0.2)],
            bottleneck_nodes=["B"],
            info_inflow={"A": 0.25, "B": 0.1, "C": 0.45},
            info_outflow={"A": 0.3, "B": 0.2, "C": 0.3},
            comparison_with_original=comparison_df,
            interpretation="Test interpretation",
            computation_time=20.0,
            n_walks=1000,
            walk_length=50
        )
        
        analytics_results = {
            "transfer_entropy": {
                "result": te_result,
                "parameters": {"n_walks": 1000, "walk_length": 50, "n_bins": 3},
                "timestamp": "2025-01-01 00:00:00"
            }
        }
        
        exporter = AdvancedAnalyticsExporter(analytics_results)
        excel_buffer = exporter.export_to_excel()
        
        # Excelファイルとして読み込める
        excel_file = pd.ExcelFile(excel_buffer)
        assert "サマリー" in excel_file.sheet_names
        assert "TE_Matrix" in excel_file.sheet_names
        assert "TE_Flows" in excel_file.sheet_names
        assert "TE_Comparison" in excel_file.sheet_names
    
    def test_bootstrap_export_to_excel(self):
        """BootstrapのExcelエクスポートテスト"""
        node_ci = {
            "A": (0.5, 0.45, 0.55),
            "B": (0.3, 0.25, 0.35),
            "C": (0.2, 0.15, 0.25)
        }
        
        group_comparison = pd.DataFrame({
            "グループA": ["Group1"],
            "グループB": ["Group2"],
            "平均値A": [0.4],
            "平均値B": [0.25],
            "平均値の差": [0.15],
            "p値": [0.03],
            "有意性": ["✅ 有意"]
        })
        
        bs_result = MockBootstrapResult(
            metric_name="PageRank",
            node_ci=node_ci,
            group_comparison=group_comparison,
            stable_findings=["A: 0.5 [0.45, 0.55]"],
            unstable_findings=[],
            interpretation="Test interpretation",
            computation_time=15.0,
            n_bootstrap=1000,
            alpha=0.05
        )
        
        analytics_results = {
            "bootstrap": {
                "result": bs_result,
                "parameters": {"n_bootstrap": 1000, "alpha": 0.05},
                "timestamp": "2025-01-01 00:00:00"
            }
        }
        
        exporter = AdvancedAnalyticsExporter(analytics_results)
        excel_buffer = exporter.export_to_excel()
        
        # Excelファイルとして読み込める
        excel_file = pd.ExcelFile(excel_buffer)
        assert "サマリー" in excel_file.sheet_names
        assert "Bootstrap_CI" in excel_file.sheet_names
        assert "Bootstrap_Groups" in excel_file.sheet_names
    
    def test_json_export(self):
        """JSONエクスポートテスト"""
        shapley_result = MockShapleyResult(
            shapley_values={"A": 0.5, "B": 0.3},
            top_contributors=[("A", 0.5), ("B", 0.3)],
            cumulative_contribution=[],
            category_contributions={},
            total_value=0.8,
            computation_time=10.0,
            n_samples=1000,
            interpretation="Test"
        )
        
        analytics_results = {
            "shapley": {
                "result": shapley_result,
                "parameters": {"n_samples": 1000},
                "timestamp": "2025-01-01 00:00:00"
            }
        }
        
        exporter = AdvancedAnalyticsExporter(analytics_results)
        json_str = exporter.export_to_json()
        
        # JSONとしてパース可能
        json_data = json.loads(json_str)
        
        # 構造確認
        assert json_data["export_version"] == "1.0.0"
        assert "export_timestamp" in json_data
        assert "analyses" in json_data
        assert "shapley_value" in json_data["analyses"]
        
        # Shapleyデータ確認
        shapley_data = json_data["analyses"]["shapley_value"]
        assert shapley_data["shapley_values"] == {"A": 0.5, "B": 0.3}
        assert shapley_data["total_value"] == 0.8
        assert shapley_data["computation_time"] == 10.0
    
    def test_multiple_analyses_export(self):
        """複数分析のエクスポートテスト"""
        shapley_result = MockShapleyResult(
            shapley_values={"A": 0.5},
            top_contributors=[("A", 0.5)],
            cumulative_contribution=[],
            category_contributions={},
            total_value=0.5,
            computation_time=5.0,
            n_samples=100,
            interpretation="Test"
        )
        
        te_result = MockTEResult(
            te_matrix=np.array([[0, 0.1], [0.15, 0]]),
            significant_flows=[],
            bottleneck_nodes=[],
            info_inflow={},
            info_outflow={},
            comparison_with_original=pd.DataFrame(),
            interpretation="Test",
            computation_time=10.0,
            n_walks=100,
            walk_length=20
        )
        
        analytics_results = {
            "shapley": {
                "result": shapley_result,
                "parameters": {},
                "timestamp": "2025-01-01 00:00:00"
            },
            "transfer_entropy": {
                "result": te_result,
                "parameters": {},
                "timestamp": "2025-01-01 00:00:00"
            }
        }
        
        exporter = AdvancedAnalyticsExporter(analytics_results)
        
        # Excelエクスポート
        excel_buffer = exporter.export_to_excel()
        excel_file = pd.ExcelFile(excel_buffer)
        
        # 両方の分析のシートが存在
        assert "Shapley_Values" in excel_file.sheet_names
        assert "TE_Matrix" in excel_file.sheet_names
        
        # JSONエクスポート
        json_str = exporter.export_to_json()
        json_data = json.loads(json_str)
        
        # 両方の分析データが存在
        assert "shapley_value" in json_data["analyses"]
        assert "transfer_entropy" in json_data["analyses"]


    def test_graph_embedding_export(self):
        """Graph EmbeddingのExcelエクスポートテスト"""
        from dataclasses import dataclass
        import numpy as np
        
        @dataclass
        class MockGraphEmbeddingResult:
            node_embeddings: dict
            node_positions_2d: dict
            communities: dict
            community_labels: dict
            similarity_matrix: np.ndarray
            modularity: float
            top_similar_pairs: list
            computation_time: float
            embedding_dim: int
            n_communities: int
            interpretation: str
        
        ge_result = MockGraphEmbeddingResult(
            node_embeddings={"A": np.array([0.1, 0.2]), "B": np.array([0.3, 0.4])},
            node_positions_2d={"A": (1.0, 2.0), "B": (3.0, 4.0)},
            communities={"A": 0, "B": 0},
            community_labels={0: "コミュニティ1 (2ノード)"},
            similarity_matrix=np.array([[1.0, 0.8], [0.8, 1.0]]),
            modularity=0.5,
            top_similar_pairs=[("A", "B", 0.8)],
            computation_time=15.2,
            embedding_dim=64,
            n_communities=1,
            interpretation="Test interpretation"
        )
        
        analytics_results = {
            "graph_embedding": {
                "result": ge_result,
                "parameters": {"embedding_dim": 64, "walk_length": 20},
                "timestamp": "2025-01-01 00:00:00"
            }
        }
        
        exporter = AdvancedAnalyticsExporter(analytics_results)
        excel_buffer = exporter.export_to_excel()
        
        assert isinstance(excel_buffer, BytesIO)
        assert excel_buffer.getbuffer().nbytes > 0
        
        # JSONエクスポート
        json_str = exporter.export_to_json()
        json_data = json.loads(json_str)
        
        assert "graph_embedding" in json_data["analyses"]
        ge_json = json_data["analyses"]["graph_embedding"]
        assert ge_json["modularity"] == 0.5
        assert ge_json["n_communities"] == 1
    
    def test_fisher_information_export(self):
        """Fisher InformationのExcelエクスポートテスト"""
        from dataclasses import dataclass
        import numpy as np
        
        @dataclass
        class MockFisherInformationResult:
            fisher_matrix: np.ndarray
            sensitivity_scores: dict
            top_sensitive_edges: list
            cramer_rao_bounds: dict
            condition_number: float
            effective_rank: int
            eigenvalues: np.ndarray
            computation_time: float
            n_edges: int
            interpretation: str
        
        fi_result = MockFisherInformationResult(
            fisher_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            sensitivity_scores={("A", "B"): 1.0, ("B", "C"): 0.5},
            top_sensitive_edges=[("A", "B", 1.0), ("B", "C", 0.5)],
            cramer_rao_bounds={("A", "B"): 0.8, ("B", "C"): 1.2},
            condition_number=2.0,
            effective_rank=2,
            eigenvalues=np.array([1.5, 0.5]),
            computation_time=0.5,
            n_edges=2,
            interpretation="Test interpretation"
        )
        
        analytics_results = {
            "fisher_information": {
                "result": fi_result,
                "parameters": {"noise_variance": 1.0},
                "timestamp": "2025-01-01 00:00:00"
            }
        }
        
        exporter = AdvancedAnalyticsExporter(analytics_results)
        excel_buffer = exporter.export_to_excel()
        
        assert isinstance(excel_buffer, BytesIO)
        assert excel_buffer.getbuffer().nbytes > 0
        
        json_str = exporter.export_to_json()
        json_data = json.loads(json_str)
        
        assert "fisher_information" in json_data["analyses"]
        fi_json = json_data["analyses"]["fisher_information"]
        assert fi_json["n_edges"] == 2
        assert fi_json["condition_number"] == 2.0
        assert fi_json["effective_rank"] == 2
    
    def test_bayesian_inference_export(self):
        """Bayesian InferenceのExcelエクスポートテスト"""
        from dataclasses import dataclass
        import numpy as np
        
        @dataclass
        class MockBayesianInferenceResult:
            posterior_mean: dict
            posterior_std: dict
            credible_intervals: dict
            high_uncertainty_edges: list
            uncertainty_scores: dict
            prior_params: dict
            computation_time: float
            n_edges: int
            n_bootstrap: int
            credible_level: float
            interpretation: str
        
        bi_result = MockBayesianInferenceResult(
            posterior_mean={("A", "B"): 5.0, ("B", "C"): 3.0},
            posterior_std={("A", "B"): 0.5, ("B", "C"): 0.8},
            credible_intervals={("A", "B"): (5.0, 4.0, 6.0), ("B", "C"): (3.0, 1.5, 4.5)},
            high_uncertainty_edges=[("B", "C", 0.6), ("A", "B", 0.2)],
            uncertainty_scores={("A", "B"): 0.2, ("B", "C"): 0.6},
            prior_params={"mu_0": 0.0, "sigma_0": 5.0},
            computation_time=1.5,
            n_edges=2,
            n_bootstrap=1000,
            credible_level=0.95,
            interpretation="Test interpretation"
        )
        
        analytics_results = {
            "bayesian_inference": {
                "result": bi_result,
                "parameters": {"n_bootstrap": 1000, "credible_level": 0.95},
                "timestamp": "2025-01-01 00:00:00"
            }
        }
        
        exporter = AdvancedAnalyticsExporter(analytics_results)
        excel_buffer = exporter.export_to_excel()
        
        assert isinstance(excel_buffer, BytesIO)
        assert excel_buffer.getbuffer().nbytes > 0
        
        json_str = exporter.export_to_json()
        json_data = json.loads(json_str)
        
        assert "bayesian_inference" in json_data["analyses"]
        bi_json = json_data["analyses"]["bayesian_inference"]
        assert bi_json["n_edges"] == 2
        assert bi_json["credible_level"] == 0.95
        assert bi_json["n_bootstrap"] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
