"""
Test cases for Graph Embedding Analysis
"""

import pytest
import numpy as np
import networkx as nx
from utils.graph_embedding import GraphEmbeddingAnalyzer


class TestGraphEmbeddingAnalyzer:
    """GraphEmbeddingAnalyzerのテスト"""
    
    def test_initialization(self):
        """初期化テスト"""
        matrix = np.array([
            [0, 3, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 4],
            [1, 0, 0, 0]
        ])
        nodes = ["A", "B", "C", "D"]
        
        analyzer = GraphEmbeddingAnalyzer(matrix, nodes, embedding_dim=32)
        
        assert analyzer.n == 4
        assert len(analyzer.node_names) == 4
        assert analyzer.embedding_dim == 32
        assert analyzer.graph.number_of_nodes() == 4
        assert analyzer.graph.number_of_edges() == 4
    
    def test_random_walk_generation(self):
        """ランダムウォーク生成テスト"""
        matrix = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [0, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = GraphEmbeddingAnalyzer(matrix, nodes, walk_length=10, num_walks=5)
        walks = analyzer._generate_random_walks()
        
        # 各ノードから5本ずつ
        assert len(walks) == 3 * 5
        
        # ウォークの開始ノードが正しい
        node_walk_counts = {node: 0 for node in nodes}
        for walk in walks:
            assert len(walk) > 0
            node_walk_counts[walk[0]] += 1
        
        for node in nodes:
            assert node_walk_counts[node] == 5
    
    def test_node_embeddings(self):
        """ノード埋め込みテスト"""
        matrix = np.array([
            [0, 3, 2],
            [1, 0, 3],
            [2, 1, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = GraphEmbeddingAnalyzer(matrix, nodes, embedding_dim=16)
        walks = analyzer._generate_random_walks()
        embeddings = analyzer._compute_node_embeddings(walks)
        
        # 全ノードの埋め込みが存在
        assert len(embeddings) == 3
        
        # 次元数が正しい
        for node in nodes:
            assert len(embeddings[node]) == 16
            # ゼロベクトルでない
            assert np.linalg.norm(embeddings[node]) > 0
    
    def test_community_detection(self):
        """コミュニティ検出テスト"""
        # 2つの密なクラスター
        matrix = np.array([
            [0, 5, 5, 1, 0],
            [5, 0, 5, 0, 1],
            [5, 5, 0, 1, 0],
            [1, 0, 1, 0, 5],
            [0, 1, 0, 5, 0]
        ])
        nodes = ["A", "B", "C", "D", "E"]
        
        analyzer = GraphEmbeddingAnalyzer(matrix, nodes)
        communities = analyzer._detect_communities()
        
        # 全ノードにコミュニティが割り当てられている
        assert len(communities) == 5
        
        # コミュニティIDは0以上の整数
        for comm_id in communities.values():
            assert isinstance(comm_id, int)
            assert comm_id >= 0
    
    def test_modularity_calculation(self):
        """Modularity計算テスト"""
        matrix = np.array([
            [0, 5, 0],
            [5, 0, 0],
            [0, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = GraphEmbeddingAnalyzer(matrix, nodes)
        communities = analyzer._detect_communities()
        modularity = analyzer._compute_modularity(communities)
        
        # Modularityは-1～1の範囲
        assert -1 <= modularity <= 1
    
    def test_similarity_matrix(self):
        """類似度行列計算テスト"""
        matrix = np.array([
            [0, 3, 0],
            [0, 0, 2],
            [1, 0, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = GraphEmbeddingAnalyzer(matrix, nodes, embedding_dim=8)
        walks = analyzer._generate_random_walks()
        embeddings = analyzer._compute_node_embeddings(walks)
        similarity_matrix = analyzer._compute_similarity_matrix(embeddings)
        
        # サイズ確認
        assert similarity_matrix.shape == (3, 3)
        
        # 対角要素は1に近い（自己類似度）
        for i in range(3):
            assert 0.9 <= similarity_matrix[i, i] <= 1.1
        
        # 対称行列
        for i in range(3):
            for j in range(3):
                assert abs(similarity_matrix[i, j] - similarity_matrix[j, i]) < 0.01
    
    def test_top_similar_pairs(self):
        """類似ペア抽出テスト"""
        similarity_matrix = np.array([
            [1.0, 0.9, 0.3],
            [0.9, 1.0, 0.2],
            [0.3, 0.2, 1.0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = GraphEmbeddingAnalyzer(np.zeros((3, 3)), nodes)
        analyzer.n = 3
        pairs = analyzer._find_top_similar_pairs(similarity_matrix, top_k=3)
        
        # 3ペア抽出（3ノードから3C2 = 3ペア）
        assert len(pairs) == 3
        
        # 最も類似度が高いペアは(A, B)
        assert pairs[0][0] in ["A", "B"]
        assert pairs[0][1] in ["A", "B"]
        assert pairs[0][2] == 0.9
    
    def test_2d_reduction(self):
        """2D座標生成テスト"""
        matrix = np.array([
            [0, 3, 2],
            [1, 0, 3],
            [2, 1, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = GraphEmbeddingAnalyzer(matrix, nodes, embedding_dim=16, reduction_method="mds")
        walks = analyzer._generate_random_walks()
        embeddings = analyzer._compute_node_embeddings(walks)
        positions_2d = analyzer._reduce_to_2d(embeddings)
        
        # 全ノードの座標が生成される
        assert len(positions_2d) == 3
        
        # 各座標は(x, y)のタプル
        for node in nodes:
            assert len(positions_2d[node]) == 2
            x, y = positions_2d[node]
            assert isinstance(x, float)
            assert isinstance(y, float)
    
    def test_community_labels(self):
        """コミュニティラベル生成テスト"""
        communities = {"A": 0, "B": 0, "C": 1, "D": 1, "E": 1}
        
        analyzer = GraphEmbeddingAnalyzer(np.zeros((5, 5)), ["A", "B", "C", "D", "E"])
        labels = analyzer._assign_community_labels(communities)
        
        # 2つのコミュニティ
        assert len(labels) == 2
        
        # ラベルにサイズ情報が含まれる
        assert "2ノード" in labels[0]
        assert "3ノード" in labels[1]
    
    def test_comprehensive_analysis(self):
        """包括的分析のテスト"""
        # 実際のネットワーク
        matrix = np.array([
            [0, 5, 3, 0, 0],
            [2, 0, 4, 1, 0],
            [1, 3, 0, 2, 0],
            [0, 0, 1, 0, 5],
            [0, 0, 0, 3, 0]
        ])
        nodes = ["A", "B", "C", "D", "E"]
        
        analyzer = GraphEmbeddingAnalyzer(matrix, nodes, embedding_dim=16, num_walks=50)
        result = analyzer.compute_graph_embedding()
        
        # 結果の構造確認
        assert isinstance(result.node_embeddings, dict)
        assert isinstance(result.node_positions_2d, dict)
        assert isinstance(result.communities, dict)
        assert isinstance(result.community_labels, dict)
        assert isinstance(result.similarity_matrix, np.ndarray)
        assert isinstance(result.modularity, float)
        assert isinstance(result.top_similar_pairs, list)
        assert isinstance(result.interpretation, str)
        assert result.computation_time > 0
        
        # 埋め込み次元確認
        assert result.embedding_dim == 16
        
        # 全ノードの埋め込みが存在
        assert len(result.node_embeddings) == 5
        
        # コミュニティ数
        assert result.n_communities > 0
        assert result.n_communities <= 5
    
    def test_empty_graph(self):
        """空グラフのテスト（エッジなし）"""
        matrix = np.zeros((3, 3))
        nodes = ["A", "B", "C"]
        
        analyzer = GraphEmbeddingAnalyzer(matrix, nodes, embedding_dim=8)
        result = analyzer.compute_graph_embedding()
        
        # エラーなく完了
        assert result is not None
        
        # 各ノードが独自のコミュニティ
        assert len(set(result.communities.values())) == 3
        
        # Modularity = 0
        assert abs(result.modularity) < 0.1
    
    def test_spectral_embedding(self):
        """Spectral Embedding手法のテスト"""
        matrix = np.array([
            [0, 3, 2],
            [1, 0, 3],
            [2, 1, 0]
        ])
        nodes = ["A", "B", "C"]
        
        analyzer = GraphEmbeddingAnalyzer(
            matrix, nodes, 
            embedding_dim=8, 
            reduction_method="spectral"
        )
        result = analyzer.compute_graph_embedding()
        
        # 2D座標が生成される
        assert len(result.node_positions_2d) == 3
        
        # 各座標は2次元
        for node in nodes:
            x, y = result.node_positions_2d[node]
            assert isinstance(x, float)
            assert isinstance(y, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
