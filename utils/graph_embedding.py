"""
Graph Embedding Analysis for Process Insight Modeler
潜在構造発見（Graph Embedding + Community Detection）

表面的な接続を超えた「本質的な類似性」を発見し、機能的なグループを自動検出する。
"""

from typing import List, Dict, Tuple, Any, Callable
import numpy as np
import pandas as pd
import networkx as nx
import logging
from dataclasses import dataclass
import time
from sklearn.manifold import MDS, SpectralEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain

logger = logging.getLogger(__name__)


@dataclass
class GraphEmbeddingResult:
    """Graph Embedding分析の結果"""
    node_embeddings: Dict[str, np.ndarray]  # 高次元埋め込み（ノード→ベクトル）
    node_positions_2d: Dict[str, Tuple[float, float]]  # 2D座標
    communities: Dict[str, int]  # ノード→コミュニティID
    community_labels: Dict[int, str]  # コミュニティID→名前
    similarity_matrix: np.ndarray  # 埋め込み空間での類似度行列
    modularity: float  # コミュニティ検出の品質（-1～1）
    top_similar_pairs: List[Tuple[str, str, float]]  # 類似ノードペア
    computation_time: float
    embedding_dim: int
    n_communities: int
    interpretation: str


class GraphEmbeddingAnalyzer:
    """
    Graph Embedding分析クラス
    
    ランダムウォークベースのグラフ埋め込みとコミュニティ検出:
    - Node2Vec風の手法でノードを低次元ベクトルに変換
    - Louvainアルゴリズムでコミュニティ検出
    - 2D可視化で直感的理解を支援
    """
    
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_names: List[str],
        embedding_dim: int = 64,
        walk_length: int = 20,
        num_walks: int = 100,
        reduction_method: str = "mds"
    ):
        """
        Args:
            adjacency_matrix: 隣接行列（N×N）
            node_names: ノード名リスト
            embedding_dim: 埋め込み次元数
            walk_length: ランダムウォークの長さ
            num_walks: 各ノードからのウォーク回数
            reduction_method: 2D化手法（"mds" or "spectral"）
        """
        self.matrix = adjacency_matrix.copy()
        self.node_names = node_names
        self.n = len(node_names)
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.reduction_method = reduction_method
        
        # グラフ構築（重み付き有向グラフ）
        self.graph = nx.DiGraph()
        
        # 全ノードを追加（エッジがない場合でもノードは存在）
        for node in node_names:
            self.graph.add_node(node)
        
        # エッジを追加
        for i, source in enumerate(node_names):
            for j, target in enumerate(node_names):
                if i != j and abs(self.matrix[i, j]) > 1e-6:
                    # 重みは絶対値を使用（負の影響も考慮）
                    weight = abs(self.matrix[i, j])
                    self.graph.add_edge(source, target, weight=weight)
        
        # 無向グラフ（コミュニティ検出用）
        self.undirected_graph = self.graph.to_undirected()
        
        logger.info(f"GraphEmbeddingAnalyzer初期化: {self.n}ノード, 埋め込み次元{embedding_dim}")
    
    def compute_graph_embedding(
        self,
        progress_callback: Callable[[str, float], None] = None
    ) -> GraphEmbeddingResult:
        """
        包括的なGraph Embedding分析を実行
        
        Args:
            progress_callback: 進捗コールバック(message, pct)
        
        Returns:
            GraphEmbeddingResult
        """
        start_time = time.time()
        
        # 1. ランダムウォークベース埋め込み
        if progress_callback:
            progress_callback("ランダムウォーク生成中...", 0.0)
        
        walks = self._generate_random_walks()
        
        if progress_callback:
            progress_callback("ノード埋め込み計算中...", 0.3)
        
        node_embeddings = self._compute_node_embeddings(walks)
        
        # 2. コミュニティ検出
        if progress_callback:
            progress_callback("コミュニティ検出中...", 0.5)
        
        communities = self._detect_communities()
        modularity = self._compute_modularity(communities)
        
        # 3. 類似度計算
        if progress_callback:
            progress_callback("類似度計算中...", 0.6)
        
        similarity_matrix = self._compute_similarity_matrix(node_embeddings)
        top_similar_pairs = self._find_top_similar_pairs(similarity_matrix)
        
        # 4. 2D座標生成
        if progress_callback:
            progress_callback("2D座標生成中...", 0.8)
        
        node_positions_2d = self._reduce_to_2d(node_embeddings)
        
        # 5. コミュニティラベル生成
        community_labels = self._assign_community_labels(communities)
        n_communities = len(set(communities.values()))
        
        computation_time = time.time() - start_time
        logger.info(f"Graph Embedding分析完了: {computation_time:.2f}秒, {n_communities}コミュニティ")
        
        # 6. 解釈文生成
        interpretation = self._generate_interpretation(
            node_embeddings, communities, modularity,
            top_similar_pairs, n_communities
        )
        
        return GraphEmbeddingResult(
            node_embeddings=node_embeddings,
            node_positions_2d=node_positions_2d,
            communities=communities,
            community_labels=community_labels,
            similarity_matrix=similarity_matrix,
            modularity=modularity,
            top_similar_pairs=top_similar_pairs,
            computation_time=computation_time,
            embedding_dim=self.embedding_dim,
            n_communities=n_communities,
            interpretation=interpretation
        )
    
    def _generate_random_walks(self) -> List[List[str]]:
        """
        ランダムウォークを生成（Node2Vec風）
        
        Returns:
            ウォークのリスト（各ウォークはノード名のリスト）
        """
        walks = []
        
        for node in self.node_names:
            for _ in range(self.num_walks):
                walk = [node]
                current = node
                
                for _ in range(self.walk_length - 1):
                    neighbors = list(self.graph.neighbors(current))
                    
                    if not neighbors:
                        break
                    
                    # 重み付き確率でサンプリング
                    weights = [self.graph[current][neighbor]['weight'] 
                              for neighbor in neighbors]
                    total_weight = sum(weights)
                    
                    if total_weight > 0:
                        probs = [w / total_weight for w in weights]
                        current = np.random.choice(neighbors, p=probs)
                        walk.append(current)
                    else:
                        break
                
                walks.append(walk)
        
        logger.info(f"ランダムウォーク生成完了: {len(walks)}本")
        return walks
    
    def _compute_node_embeddings(self, walks: List[List[str]]) -> Dict[str, np.ndarray]:
        """
        ウォークからノード埋め込みを計算（簡易Skip-gram風）
        
        Args:
            walks: ランダムウォークのリスト
        
        Returns:
            ノード名→埋め込みベクトル
        """
        # 初期化（ランダムベクトル）
        embeddings = {
            node: np.random.randn(self.embedding_dim) * 0.01
            for node in self.node_names
        }
        
        # 共起行列を作成
        cooccurrence = {node: {other: 0 for other in self.node_names} 
                       for node in self.node_names}
        
        window_size = 5
        for walk in walks:
            for i, node in enumerate(walk):
                # ウィンドウ内の共起をカウント
                start = max(0, i - window_size)
                end = min(len(walk), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_node = walk[j]
                        cooccurrence[node][context_node] += 1
        
        # 共起行列から特徴ベクトルを生成
        for node in self.node_names:
            context_vector = np.array([
                cooccurrence[node][other] for other in self.node_names
            ], dtype=float)
            
            # 正規化
            norm = np.linalg.norm(context_vector)
            if norm > 0:
                context_vector /= norm
            
            # 次元削減（SVD的なアプローチ）
            if context_vector.sum() > 0:
                # 共起ベクトルを埋め込み次元に射影
                projection = np.random.randn(len(self.node_names), self.embedding_dim)
                embeddings[node] = context_vector @ projection
                
                # 正規化
                norm = np.linalg.norm(embeddings[node])
                if norm > 0:
                    embeddings[node] /= norm
        
        return embeddings
    
    def _detect_communities(self) -> Dict[str, int]:
        """
        Louvainアルゴリズムでコミュニティ検出
        
        Returns:
            ノード名→コミュニティID
        """
        if len(self.undirected_graph.edges()) == 0:
            # エッジがない場合は各ノードが独自のコミュニティ
            return {node: i for i, node in enumerate(self.node_names)}
        
        # Louvainアルゴリズム
        partition = community_louvain.best_partition(self.undirected_graph, weight='weight')
        
        logger.info(f"コミュニティ検出完了: {len(set(partition.values()))}コミュニティ")
        return partition
    
    def _compute_modularity(self, communities: Dict[str, int]) -> float:
        """
        Modularityを計算（コミュニティ検出の品質指標）
        
        Returns:
            Modularity値（-1～1、高いほど良い）
        """
        if len(self.undirected_graph.edges()) == 0:
            return 0.0
        
        modularity = community_louvain.modularity(communities, self.undirected_graph, weight='weight')
        return modularity
    
    def _compute_similarity_matrix(
        self,
        node_embeddings: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        埋め込み空間でのコサイン類似度行列を計算
        
        Returns:
            類似度行列（N×N）
        """
        embedding_matrix = np.array([node_embeddings[node] for node in self.node_names])
        similarity = cosine_similarity(embedding_matrix)
        
        return similarity
    
    def _find_top_similar_pairs(
        self,
        similarity_matrix: np.ndarray,
        top_k: int = 20
    ) -> List[Tuple[str, str, float]]:
        """
        最も類似度が高いノードペアを抽出
        
        Args:
            similarity_matrix: 類似度行列
            top_k: 抽出数
        
        Returns:
            [(node1, node2, similarity), ...]
        """
        pairs = []
        
        for i in range(self.n):
            for j in range(i + 1, self.n):
                similarity = similarity_matrix[i, j]
                pairs.append((self.node_names[i], self.node_names[j], similarity))
        
        # 類似度降順でソート
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs[:top_k]
    
    def _reduce_to_2d(
        self,
        node_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, Tuple[float, float]]:
        """
        高次元埋め込みを2Dに削減
        
        Returns:
            ノード名→(x, y)座標
        """
        embedding_matrix = np.array([node_embeddings[node] for node in self.node_names])
        
        # ノード数が少ない場合はMDSを使用（Spectralはn_components << n_samplesが必要）
        if self.reduction_method == "spectral" and self.n > 10:
            try:
                reducer = SpectralEmbedding(n_components=2, random_state=42)
                coords_2d = reducer.fit_transform(embedding_matrix)
            except Exception as e:
                # SpectralEmbeddingが失敗した場合はMDSにフォールバック
                logger.warning(f"SpectralEmbedding failed, falling back to MDS: {e}")
                reducer = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
                coords_2d = reducer.fit_transform(embedding_matrix)
        else:
            # MDS or fallback
            reducer = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
            coords_2d = reducer.fit_transform(embedding_matrix)
        
        positions = {
            self.node_names[i]: (float(coords_2d[i, 0]), float(coords_2d[i, 1]))
            for i in range(self.n)
        }
        
        return positions
    
    def _assign_community_labels(self, communities: Dict[str, int]) -> Dict[int, str]:
        """
        コミュニティに名前を付ける
        
        Returns:
            コミュニティID→ラベル
        """
        # コミュニティごとにメンバーをグループ化
        community_members = {}
        for node, comm_id in communities.items():
            if comm_id not in community_members:
                community_members[comm_id] = []
            community_members[comm_id].append(node)
        
        # 各コミュニティに名前を付ける
        labels = {}
        for comm_id, members in community_members.items():
            # サイズに応じたラベル
            size = len(members)
            labels[comm_id] = f"コミュニティ{comm_id+1} ({size}ノード)"
        
        return labels
    
    def _generate_interpretation(
        self,
        node_embeddings: Dict[str, np.ndarray],
        communities: Dict[str, int],
        modularity: float,
        top_similar_pairs: List[Tuple[str, str, float]],
        n_communities: int
    ) -> str:
        """平易な日本語の解釈文を生成"""
        
        # コミュニティ別メンバー数
        comm_sizes = {}
        for node, comm_id in communities.items():
            comm_sizes[comm_id] = comm_sizes.get(comm_id, 0) + 1
        
        largest_comm_size = max(comm_sizes.values()) if comm_sizes else 0
        smallest_comm_size = min(comm_sizes.values()) if comm_sizes else 0
        
        interpretation = f"""
## 🔍 Graph Embedding分析結果の解釈

### 潜在構造の発見

グラフ埋め込みにより、**{self.n}ノード**を**{self.embedding_dim}次元**の潜在空間に変換しました。
この空間では、表面的な接続を超えた「本質的な類似性」が距離として表現されます。

### 🎯 コミュニティ検出結果

**検出されたコミュニティ数**: {n_communities}
**Modularity**: {modularity:.3f}

**Modularityの解釈**:
- 0.3以上: 明確なコミュニティ構造が存在 ✅
- 0.1-0.3: 弱いコミュニティ構造
- 0.1未満: ほぼランダムな構造

"""
        
        if modularity >= 0.3:
            interpretation += f"""
✅ **このネットワークには明確なコミュニティ構造が存在します。**

各コミュニティは、内部で密に協力し、外部とは疎な関係を持つ機能グループです。
最大コミュニティ: {largest_comm_size}ノード
最小コミュニティ: {smallest_comm_size}ノード
"""
        elif modularity >= 0.1:
            interpretation += """
⚠️ **弱いコミュニティ構造が検出されました。**

ノード間の協力関係は比較的均質で、明確なグループ分けは難しい状況です。
"""
        else:
            interpretation += """
❌ **明確なコミュニティ構造は見られません。**

すべてのノードが均等に相互作用しており、特定のグループは形成されていません。
"""
        
        interpretation += """
### 🔗 類似ノードペア（上位5組）

埋め込み空間で最も近い位置にあるノードペア:

"""
        
        for i, (node1, node2, sim) in enumerate(top_similar_pairs[:5], 1):
            interpretation += f"{i}. **{node1}** ⟷ **{node2}**: 類似度 {sim:.3f}\n"
        
        interpretation += """
### 💡 活用方法

1. **プロセス再設計**: 同じコミュニティ内のノードをグループ化して管理
2. **類似ノードの統合**: 類似度が高いノードは機能が重複している可能性
3. **組織構造の最適化**: コミュニティ境界を組織境界に対応させる
4. **知識移転**: 同じコミュニティ内でベストプラクティスを共有

### 📊 2D可視化の見方

- **近くに配置されたノード**: 機能的に類似
- **同じ色のノード**: 同じコミュニティ（協力関係が強い）
- **離れた位置のノード**: 機能的に独立

### 注意事項

- 次元削減により情報の一部が失われています（可視化の制約）
- コミュニティ検出は確率的アルゴリズムのため、実行ごとに結果が微妙に変わる可能性があります
- 埋め込みの品質は元のグラフ構造の複雑さに依存します
"""
        
        return interpretation.strip()
