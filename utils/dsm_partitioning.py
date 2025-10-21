"""
DSM Partitioning and Design Sequence Analysis
DSMパーティショニングと設計順序分析

モジュール化、デザインシーケンス導出、手戻り検出を実装
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse.csgraph import reverse_cuthill_mckee
import logging

logger = logging.getLogger(__name__)


class DSMPartitioner:
    """
    DSM行列のパーティショニング分析
    
    機能:
    1. モジュール検出（階層的クラスタリング）
    2. デザインシーケンス導出（最適設計順序）
    3. フィードバックループ検出（手戻り箇所）
    4. ブロック対角化（Bandwidth Minimization）
    """
    
    def __init__(self, dsm_matrix: np.ndarray, node_names: List[str]):
        """
        Args:
            dsm_matrix: DSM行列（N×N）
            node_names: ノード名リスト
        """
        self.matrix = dsm_matrix.copy()
        self.node_names = node_names
        self.n = len(node_names)
        
        if self.matrix.shape[0] != self.n or self.matrix.shape[1] != self.n:
            raise ValueError(f"行列サイズとノード数が不一致: {self.matrix.shape} vs {self.n}")
        
        self.modules = None
        self.design_sequence = None
        self.feedback_loops = None
        self.partitioned_matrix = None
        self.partitioned_nodes = None
    
    def detect_modules(self, n_clusters: int = None, method: str = "ward") -> Dict[str, Any]:
        """
        階層的クラスタリングによるモジュール検出
        
        Args:
            n_clusters: モジュール数（Noneの場合は自動決定）
            method: クラスタリング手法（'ward', 'average', 'complete'）
        
        Returns:
            モジュール情報の辞書
        """
        binary_matrix = (self.matrix != 0).astype(int)
        symmetric_matrix = binary_matrix + binary_matrix.T
        
        distance_matrix = 1 - (symmetric_matrix / np.max(symmetric_matrix))
        np.fill_diagonal(distance_matrix, 0)
        
        condensed_distance = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                condensed_distance.append(distance_matrix[i, j])
        
        if len(condensed_distance) == 0:
            self.modules = {i: 0 for i in range(self.n)}
            return {
                "n_modules": 1,
                "module_labels": [0] * self.n,
                "module_sizes": {0: self.n}
            }
        
        Z = linkage(condensed_distance, method=method)
        
        if n_clusters is None:
            n_clusters = max(2, min(7, int(np.sqrt(self.n / 2))))
        
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        labels = labels - 1
        
        self.modules = {i: labels[i] for i in range(self.n)}
        
        module_sizes = {}
        for module_id in range(n_clusters):
            module_sizes[module_id] = np.sum(labels == module_id)
        
        logger.info(f"検出されたモジュール数: {n_clusters}")
        for module_id, size in module_sizes.items():
            logger.info(f"  モジュール{module_id}: {size}ノード")
        
        return {
            "n_modules": n_clusters,
            "module_labels": labels.tolist(),
            "module_sizes": module_sizes,
            "linkage_matrix": Z
        }
    
    def compute_design_sequence(self) -> Dict[str, Any]:
        """
        デザインシーケンス（最適設計順序）を計算
        
        Reverse Cuthill-McKeeアルゴリズムで帯幅最小化
        
        Returns:
            設計順序情報の辞書
        """
        binary_matrix = (self.matrix != 0).astype(int)
        
        try:
            perm = reverse_cuthill_mckee(binary_matrix, symmetric_mode=False)
            
            if len(perm) == 0:
                perm = np.arange(self.n)
        except:
            perm = np.arange(self.n)
        
        self.design_sequence = perm.tolist()
        
        reordered_matrix = self.matrix[perm, :][:, perm]
        reordered_nodes = [self.node_names[i] for i in perm]
        
        self.partitioned_matrix = reordered_matrix
        self.partitioned_nodes = reordered_nodes
        
        logger.info(f"デザインシーケンス: {reordered_nodes[:5]}... ({self.n}ノード)")
        
        return {
            "sequence": perm.tolist(),
            "reordered_matrix": reordered_matrix,
            "reordered_nodes": reordered_nodes
        }
    
    def detect_feedback_loops(self) -> Dict[str, Any]:
        """
        フィードバックループ（手戻り）を検出
        
        Returns:
            フィードバックループ情報の辞書
        """
        if self.partitioned_matrix is None:
            self.compute_design_sequence()
        
        matrix = self.partitioned_matrix
        nodes = self.partitioned_nodes
        
        feedforward_elements = []
        feedback_elements = []
        diagonal_elements = []
        
        for i in range(self.n):
            for j in range(self.n):
                if matrix[i, j] != 0:
                    if i == j:
                        diagonal_elements.append({
                            "from": nodes[i],
                            "to": nodes[j],
                            "value": float(matrix[i, j]),
                            "position": (i, j)
                        })
                    elif i < j:
                        feedforward_elements.append({
                            "from": nodes[i],
                            "to": nodes[j],
                            "value": float(matrix[i, j]),
                            "position": (i, j)
                        })
                    else:
                        feedback_elements.append({
                            "from": nodes[i],
                            "to": nodes[j],
                            "value": float(matrix[i, j]),
                            "position": (i, j)
                        })
        
        total_nonzero = len(feedforward_elements) + len(feedback_elements) + len(diagonal_elements)
        
        if total_nonzero > 0:
            feedback_ratio = len(feedback_elements) / total_nonzero
        else:
            feedback_ratio = 0.0
        
        self.feedback_loops = feedback_elements
        
        logger.info(f"フィードフォワード要素: {len(feedforward_elements)}")
        logger.info(f"フィードバック要素（手戻り）: {len(feedback_elements)}")
        logger.info(f"フィードバック比率: {feedback_ratio:.1%}")
        
        return {
            "feedforward_count": len(feedforward_elements),
            "feedback_count": len(feedback_elements),
            "diagonal_count": len(diagonal_elements),
            "feedback_ratio": feedback_ratio,
            "feedforward_elements": feedforward_elements,
            "feedback_elements": feedback_elements,
            "diagonal_elements": diagonal_elements
        }
    
    def compute_modularity_score(self) -> float:
        """
        モジュラリティスコアを計算
        
        Returns:
            モジュラリティスコア（-1～1、高いほど良いモジュール分割）
        """
        if self.modules is None:
            raise ValueError("先にdetect_modules()を実行してください")
        
        binary_matrix = (self.matrix != 0).astype(int)
        symmetric_matrix = binary_matrix + binary_matrix.T
        
        m = np.sum(symmetric_matrix) / 2
        
        if m == 0:
            return 0.0
        
        Q = 0.0
        
        for i in range(self.n):
            for j in range(self.n):
                if self.modules[i] == self.modules[j]:
                    A_ij = symmetric_matrix[i, j]
                    k_i = np.sum(symmetric_matrix[i, :])
                    k_j = np.sum(symmetric_matrix[j, :])
                    Q += A_ij - (k_i * k_j) / (2 * m)
        
        Q /= (2 * m)
        
        logger.info(f"モジュラリティスコア: {Q:.3f}")
        
        return Q
    
    def analyze_module_coupling(self) -> pd.DataFrame:
        """
        モジュール間の結合度を分析
        
        Returns:
            モジュール間結合度のDataFrame
        """
        if self.modules is None:
            raise ValueError("先にdetect_modules()を実行してください")
        
        n_modules = max(self.modules.values()) + 1
        
        coupling_matrix = np.zeros((n_modules, n_modules))
        
        binary_matrix = (self.matrix != 0).astype(int)
        
        for i in range(self.n):
            for j in range(self.n):
                if binary_matrix[i, j] > 0:
                    module_i = self.modules[i]
                    module_j = self.modules[j]
                    coupling_matrix[module_i, module_j] += 1
        
        module_names = [f"モジュール{i}" for i in range(n_modules)]
        
        coupling_df = pd.DataFrame(
            coupling_matrix,
            index=module_names,
            columns=module_names
        )
        
        return coupling_df
    
    def get_module_members(self) -> Dict[int, List[str]]:
        """
        各モジュールのメンバーノードを取得
        
        Returns:
            モジュールID -> ノード名リストの辞書
        """
        if self.modules is None:
            raise ValueError("先にdetect_modules()を実行してください")
        
        module_members = {}
        
        for node_idx, module_id in self.modules.items():
            if module_id not in module_members:
                module_members[module_id] = []
            module_members[module_id].append(self.node_names[node_idx])
        
        return module_members
    
    def full_analysis(self, n_clusters: int = None) -> Dict[str, Any]:
        """
        完全な分析を実行（モジュール検出 + デザインシーケンス + フィードバック検出）
        
        Args:
            n_clusters: モジュール数（Noneの場合は自動決定）
        
        Returns:
            全分析結果の辞書
        """
        logger.info("=== DSMパーティショニング分析開始 ===")
        
        module_info = self.detect_modules(n_clusters=n_clusters)
        
        sequence_info = self.compute_design_sequence()
        
        feedback_info = self.detect_feedback_loops()
        
        modularity = self.compute_modularity_score()
        
        coupling_df = self.analyze_module_coupling()
        
        module_members = self.get_module_members()
        
        logger.info("=== 分析完了 ===")
        
        return {
            "modules": module_info,
            "design_sequence": sequence_info,
            "feedback_loops": feedback_info,
            "modularity_score": modularity,
            "coupling_matrix": coupling_df,
            "module_members": module_members
        }
