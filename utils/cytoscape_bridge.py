"""
Cytoscape.js統合: PIMデータをCytoscape形式に変換

このモジュールは以下を行います:
1. PIMのノードと隣接行列をCytoscape JSON形式に変換
2. スコア閾値に基づくフィルタリング
3. IDEF0構造に基づくノード分類とレベル設定
"""

import numpy as np
from typing import List, Dict, Any, Optional


class CytoscapeConverter:
    """PIMデータをCytoscape形式に変換するクラス"""
    
    def __init__(self, threshold: float = 2.0):
        """
        Parameters
        ----------
        threshold : float
            エッジを表示する最小スコア閾値（デフォルト: 2.0）
        """
        self.threshold = threshold
    
    def convert(
        self,
        nodes: List[str],
        adjacency_matrix: np.ndarray,
        categories: List[str],
        idef0_data: Dict[str, Dict[str, Any]],
        use_hierarchical_layout: bool = False,
        network_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        メイン変換関数
        
        Parameters
        ----------
        nodes : List[str]
            ノード名のリスト
        adjacency_matrix : np.ndarray
            隣接行列（N×N）、値は-9～+9のスコア
        categories : List[str]
            機能カテゴリのリスト（時系列順）
        idef0_data : Dict[str, Dict[str, Any]]
            IDEF0データ（カテゴリごとのInput/Mechanism/Output）
        use_hierarchical_layout : bool
            階層的レイアウト用の座標を計算するか（デフォルト: False）
        network_metrics : Optional[Dict[str, Dict[str, float]]]
            ネットワーク分析メトリクス（ノード名 → {pagerank, betweenness, ...}）
        
        Returns
        -------
        Dict[str, Any]
            Cytoscape互換のJSONデータ
        """
        # ノードの分類とレベル決定
        node_info = self._classify_nodes(nodes, categories, idef0_data)
        
        # 階層的レイアウト用の座標計算
        positions = {}
        if use_hierarchical_layout:
            positions = self._calculate_2d_positions(nodes, categories, idef0_data)
        
        # 閾値以上のエッジのみ抽出
        filtered_edges = self._filter_edges(nodes, adjacency_matrix)
        
        # フィルタ済エッジが参照するノードIDを収集
        connected_node_ids = set()
        for edge in filtered_edges:
            connected_node_ids.add(edge["source_idx"])
            connected_node_ids.add(edge["target_idx"])
        
        # Cytoscapeノード作成（接続されているもののみ）
        cyto_nodes = []
        for idx, node_name in enumerate(nodes):
            if idx in connected_node_ids:
                info = node_info.get(node_name, {})
                node_data = {
                    "id": f"n{idx}",
                    "name": node_name,
                    "label": info.get("label", "unknown"),
                    "level": info.get("level", 1),
                    "category": info.get("category", "")
                }
                
                # ネットワークメトリクスを追加
                if network_metrics and node_name in network_metrics:
                    metrics = network_metrics[node_name]
                    node_data["pagerank"] = metrics.get("pagerank", 0)
                    node_data["betweenness"] = metrics.get("betweenness", 0)
                    node_data["in_degree"] = metrics.get("in_degree", 0)
                    node_data["out_degree"] = metrics.get("out_degree", 0)
                
                # 位置情報を追加（階層的レイアウト時のみ）
                node_obj = {"data": node_data}
                if node_name in positions:
                    node_obj["position"] = positions[node_name]
                
                cyto_nodes.append(node_obj)
        
        # Cytoscapeエッジ作成
        cyto_edges = []
        for edge_idx, edge in enumerate(filtered_edges):
            cyto_edges.append({
                "data": {
                    "id": f"e{edge_idx}",
                    "source": f"n{edge['source_idx']}",
                    "target": f"n{edge['target_idx']}",
                    "label": f"{edge['score']:.1f}",
                    "name": f"{edge['score']:.1f}"
                }
            })
        
        return {
            "nodes": cyto_nodes,
            "edges": cyto_edges
        }
    
    def _classify_nodes(
        self,
        nodes: List[str],
        categories: List[str],
        idef0_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        ノードを分類してラベルとレベルを決定
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            ノード名 → {label, level, category}
        """
        node_info = {}
        
        for category in categories:
            if category not in idef0_data:
                continue
            
            idef0 = idef0_data[category]
            
            # Input: level=2, label="input"
            if idef0.get("inputs"):
                for input_node in idef0["inputs"]:
                    node_info[input_node] = {
                        "label": "input",
                        "level": 2,
                        "category": category
                    }
            
            # Mechanism: level=1, label="mechanism"
            if idef0.get("mechanisms"):
                for mech_node in idef0["mechanisms"]:
                    node_info[mech_node] = {
                        "label": "mechanism",
                        "level": 1,
                        "category": category
                    }
            
            # Output: level=0, label="output"
            if idef0.get("outputs"):
                for output_node in idef0["outputs"]:
                    node_info[output_node] = {
                        "label": "output",
                        "level": 0,
                        "category": category
                    }
        
        return node_info
    
    def _filter_edges(
        self,
        nodes: List[str],
        adjacency_matrix: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        閾値以上のスコアを持つエッジのみ抽出
        
        Returns
        -------
        List[Dict[str, Any]]
            [{"source_idx": i, "target_idx": j, "score": score}, ...]
        """
        filtered = []
        
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                score = adjacency_matrix[i, j]
                if abs(score) >= self.threshold:
                    filtered.append({
                        "source_idx": i,
                        "target_idx": j,
                        "score": float(score)
                    })
        
        return filtered
    
    def _calculate_2d_positions(
        self,
        nodes: List[str],
        categories: List[str],
        idef0_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        3D配置をXY平面に投影して階層的レイアウト用の座標を計算
        
        X軸: カテゴリ順（0, 200, 400, ...）
        Y軸: 層別（Input=0（下）, Mechanism=100, Output=200, Function=300（上））
        
        Parameters
        ----------
        nodes : List[str]
            ノード名のリスト
        categories : List[str]
            機能カテゴリのリスト（時系列順）
        idef0_data : Dict[str, Dict[str, Any]]
            IDEF0データ（カテゴリごとのInput/Mechanism/Output）
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            ノード名 → {"x": float, "y": float}
        """
        positions = {}
        CATEGORY_SPACING_X = 200
        LAYER_Y = {
            "input": 0,        # 最下層
            "mechanism": 100,
            "output": 200,
            "function": 300    # 最上層
        }
        NODE_OFFSET_X = 30  # 同じ層内の複数ノード間隔
        
        for cat_idx, category in enumerate(categories):
            base_x = cat_idx * CATEGORY_SPACING_X
            
            if category not in idef0_data:
                continue
            
            idef0 = idef0_data[category]
            
            # Function nodes（最上層）
            func_name = idef0.get("function")
            if func_name:
                positions[func_name] = {
                    "x": base_x,
                    "y": LAYER_Y["function"]
                }
            
            # Input nodes（最下層）
            if idef0.get("inputs"):
                inputs = idef0["inputs"]
                num_inputs = len(inputs)
                for i, input_node in enumerate(inputs):
                    offset_x = (i - (num_inputs - 1) / 2) * NODE_OFFSET_X
                    positions[input_node] = {
                        "x": base_x + offset_x,
                        "y": LAYER_Y["input"]
                    }
            
            # Mechanism nodes（中層下）
            if idef0.get("mechanisms"):
                mechanisms = idef0["mechanisms"]
                num_mechs = len(mechanisms)
                for i, mech_node in enumerate(mechanisms):
                    offset_x = (i - (num_mechs - 1) / 2) * NODE_OFFSET_X
                    positions[mech_node] = {
                        "x": base_x + offset_x,
                        "y": LAYER_Y["mechanism"]
                    }
            
            # Output nodes（中層上）
            if idef0.get("outputs"):
                outputs = idef0["outputs"]
                num_outputs = len(outputs)
                for i, output_node in enumerate(outputs):
                    offset_x = (i - (num_outputs - 1) / 2) * NODE_OFFSET_X
                    positions[output_node] = {
                        "x": base_x + offset_x,
                        "y": LAYER_Y["output"]
                    }
        
        return positions


def convert_pim_to_cytoscape(
    nodes: List[str],
    adjacency_matrix: np.ndarray,
    categories: List[str],
    idef0_data: Dict[str, Dict[str, Any]],
    threshold: float = 2.0,
    use_hierarchical_layout: bool = False,
    network_metrics: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, Any]:
    """
    便利関数: PIMデータをCytoscape形式に変換
    
    Parameters
    ----------
    nodes : List[str]
        ノード名のリスト
    adjacency_matrix : np.ndarray
        隣接行列
    categories : List[str]
        機能カテゴリのリスト
    idef0_data : Dict[str, Dict[str, Any]]
        IDEF0データ
    threshold : float
        エッジ表示の閾値
    use_hierarchical_layout : bool
        階層的レイアウト用の座標を計算するか
    network_metrics : Optional[Dict[str, Dict[str, float]]]
        ネットワーク分析メトリクス（ノード名 → {pagerank, betweenness, ...}）
    
    Returns
    -------
    Dict[str, Any]
        Cytoscape JSON
    """
    converter = CytoscapeConverter(threshold=threshold)
    return converter.convert(
        nodes, adjacency_matrix, categories, idef0_data,
        use_hierarchical_layout=use_hierarchical_layout,
        network_metrics=network_metrics
    )
