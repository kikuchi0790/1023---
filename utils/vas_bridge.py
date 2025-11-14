"""
VAS System 3D Viewer Bridge for Process Insight Modeler

3DFAILURE-mainのVAS System 3D ViewerのJSON形式に変換するモジュール。
PIMのIDEF0構造とコミュニティ検出結果をVAS形式に変換し、
Three.jsベースの3D可視化を実現する。
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def convert_pim_to_vas(
    nodes: List[str],
    adjacency_matrix: np.ndarray,
    categories: List[str],
    idef0_data: Dict[str, Dict[str, Any]],
    score_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    PIMデータをVAS System 3D Viewer形式に変換
    
    Args:
        nodes: ノード名リスト
        adjacency_matrix: 隣接行列（N×N）
        categories: 機能カテゴリリスト
        idef0_data: IDEF0構造データ {category: {function, inputs, mechanisms, outputs}}
        score_threshold: エッジを含める最小スコア（絶対値）
    
    Returns:
        VAS形式のJSONデータ
    """
    logger.info(f"PIM→VAS変換開始: {len(nodes)}ノード")
    
    # ノードタイプマッピングを作成
    node_type_map = _create_node_type_map(nodes, categories, idef0_data)
    node_category_map = _create_node_category_map(nodes, categories, idef0_data)
    
    # VASノード生成
    vas_nodes = []
    for idx, node_name in enumerate(nodes):
        node_type = node_type_map.get(node_name, "System")
        category = node_category_map.get(node_name, "Unknown")
        level = _get_level_from_type(node_type)
        
        vas_node = {
            "id": str(idx),
            "labels": [node_type],
            "properties": {
                "name": node_name,
                "pim_node_id": idx,
                "level": level,
                "category": category,
                "source": "PIM",
                "number": f"PIM-{idx:04d}",
                "通しno": idx + 1
            }
        }
        
        # IDEF0詳細を追加
        idef0_details = _get_idef0_details_for_node(node_name, categories, idef0_data)
        if idef0_details:
            vas_node["properties"]["idef0"] = idef0_details
        
        vas_nodes.append(vas_node)
    
    # VASリンク生成
    vas_links = []
    link_id = 0
    
    for i, source_node in enumerate(nodes):
        for j, target_node in enumerate(nodes):
            if i == j:
                continue
            
            score = adjacency_matrix[i, j]
            if abs(score) < score_threshold:
                continue
            
            # リンクタイプを決定
            link_type = _determine_link_type(score, node_type_map.get(source_node, "System"))
            
            vas_link = {
                "id": str(link_id),
                "type": link_type,
                "startNode": str(i),
                "endNode": str(j),
                "properties": {
                    "source": "PIM",
                    "score": float(score),
                    "abs_score": abs(float(score)),
                    "direction": "positive" if score > 0 else "negative"
                }
            }
            
            vas_links.append(vas_link)
            link_id += 1
    
    logger.info(f"PIM→VAS変換完了: {len(vas_nodes)}ノード, {len(vas_links)}リンク")
    
    return {
        "nodes": vas_nodes,
        "links": vas_links
    }


def convert_community_to_vas(
    nodes: List[str],
    adjacency_matrix: np.ndarray,
    communities: Dict[str, int],
    community_labels: Dict[int, str],
    node_positions_2d: Dict[str, Tuple[float, float]],
    categories: Optional[List[str]] = None,
    idef0_data: Optional[Dict[str, Dict[str, Any]]] = None,
    score_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    コミュニティ検出結果をVAS System 3D Viewer形式に変換
    
    Args:
        nodes: ノード名リスト
        adjacency_matrix: 隣接行列（N×N）
        communities: ノード→コミュニティIDマッピング
        community_labels: コミュニティID→ラベルマッピング
        node_positions_2d: ノード→2D座標マッピング
        categories: 機能カテゴリリスト（オプション）
        idef0_data: IDEF0構造データ（オプション）
        score_threshold: エッジを含める最小スコア（絶対値）
    
    Returns:
        VAS形式のJSONデータ（コミュニティ別レイヤー配置）
    """
    logger.info(f"Community→VAS変換開始: {len(nodes)}ノード, {len(set(communities.values()))}コミュニティ")
    
    # VASノード生成（コミュニティ別）
    vas_nodes = []
    for idx, node_name in enumerate(nodes):
        comm_id = communities.get(node_name, 0)
        comm_label = community_labels.get(comm_id, f"コミュニティ{comm_id+1}")
        
        # 2D座標を取得
        pos_2d = node_positions_2d.get(node_name, (0, 0))
        
        vas_node = {
            "id": str(idx),
            "labels": ["CommunityNode"],
            "properties": {
                "name": node_name,
                "pim_node_id": idx,
                "level": comm_id,  # コミュニティIDをレベルとして使用
                "community_id": comm_id,
                "community_label": comm_label,
                "source": "PIM_Community",
                "number": f"COM-{comm_id}-{idx:04d}",
                "通しno": idx + 1,
                "pos_2d_x": float(pos_2d[0]),
                "pos_2d_y": float(pos_2d[1])
            }
        }
        
        # IDEF0詳細を追加（存在する場合）
        if categories and idef0_data:
            idef0_details = _get_idef0_details_for_node(node_name, categories, idef0_data)
            if idef0_details:
                vas_node["properties"]["idef0"] = idef0_details
        
        vas_nodes.append(vas_node)
    
    # VASリンク生成（コミュニティ内/間を区別）
    vas_links = []
    link_id = 0
    
    for i, source_node in enumerate(nodes):
        for j, target_node in enumerate(nodes):
            if i == j:
                continue
            
            score = adjacency_matrix[i, j]
            if abs(score) < score_threshold:
                continue
            
            source_comm = communities.get(source_node, 0)
            target_comm = communities.get(target_node, 0)
            is_intra_community = (source_comm == target_comm)
            
            # リンクタイプを決定
            if is_intra_community:
                link_type = "IntraCommunity_Information" if score > 0 else "IntraCommunity_Constraint"
            else:
                link_type = "InterCommunity_Information" if score > 0 else "InterCommunity_Constraint"
            
            vas_link = {
                "id": str(link_id),
                "type": link_type,
                "startNode": str(i),
                "endNode": str(j),
                "properties": {
                    "source": "PIM_Community",
                    "score": float(score),
                    "abs_score": abs(float(score)),
                    "direction": "positive" if score > 0 else "negative",
                    "is_intra_community": is_intra_community,
                    "source_community": source_comm,
                    "target_community": target_comm
                }
            }
            
            vas_links.append(vas_link)
            link_id += 1
    
    logger.info(f"Community→VAS変換完了: {len(vas_nodes)}ノード, {len(vas_links)}リンク")
    
    return {
        "nodes": vas_nodes,
        "links": vas_links
    }


def _create_node_type_map(
    nodes: List[str],
    categories: List[str],
    idef0_data: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    """
    ノード名→VASノードタイプのマッピングを作成
    
    IDEF0タイプマッピング:
    - Output → System (level=0, 緑色)
    - Mechanism → Process (level=1, 青色)
    - Input → Component (level=2, 橙色)
    """
    node_type_map = {}
    
    for category in categories:
        if category not in idef0_data:
            continue
        
        category_data = idef0_data[category]
        
        # Outputs → System
        outputs = category_data.get("outputs", [])
        for output in outputs:
            node_type_map[output] = "System"
        
        # Mechanisms → Process
        mechanisms = category_data.get("mechanisms", [])
        for mechanism in mechanisms:
            node_type_map[mechanism] = "Process"
        
        # Inputs → Component
        inputs = category_data.get("inputs", [])
        for input_node in inputs:
            node_type_map[input_node] = "Component"
    
    # マッピングされていないノードはSystemとして扱う
    for node in nodes:
        if node not in node_type_map:
            node_type_map[node] = "System"
    
    return node_type_map


def _create_node_category_map(
    nodes: List[str],
    categories: List[str],
    idef0_data: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    """
    ノード名→カテゴリのマッピングを作成
    """
    node_category_map = {}
    
    for category in categories:
        if category not in idef0_data:
            continue
        
        category_data = idef0_data[category]
        
        # 全てのノードをカテゴリにマッピング
        for node_list in [category_data.get("outputs", []),
                          category_data.get("mechanisms", []),
                          category_data.get("inputs", [])]:
            for node in node_list:
                node_category_map[node] = category
    
    return node_category_map


def _get_level_from_type(node_type: str) -> int:
    """
    ノードタイプからレベル（階層）を取得
    """
    level_map = {
        "System": 0,    # Output（性能指標）
        "Process": 1,   # Mechanism（手段・道具）
        "Component": 2  # Input（材料・情報）
    }
    return level_map.get(node_type, 0)


def _determine_link_type(score: float, source_type: str) -> str:
    """
    スコアとソースノードタイプからリンクタイプを決定
    
    正のスコア: Information/Energy（青系）
    負のスコア: Constraint/Risk（赤系）
    """
    if score > 0:
        # 正の影響
        if source_type == "System":
            return "Performance_Information"
        elif source_type == "Process":
            return "Process_Energy"
        else:
            return "Material_Information"
    else:
        # 負の影響
        if source_type == "System":
            return "Performance_Constraint"
        elif source_type == "Process":
            return "Process_Risk"
        else:
            return "Material_Constraint"


def _get_idef0_details_for_node(
    node_name: str,
    categories: List[str],
    idef0_data: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    ノードのIDEF0詳細情報を取得
    """
    for category in categories:
        if category not in idef0_data:
            continue
        
        category_data = idef0_data[category]
        
        # Outputsに含まれる場合
        if node_name in category_data.get("outputs", []):
            return {
                "type": "Output",
                "category": category,
                "function": category_data.get("function", category),
                "description": f"{category}の成果物・性能指標"
            }
        
        # Mechanismsに含まれる場合
        if node_name in category_data.get("mechanisms", []):
            return {
                "type": "Mechanism",
                "category": category,
                "function": category_data.get("function", category),
                "description": f"{category}を実行する手段・道具"
            }
        
        # Inputsに含まれる場合
        if node_name in category_data.get("inputs", []):
            return {
                "type": "Input",
                "category": category,
                "function": category_data.get("function", category),
                "description": f"{category}に必要な材料・情報"
            }
    
    return None


def generate_vas_layout_metadata(
    nodes: List[str],
    node_type_map: Dict[str, str]
) -> Dict[str, Any]:
    """
    VAS 3D Viewerのレイアウトメタデータを生成
    
    レベル別統計、ノードタイプ別統計などを含む
    """
    level_counts = {"0": 0, "1": 0, "2": 0}
    type_counts = {"System": 0, "Process": 0, "Component": 0}
    
    for node in nodes:
        node_type = node_type_map.get(node, "System")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        level = _get_level_from_type(node_type)
        level_counts[str(level)] = level_counts.get(str(level), 0) + 1
    
    return {
        "total_nodes": len(nodes),
        "level_counts": level_counts,
        "type_counts": type_counts,
        "layout": "IDEF0_Hierarchical"
    }
