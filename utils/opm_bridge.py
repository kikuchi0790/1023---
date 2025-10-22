"""
OPM Bridge: PIMデータをOPM形式に変換

ShimadaSystemの3D OPMモデリング機能と統合するため、
PIMデータ（IDEF0形式）をOPM形式（layers, nodes, edges）に変換します。
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import math


def _to_python_type(value):
    """Convert numpy types to native Python types for JSON serialization"""
    if hasattr(value, 'item'):  # numpy scalar
        return value.item()
    return value


# カテゴリ→レイヤー色のデフォルトパレット
DEFAULT_LAYER_COLORS = [
    "#CC7F30",  # ビジネス（橙）
    "#3bc3ff",  # プロセス（青）
    "#70e483",  # オブジェクト（緑）
    "#731A3D",  # 変数（紫）
    "#F4DA24",  # コスト（黄）
    "#0000ff",  # 追加レイヤー1（青）
    "#ff00ff",  # 追加レイヤー2（マゼンタ）
    "#00ffff",  # 追加レイヤー3（シアン）
]

# IDEF0ノードタイプ→色マッピング
NODE_TYPE_COLORS = {
    "Output": "#70e483",     # 緑
    "Mechanism": "#3bc3ff",  # 青
    "Input": "#CC7F30",      # オレンジ
}


def convert_pim_to_opm(
    nodes: List[str],
    adjacency_matrix: np.ndarray,
    categories: List[str],
    idef0_data: Dict[str, Dict[str, Any]],
    scale: float = 10.0
) -> Dict[str, Any]:
    """
    PIMデータをOPM形式に変換
    
    Parameters:
    -----------
    nodes : List[str]
        PIMノードリスト
    adjacency_matrix : np.ndarray
        隣接行列（N×N、スコア: -9〜+9）
    categories : List[str]
        機能カテゴリリスト
    idef0_data : Dict[str, Dict[str, Any]]
        カテゴリごとのIDEF0データ
    scale : float
        3D空間のスケール（デフォルト: 10.0）
    
    Returns:
    --------
    Dict[str, Any]
        OPM形式のデータ: {
            "layers": List[{layer: str, color: str, isvisible: bool}],
            "nodes": List[{key: str, node: {name: str, layer: str}, position: {x, y, z}, isvisible: bool}],
            "edges": List[{key: str, type: str, edge: {...}, position: {...}, isvisible: bool}],
            "planeData": {m: int, d: int},
            "colorList": List[str]
        }
    """
    
    # 1. レイヤー生成（カテゴリ = レイヤー）
    layers = []
    for i, category in enumerate(categories):
        color = DEFAULT_LAYER_COLORS[i % len(DEFAULT_LAYER_COLORS)]
        layers.append({
            "layer": category,
            "color": color,
            "isvisible": True
        })
    
    # 2. ノード→カテゴリマッピング作成
    node_to_category = _build_node_category_mapping(nodes, categories, idef0_data)
    node_to_type = _build_node_type_mapping(nodes, idef0_data)
    
    # 3. 3D座標を自動計算
    node_positions = _calculate_3d_layout(
        nodes, 
        categories, 
        node_to_category, 
        adjacency_matrix,
        scale=scale
    )
    
    # 4. OPMノードデータ生成
    opm_nodes = []
    for node_name in nodes:
        category = node_to_category.get(node_name, categories[0] if categories else "Unknown")
        node_type = node_to_type.get(node_name, "Output")
        position = node_positions[node_name]
        
        opm_nodes.append({
            "key": f"{node_name} ( {category} )",
            "node": {
                "name": node_name,
                "layer": category
            },
            "position": position,
            "isvisible": True,
            "nodeType": node_type  # IDEF0タイプを追加
        })
    
    # 5. エッジデータ生成（隣接行列から）
    opm_edges = _generate_edges_from_matrix(
        nodes,
        adjacency_matrix,
        node_to_category,
        node_positions,
        categories,
        layers
    )
    
    # 6. プレーンデータ計算（3D空間サイズ）
    plane_data = _calculate_plane_data(len(categories), scale)
    
    return {
        "projectName": "PIM to OPM",
        "projectNumber": 1,
        "version": "1.4.0",
        "colorList": DEFAULT_LAYER_COLORS[:len(categories)],
        "layers": layers,
        "nodes": opm_nodes,
        "edges": opm_edges,
        "planeData": plane_data,
        "nodePositions": [
            {
                "key": node["key"],
                "position": node["position"],
                "name": node["node"]["name"],
                "layer": node["node"]["layer"]
            }
            for node in opm_nodes
        ],
        "edgePositions": [
            {
                "key": edge["key"],
                "position": edge["position"],
                "fromname": edge["edge"]["fromname"],
                "fromlayer": edge["edge"]["fromlayer"],
                "toname": edge["edge"]["toname"],
                "tolayer": edge["edge"]["tolayer"],
                "type": edge["type"]
            }
            for edge in opm_edges
        ]
    }


def _build_node_category_mapping(
    nodes: List[str],
    categories: List[str],
    idef0_data: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    """ノード→カテゴリのマッピングを構築"""
    mapping = {}
    
    for category in categories:
        if category in idef0_data:
            category_data = idef0_data[category]
            
            # Outputs
            if "outputs" in category_data:
                for output in category_data["outputs"]:
                    if output in nodes:
                        mapping[output] = category
            
            # Mechanisms
            if "mechanisms" in category_data:
                for mechanism in category_data["mechanisms"]:
                    if mechanism in nodes:
                        mapping[mechanism] = category
            
            # Inputs
            if "inputs" in category_data:
                for input_item in category_data["inputs"]:
                    if input_item in nodes:
                        mapping[input_item] = category
    
    return mapping


def _build_node_type_mapping(
    nodes: List[str],
    idef0_data: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    """ノード→タイプ（Output/Mechanism/Input）のマッピングを構築"""
    mapping = {}
    
    for category, category_data in idef0_data.items():
        # Outputs
        if "outputs" in category_data:
            for output in category_data["outputs"]:
                if output in nodes:
                    mapping[output] = "Output"
        
        # Mechanisms
        if "mechanisms" in category_data:
            for mechanism in category_data["mechanisms"]:
                if mechanism in nodes:
                    mapping[mechanism] = "Mechanism"
        
        # Inputs
        if "inputs" in category_data:
            for input_item in category_data["inputs"]:
                if input_item in nodes:
                    mapping[input_item] = "Input"
    
    return mapping


def _calculate_3d_layout(
    nodes: List[str],
    categories: List[str],
    node_to_category: Dict[str, str],
    adjacency_matrix: np.ndarray,
    scale: float = 10.0
) -> Dict[str, Dict[str, float]]:
    """
    3D座標を自動計算
    
    レイアウト方針:
    - Z軸: カテゴリインデックス（レイヤー）
    - XY平面: 簡易的な力学モデル（Spring Force Layout）
    """
    positions = {}
    
    # カテゴリ→Z座標
    category_to_z = {cat: i for i, cat in enumerate(categories)}
    
    # カテゴリごとにノードを分類
    nodes_by_category = {cat: [] for cat in categories}
    for node in nodes:
        category = node_to_category.get(node, categories[0] if categories else "Unknown")
        if category in nodes_by_category:
            nodes_by_category[category].append(node)
    
    # 各カテゴリ内でXY座標を計算
    for category, category_nodes in nodes_by_category.items():
        z = category_to_z.get(category, 0)
        
        # 簡易的な円形配置（ノード数が多い場合はグリッド配置）
        n = len(category_nodes)
        if n == 0:
            continue
        
        if n <= 12:
            # 円形配置
            radius = min(scale / 2, max(2.0, n / 2))
            for i, node in enumerate(category_nodes):
                angle = 2 * math.pi * i / n
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                positions[node] = {
                    "x": float(round(x, 2)), 
                    "y": float(round(y, 2)), 
                    "z": int(z)
                }
        else:
            # グリッド配置
            grid_size = math.ceil(math.sqrt(n))
            spacing = scale / (grid_size + 1)
            half_scale = scale / 2
            
            for i, node in enumerate(category_nodes):
                row = i // grid_size
                col = i % grid_size
                x = -half_scale + spacing * (col + 1)
                y = -half_scale + spacing * (row + 1)
                positions[node] = {
                    "x": float(round(x, 2)), 
                    "y": float(round(y, 2)), 
                    "z": int(z)
                }
    
    return positions


def _generate_edges_from_matrix(
    nodes: List[str],
    adjacency_matrix: np.ndarray,
    node_to_category: Dict[str, str],
    node_positions: Dict[str, Dict[str, float]],
    categories: List[str],
    layers: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """隣接行列からエッジデータを生成"""
    edges = []
    
    for i, from_node in enumerate(nodes):
        for j, to_node in enumerate(nodes):
            score = _to_python_type(adjacency_matrix[i, j])
            
            # スコアが0ならスキップ
            if score == 0:
                continue
            
            from_category = node_to_category.get(from_node, categories[0] if categories else "Unknown")
            to_category = node_to_category.get(to_node, categories[0] if categories else "Unknown")
            
            from_pos = node_positions.get(from_node, {"x": 0, "y": 0, "z": 0})
            to_pos = node_positions.get(to_node, {"x": 0, "y": 0, "z": 0})
            
            # エッジタイプを決定（スコアに応じて）
            edge_type = _determine_edge_type(score)
            
            # エッジキー
            edge_key = f"{from_node} ( {from_category} ) --({edge_type})-> {to_node} ( {to_category} )"
            
            # 両レイヤーが表示中か確認
            from_layer_visible = any(l["layer"] == from_category and l.get("isvisible", True) for l in layers)
            to_layer_visible = any(l["layer"] == to_category and l.get("isvisible", True) for l in layers)
            is_visible = from_layer_visible and to_layer_visible
            
            edges.append({
                "key": edge_key,
                "type": edge_type,
                "edge": {
                    "fromkey": f"{from_node} ( {from_category} )",
                    "fromname": from_node,
                    "fromlayer": from_category,
                    "tokey": f"{to_node} ( {to_category} )",
                    "toname": to_node,
                    "tolayer": to_category
                },
                "position": {
                    "from": from_pos,
                    "to": to_pos
                },
                "info": {
                    "direction": "forward",
                    "weight": f"+{int(score)}" if score > 0 else str(int(score)),
                    "score": float(score) if isinstance(score, float) else int(score)
                },
                "isvisible": is_visible
            })
    
    return edges


def _determine_edge_type(score: float) -> str:
    """スコアに応じてエッジタイプを決定"""
    if abs(score) >= 7:
        return "affects"  # 強い影響
    elif abs(score) >= 4:
        return "affects"  # 中程度の影響
    else:
        return "affects"  # 弱い影響（すべて"affects"タイプ）


def _calculate_plane_data(num_layers: int, scale: float = 10.0) -> Dict[str, int]:
    """3D空間のプレーンデータを計算"""
    # m: プレーンのサイズ（XY平面）
    # d: レイヤー間の距離（Z軸）
    m = int(scale)
    d = max(3, num_layers // 2)  # レイヤー数に応じて調整
    
    return {"m": m, "d": d}


def get_node_color_by_type(node_type: str) -> str:
    """ノードタイプから色を取得"""
    return NODE_TYPE_COLORS.get(node_type, "#70e483")
