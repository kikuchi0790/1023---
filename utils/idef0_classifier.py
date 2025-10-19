"""
IDEF0 node classification and Zigzagging pair generation.
IDEF0ノード分類とZigzagging評価ペア生成
"""

from typing import List, Dict, Any, Tuple
from itertools import product


class NodeType:
    OUTPUT = "output"
    MECHANISM = "mechanism"
    INPUT = "input"


class EvaluationPhase:
    PERF_TO_CHAR = "perf_to_char"
    CHAR_TO_PERF = "char_to_perf"
    PERF_TO_PERF = "perf_to_perf"
    CHAR_TO_CHAR = "char_to_char"


def classify_node_type(
    node_name: str,
    idef0_nodes: Dict[str, Dict[str, Any]]
) -> Tuple[str, str]:
    """
    ノードのIDEF0タイプとカテゴリを判定
    
    Args:
        node_name: ノード名
        idef0_nodes: カテゴリ名をキーとしたIDEF0ノードデータ
                     形式: {category: {"inputs": [...], "mechanisms": [...], "outputs": [...]}}
    
    Returns:
        (node_type, category): ノードタイプとカテゴリ名のタプル
        node_type: "output" | "mechanism" | "input"
    """
    for category, idef0_data in idef0_nodes.items():
        if node_name in idef0_data.get("outputs", []):
            return NodeType.OUTPUT, category
        
        if node_name in idef0_data.get("mechanisms", []):
            return NodeType.MECHANISM, category
        
        if node_name in idef0_data.get("inputs", []):
            return NodeType.INPUT, category
    
    return NodeType.MECHANISM, "不明"


def get_nodes_by_type(
    nodes: List[str],
    idef0_nodes: Dict[str, Dict[str, Any]]
) -> Dict[str, List[str]]:
    """
    ノードをタイプ別に分類
    
    Args:
        nodes: 全ノードリスト
        idef0_nodes: IDEF0ノードデータ
    
    Returns:
        タイプ別ノードリスト
        {"output": [...], "mechanism": [...], "input": [...]}
    """
    nodes_by_type = {
        NodeType.OUTPUT: [],
        NodeType.MECHANISM: [],
        NodeType.INPUT: []
    }
    
    for node in nodes:
        node_type, _ = classify_node_type(node, idef0_nodes)
        nodes_by_type[node_type].append(node)
    
    return nodes_by_type


def generate_zigzagging_pairs(
    nodes: List[str],
    idef0_nodes: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Zigzagging手法に基づく評価ペアを生成
    
    文献に基づき、論理的依存関係に沿った順序で評価ペアを生成します：
    1. フェーズ1: 性能→特性 (Output → Mechanism/Input)
    2. フェーズ2: 特性→性能 (Mechanism/Input → Output)
    3. フェーズ3: 性能間 (Output ↔ Output)
    4. フェーズ4: 特性間 (Mechanism/Input間)
    
    Args:
        nodes: 全ノードリスト
        idef0_nodes: IDEF0ノードデータ
    
    Returns:
        評価ペアリスト
        [
            {
                "from_node": str,
                "to_node": str,
                "from_type": str,
                "to_type": str,
                "from_category": str,
                "to_category": str,
                "evaluation_phase": str,
                "phase_name": str,
                "phase_description": str
            },
            ...
        ]
    """
    nodes_by_type = get_nodes_by_type(nodes, idef0_nodes)
    
    output_nodes = nodes_by_type[NodeType.OUTPUT]
    mechanism_nodes = nodes_by_type[NodeType.MECHANISM]
    input_nodes = nodes_by_type[NodeType.INPUT]
    
    characteristic_nodes = mechanism_nodes + input_nodes
    
    pairs = []
    
    for from_node in output_nodes:
        for to_node in characteristic_nodes:
            from_type, from_cat = classify_node_type(from_node, idef0_nodes)
            to_type, to_cat = classify_node_type(to_node, idef0_nodes)
            
            pairs.append({
                "from_node": from_node,
                "to_node": to_node,
                "from_type": from_type,
                "to_type": to_type,
                "from_category": from_cat,
                "to_category": to_cat,
                "evaluation_phase": EvaluationPhase.PERF_TO_CHAR,
                "phase_name": "フェーズ1: 性能→特性",
                "phase_description": "この性能を達成するために、この手段/材料はどれほど重要か？"
            })
    
    for from_node in characteristic_nodes:
        for to_node in output_nodes:
            from_type, from_cat = classify_node_type(from_node, idef0_nodes)
            to_type, to_cat = classify_node_type(to_node, idef0_nodes)
            
            pairs.append({
                "from_node": from_node,
                "to_node": to_node,
                "from_type": from_type,
                "to_type": to_type,
                "from_category": from_cat,
                "to_category": to_cat,
                "evaluation_phase": EvaluationPhase.CHAR_TO_PERF,
                "phase_name": "フェーズ2: 特性→性能",
                "phase_description": "この手段/材料を改善すると、この性能はどれほど向上するか？"
            })
    
    for from_node in output_nodes:
        for to_node in output_nodes:
            if from_node == to_node:
                continue
            
            from_type, from_cat = classify_node_type(from_node, idef0_nodes)
            to_type, to_cat = classify_node_type(to_node, idef0_nodes)
            
            pairs.append({
                "from_node": from_node,
                "to_node": to_node,
                "from_type": from_type,
                "to_type": to_type,
                "from_category": from_cat,
                "to_category": to_cat,
                "evaluation_phase": EvaluationPhase.PERF_TO_PERF,
                "phase_name": "フェーズ3: 性能間",
                "phase_description": "ある性能を向上させると、他の性能との間にトレードオフが生じるか？"
            })
    
    for from_node in characteristic_nodes:
        for to_node in characteristic_nodes:
            if from_node == to_node:
                continue
            
            from_type, from_cat = classify_node_type(from_node, idef0_nodes)
            to_type, to_cat = classify_node_type(to_node, idef0_nodes)
            
            pairs.append({
                "from_node": from_node,
                "to_node": to_node,
                "from_type": from_type,
                "to_type": to_type,
                "from_category": from_cat,
                "to_category": to_cat,
                "evaluation_phase": EvaluationPhase.CHAR_TO_CHAR,
                "phase_name": "フェーズ4: 特性間",
                "phase_description": "手段・材料間の相互影響はどうか？"
            })
    
    return pairs


def get_phase_statistics(pairs: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    フェーズごとのペア数を集計
    
    Args:
        pairs: 評価ペアリスト
    
    Returns:
        フェーズごとのペア数
    """
    stats = {
        EvaluationPhase.PERF_TO_CHAR: 0,
        EvaluationPhase.CHAR_TO_PERF: 0,
        EvaluationPhase.PERF_TO_PERF: 0,
        EvaluationPhase.CHAR_TO_CHAR: 0
    }
    
    for pair in pairs:
        phase = pair.get("evaluation_phase")
        if phase in stats:
            stats[phase] += 1
    
    return stats


def get_current_phase_info(
    pairs: List[Dict[str, Any]],
    current_index: int
) -> Dict[str, Any]:
    """
    現在の評価ペアのフェーズ情報を取得
    
    Args:
        pairs: 評価ペアリスト
        current_index: 現在のインデックス
    
    Returns:
        フェーズ情報
        {
            "phase": str,
            "phase_name": str,
            "phase_description": str,
            "phase_start_index": int,
            "phase_end_index": int,
            "phase_progress": int,
            "phase_total": int
        }
    """
    if not pairs or current_index >= len(pairs):
        return {}
    
    current_pair = pairs[current_index]
    current_phase = current_pair.get("evaluation_phase")
    
    phase_start = None
    phase_end = None
    
    for i, pair in enumerate(pairs):
        if pair.get("evaluation_phase") == current_phase:
            if phase_start is None:
                phase_start = i
            phase_end = i
    
    phase_progress = current_index - phase_start + 1 if phase_start is not None else 0
    phase_total = phase_end - phase_start + 1 if phase_start is not None and phase_end is not None else 0
    
    return {
        "phase": current_phase,
        "phase_name": current_pair.get("phase_name", ""),
        "phase_description": current_pair.get("phase_description", ""),
        "phase_start_index": phase_start,
        "phase_end_index": phase_end,
        "phase_progress": phase_progress,
        "phase_total": phase_total
    }


def get_node_type_label(node_type: str) -> str:
    """
    ノードタイプの日本語ラベルを取得
    
    Args:
        node_type: ノードタイプ
    
    Returns:
        日本語ラベル
    """
    labels = {
        NodeType.OUTPUT: "性能 (Output)",
        NodeType.MECHANISM: "手段 (Mechanism)",
        NodeType.INPUT: "材料 (Input)"
    }
    return labels.get(node_type, "不明")
