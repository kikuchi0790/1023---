"""
Evaluation pair filtering based on logical rules.
論理ルールに基づく評価ペアのフィルタリング
"""

from typing import List, Dict, Any, Set, Tuple


def get_category_index(category: str, categories: List[str]) -> int:
    """
    カテゴリのインデックスを取得
    
    Args:
        category: カテゴリ名
        categories: カテゴリリスト（時系列順序）
    
    Returns:
        インデックス（0-based）、見つからない場合は-1
    """
    try:
        return categories.index(category)
    except ValueError:
        return -1


def calculate_category_distance(
    cat1: str,
    cat2: str,
    categories: List[str]
) -> int:
    """
    2つのカテゴリ間の距離を計算
    
    Args:
        cat1: カテゴリ1
        cat2: カテゴリ2
        categories: カテゴリリスト（時系列順序）
    
    Returns:
        距離（0=同一、1=隣接、2+=離れている）
    """
    idx1 = get_category_index(cat1, categories)
    idx2 = get_category_index(cat2, categories)
    
    if idx1 == -1 or idx2 == -1:
        return 999  # 不明なカテゴリは最大距離
    
    return abs(idx1 - idx2)


def should_evaluate_by_phase_logic(
    pair: Dict[str, Any],
    distance: int
) -> Tuple[bool, str]:
    """
    フェーズ固有のロジックに基づき評価の必要性を判定
    
    Args:
        pair: 評価ペア
        distance: カテゴリ間距離
    
    Returns:
        (評価すべきか, 理由)
    """
    phase = pair.get("evaluation_phase", "")
    from_type = pair.get("from_type", "")
    to_type = pair.get("to_type", "")
    
    # フェーズ1: 性能→特性 (Output → Mechanism/Input)
    if phase == "perf_to_char":
        if distance == 0:
            return True, "same_category_strong_dependency"
        elif distance == 1:
            # 前カテゴリのOutputが次カテゴリのMechanism/Inputに影響する可能性
            return True, "adjacent_category_possible_influence"
        else:
            # 距離2+: 影響はほぼない
            return False, "distant_category_no_influence"
    
    # フェーズ2: 特性→性能 (Mechanism/Input → Output)
    elif phase == "char_to_perf":
        if distance == 0:
            return True, "same_category_strong_dependency"
        elif distance == 1:
            # 前工程の手段/材料が次工程の性能に影響
            return True, "adjacent_category_process_flow"
        else:
            return False, "distant_category_no_influence"
    
    # フェーズ3: 性能間 (Output ↔ Output)
    elif phase == "perf_to_perf":
        if distance == 0:
            # 同一カテゴリ内のOutput間: トレードオフの可能性
            return True, "same_category_tradeoff"
        elif distance == 1:
            # 隣接カテゴリ: 前工程の性能が次工程の性能に影響
            return True, "adjacent_category_cascading_effect"
        else:
            # 距離2+: トレードオフはほぼない
            return False, "distant_category_no_tradeoff"
    
    # フェーズ4: 特性間 (Mechanism/Input間)
    elif phase == "char_to_char":
        if distance == 0:
            # 同一カテゴリ内: リソース競合、制約共有の可能性
            return True, "same_category_resource_conflict"
        else:
            # 異カテゴリの特性間: ほぼ無関係
            return False, "different_category_no_interaction"
    
    # デフォルト: 距離0のみ評価
    return distance == 0, "default_rule"


def filter_pairs_by_logic(
    pairs: List[Dict[str, Any]],
    idef0_nodes: Dict[str, Dict[str, Any]],
    categories: List[str]
) -> Dict[str, Any]:
    """
    論理ルールに基づき評価ペアを分類
    
    Args:
        pairs: 全評価ペアリスト
        idef0_nodes: IDEF0ノードデータ
        categories: カテゴリリスト（時系列順序）
    
    Returns:
        {
            "must_evaluate": List[Dict],      # 必ず評価すべきペア
            "should_evaluate": List[Dict],    # 評価推奨ペア
            "default_zero": List[Dict],       # デフォルト0とするペア
            "category_batches": Dict[str, List[Dict]],  # カテゴリごとのバッチ
            "statistics": Dict[str, int]      # 統計情報
        }
    """
    must_evaluate = []
    should_evaluate = []
    default_zero = []
    category_batches: Dict[str, List[Dict]] = {}
    
    # カテゴリごとのバッチを初期化
    for cat in categories:
        category_batches[cat] = []
    
    for pair in pairs:
        from_cat = pair.get("from_category", "")
        to_cat = pair.get("to_category", "")
        
        distance = calculate_category_distance(from_cat, to_cat, categories)
        should_eval, reason = should_evaluate_by_phase_logic(pair, distance)
        
        # 評価理由を追加
        pair_with_reason = pair.copy()
        pair_with_reason["filter_reason"] = reason
        pair_with_reason["category_distance"] = distance
        
        if should_eval:
            if distance == 0:
                # 同一カテゴリ: 必ず評価
                must_evaluate.append(pair_with_reason)
                category_batches[from_cat].append(pair_with_reason)
            else:
                # 隣接カテゴリ: 評価推奨
                should_evaluate.append(pair_with_reason)
        else:
            # デフォルト0
            default_zero.append(pair_with_reason)
    
    statistics = {
        "total_pairs": len(pairs),
        "must_evaluate": len(must_evaluate),
        "should_evaluate": len(should_evaluate),
        "default_zero": len(default_zero),
        "reduction_rate": 100 * (1 - (len(must_evaluate) + len(should_evaluate)) / len(pairs))
    }
    
    return {
        "must_evaluate": must_evaluate,
        "should_evaluate": should_evaluate,
        "default_zero": default_zero,
        "category_batches": category_batches,
        "statistics": statistics
    }


def get_batch_summary(
    category_batches: Dict[str, List[Dict]]
) -> List[Dict[str, Any]]:
    """
    カテゴリバッチのサマリーを取得
    
    Args:
        category_batches: カテゴリごとのペアリスト
    
    Returns:
        サマリーリスト
        [
            {
                "category": str,
                "pair_count": int,
                "phase_counts": {phase: count}
            },
            ...
        ]
    """
    summaries = []
    
    for category, pairs in category_batches.items():
        if not pairs:
            continue
        
        phase_counts: Dict[str, int] = {}
        for pair in pairs:
            phase = pair.get("evaluation_phase", "unknown")
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        summaries.append({
            "category": category,
            "pair_count": len(pairs),
            "phase_counts": phase_counts
        })
    
    return summaries


def apply_default_scores(
    default_zero_pairs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    デフォルト0のペアに対してスコア0を適用
    
    Args:
        default_zero_pairs: デフォルト0とするペアリスト
    
    Returns:
        スコアと理由を含むペアリスト
        [
            {
                "from_node": str,
                "to_node": str,
                "score": 0,
                "reason": str,
                "auto_assigned": True
            },
            ...
        ]
    """
    results = []
    
    reason_templates = {
        "distant_category_no_influence": "工程間距離が遠く、直接的な影響はほぼありません。",
        "distant_category_no_tradeoff": "工程間距離が遠く、性能トレードオフは発生しません。",
        "different_category_no_interaction": "異なるカテゴリの特性間には相互作用がありません。",
        "default_rule": "論理ルールに基づき、影響は無視できる程度です。"
    }
    
    for pair in default_zero_pairs:
        reason = pair.get("filter_reason", "default_rule")
        reason_text = reason_templates.get(reason, "影響なしと判定されました。")
        
        results.append({
            "from_node": pair["from_node"],
            "to_node": pair["to_node"],
            "score": 0,
            "reason": f"【自動評価: スコア0】\n{reason_text}",
            "auto_assigned": True,
            "filter_reason": reason,
            "category_distance": pair.get("category_distance", 999)
        })
    
    return results
