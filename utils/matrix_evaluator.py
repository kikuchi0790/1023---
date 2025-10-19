"""
Matrix-based evaluation planning and knowledge extraction.
行列ベース評価の計画とナレッジ抽出
"""

from typing import List, Dict, Any, Tuple, Optional
from utils.evaluation_filter import get_category_index, calculate_category_distance


class EvaluationPhase:
    SAME_CATEGORY = "same_category"
    ADJACENT_CATEGORY = "adjacent_category"
    DISTANT_CATEGORY = "distant_category"


class MatrixEvaluator:
    """行列ベース評価の計画とナレッジ管理"""
    
    def __init__(
        self,
        categories: List[str],
        idef0_nodes: Dict[str, Dict[str, Any]],
        all_nodes: List[str]
    ):
        """
        Args:
            categories: 機能カテゴリリスト（時系列順）
            idef0_nodes: IDEF0ノードデータ
            all_nodes: 全ノードリスト
        """
        self.categories = categories
        self.idef0_nodes = idef0_nodes
        self.all_nodes = all_nodes
        self.knowledge_base: Dict[str, int] = {}
    
    def get_category_nodes(self, category: str) -> List[str]:
        """
        指定カテゴリに属する全ノードを取得
        
        Args:
            category: カテゴリ名
        
        Returns:
            ノードリスト（outputs + mechanisms + inputs）
        """
        idef0_data = self.idef0_nodes.get(category, {})
        nodes = []
        
        for key in ["outputs", "mechanisms", "inputs"]:
            nodes.extend(idef0_data.get(key, []))
        
        return nodes
    
    def plan_evaluation_phases(
        self, 
        max_distance: int = 1,
        enable_distant: bool = False
    ) -> List[Dict[str, Any]]:
        """
        3フェーズ評価計画を作成
        
        Args:
            max_distance: 最大カテゴリ間距離（デフォルト1: 隣接まで）
            enable_distant: 遠距離評価を有効化（距離2+）
        
        Returns:
            評価計画リスト
            [
                {
                    "phase": "same_category" | "adjacent_category" | "distant_category",
                    "phase_index": 1-3,
                    "from_category": str,
                    "to_category": str,
                    "from_nodes": List[str],
                    "to_nodes": List[str],
                    "distance": int,
                    "matrix_size": (n, m),
                    "requires_knowledge": bool,
                    "knowledge_sources": List[str]
                },
                ...
            ]
        """
        plans = []
        
        for i, cat_from in enumerate(self.categories):
            from_nodes = self.get_category_nodes(cat_from)
            if not from_nodes:
                continue
            
            for j, cat_to in enumerate(self.categories):
                to_nodes = self.get_category_nodes(cat_to)
                if not to_nodes:
                    continue
                
                distance = abs(i - j)
                
                if distance == 0:
                    phase = EvaluationPhase.SAME_CATEGORY
                    phase_index = 1
                    requires_knowledge = False
                    knowledge_sources = []
                elif distance == 1:
                    if max_distance < 1:
                        continue
                    phase = EvaluationPhase.ADJACENT_CATEGORY
                    phase_index = 2
                    requires_knowledge = True
                    knowledge_sources = [cat_from, cat_to]
                else:
                    if not enable_distant:
                        continue
                    if max_distance < distance:
                        continue
                    phase = EvaluationPhase.DISTANT_CATEGORY
                    phase_index = 3
                    requires_knowledge = True
                    knowledge_sources = self._get_intermediate_categories(cat_from, cat_to)
                
                plans.append({
                    "phase": phase,
                    "phase_index": phase_index,
                    "from_category": cat_from,
                    "to_category": cat_to,
                    "from_nodes": from_nodes,
                    "to_nodes": to_nodes,
                    "distance": distance,
                    "matrix_size": (len(from_nodes), len(to_nodes)),
                    "requires_knowledge": requires_knowledge,
                    "knowledge_sources": knowledge_sources
                })
        
        plans.sort(key=lambda x: (x["phase_index"], x["from_category"], x["to_category"]))
        
        return plans
    
    def _get_intermediate_categories(self, cat_from: str, cat_to: str) -> List[str]:
        """
        2つのカテゴリ間の中間カテゴリを取得
        
        Args:
            cat_from: 開始カテゴリ
            cat_to: 終了カテゴリ
        
        Returns:
            中間カテゴリリスト（開始・終了含む）
        """
        idx_from = get_category_index(cat_from, self.categories)
        idx_to = get_category_index(cat_to, self.categories)
        
        if idx_from == -1 or idx_to == -1:
            return [cat_from, cat_to]
        
        start = min(idx_from, idx_to)
        end = max(idx_from, idx_to) + 1
        
        return self.categories[start:end]
    
    def add_evaluation_result(self, from_node: str, to_node: str, score: int):
        """
        評価結果をナレッジベースに追加
        
        Args:
            from_node: 評価元ノード
            to_node: 評価先ノード
            score: 評価スコア
        """
        if score != 0:
            key = f"{from_node}→{to_node}"
            self.knowledge_base[key] = score
    
    def extract_knowledge_for_plan(
        self,
        plan: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        評価計画に関連するナレッジを抽出
        
        Args:
            plan: 評価計画
            top_k: 抽出する最大件数
        
        Returns:
            関連ナレッジリスト
            [
                {
                    "from_node": str,
                    "to_node": str,
                    "score": int,
                    "source_category": str
                },
                ...
            ]
        """
        if not plan["requires_knowledge"]:
            return []
        
        knowledge_sources = plan["knowledge_sources"]
        relevant_knowledge = []
        
        for key, score in self.knowledge_base.items():
            from_node, to_node = key.split("→")
            
            from_cat = self._get_node_category(from_node)
            to_cat = self._get_node_category(to_node)
            
            if from_cat in knowledge_sources or to_cat in knowledge_sources:
                relevant_knowledge.append({
                    "from_node": from_node,
                    "to_node": to_node,
                    "score": score,
                    "source_category": from_cat if from_cat in knowledge_sources else to_cat,
                    "abs_score": abs(score)
                })
        
        relevant_knowledge.sort(key=lambda x: x["abs_score"], reverse=True)
        
        return relevant_knowledge[:top_k]
    
    def _get_node_category(self, node_name: str) -> Optional[str]:
        """
        ノードが属するカテゴリを取得
        
        Args:
            node_name: ノード名
        
        Returns:
            カテゴリ名、見つからない場合はNone
        """
        for category, idef0_data in self.idef0_nodes.items():
            for key in ["outputs", "mechanisms", "inputs"]:
                if node_name in idef0_data.get(key, []):
                    return category
        return None
    
    def get_phase_summary(self, plans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        評価計画の統計サマリーを取得
        
        Args:
            plans: 評価計画リスト
        
        Returns:
            統計情報
        """
        phase_stats = {
            EvaluationPhase.SAME_CATEGORY: {"count": 0, "total_pairs": 0},
            EvaluationPhase.ADJACENT_CATEGORY: {"count": 0, "total_pairs": 0},
            EvaluationPhase.DISTANT_CATEGORY: {"count": 0, "total_pairs": 0}
        }
        
        for plan in plans:
            phase = plan["phase"]
            n, m = plan["matrix_size"]
            
            if plan["distance"] == 0:
                pairs = n * (n - 1)
            else:
                pairs = n * m
            
            phase_stats[phase]["count"] += 1
            phase_stats[phase]["total_pairs"] += pairs
        
        return {
            "total_plans": len(plans),
            "total_pairs": sum(s["total_pairs"] for s in phase_stats.values()),
            "phase_1_same": phase_stats[EvaluationPhase.SAME_CATEGORY],
            "phase_2_adjacent": phase_stats[EvaluationPhase.ADJACENT_CATEGORY],
            "phase_3_distant": phase_stats[EvaluationPhase.DISTANT_CATEGORY]
        }
    
    def format_knowledge_for_prompt(
        self,
        knowledge: List[Dict[str, Any]],
        max_items: int = 10
    ) -> str:
        """
        ナレッジをLLMプロンプト用にフォーマット
        
        Args:
            knowledge: ナレッジリスト
            max_items: 最大表示件数
        
        Returns:
            フォーマット済み文字列
        """
        if not knowledge:
            return "参考評価なし（初回評価）"
        
        lines = []
        for i, item in enumerate(knowledge[:max_items], 1):
            sign = "+" if item["score"] > 0 else ""
            lines.append(
                f"{i}. {item['from_node']} → {item['to_node']}: {sign}{item['score']} "
                f"（{item['source_category']}内）"
            )
        
        if len(knowledge) > max_items:
            lines.append(f"...他{len(knowledge) - max_items}件")
        
        return "\n".join(lines)
