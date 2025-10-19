"""
Data models for Process Insight Modeler.
Pydanticを使用したデータモデル定義
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class FunctionalCategory(BaseModel):
    """
    機能カテゴリのデータモデル
    
    機能カテゴリ = プロセスを構成する動的な変換機能
    製品を作り上げるための動的プロセス（インプット→変換→アウトプット）
    """

    name: str = Field(..., description="機能名（動的プロセス）", min_length=1, max_length=100)
    description: str = Field(
        "", description="この機能が何を変換・処理するかの説明", max_length=500
    )
    transformation_type: str = Field(
        "processing",
        description="変換タイプ（preparation/processing/assembly/inspection/adjustment/packaging/transfer）"
    )
    inputs: List[str] = Field(
        default_factory=list, description="インプット（材料、部品、情報）"
    )
    outputs: List[str] = Field(
        default_factory=list, description="アウトプット（加工品、製品、データ）"
    )
    process_phase: str = Field(
        "main_process",
        description="プロセスフェーズ（preparation/main_process/verification/completion）"
    )
    importance: int = Field(
        3, description="重要度（1-5）", ge=1, le=5
    )
    examples: List[str] = Field(
        default_factory=list, description="具体的な作業工程の例"
    )

    class Config:
        """Pydantic設定"""
        json_schema_extra = {
            "example": {
                "name": "部品の組立",
                "description": "シリンダーブロックにピストンとクランクシャフトを組み付ける",
                "transformation_type": "assembly",
                "inputs": ["シリンダーブロック", "ピストン", "クランクシャフト"],
                "outputs": ["組み立て済みエンジンブロック"],
                "process_phase": "main_process",
                "importance": 5,
                "examples": ["ピストン挿入", "クランクシャフト取り付け", "締付トルク管理"]
            }
        }


class CategoryGenerationOptions(BaseModel):
    """カテゴリ生成オプション"""

    focus: str = Field(
        "balanced",
        description="分析の視点（balanced/material_flow/information_flow/quality_gates）"
    )
    granularity: str = Field(
        "standard",
        description="プロセスの分解レベル（high_level/standard/detailed）"
    )
    count_min: int = Field(5, description="最小カテゴリ数", ge=3, le=20)
    count_max: int = Field(8, description="最大カテゴリ数", ge=3, le=20)
    include_description: bool = Field(
        True, description="説明文を含めるか"
    )

    def get_count_range(self) -> tuple[int, int]:
        """粒度に基づいたカテゴリ数の範囲を取得"""
        if self.granularity == "high_level":
            return (4, 5)
        elif self.granularity == "detailed":
            return (10, 12)
        else:  # standard
            return (6, 8)

    def get_focus_description(self) -> str:
        """焦点の説明を取得"""
        focus_descriptions = {
            "balanced": "バランス型：モノ・情報・品質の流れを総合的に分析",
            "material_flow": "モノの流れ重視：物理的な材料・部品の変換プロセスに着目",
            "information_flow": "情報の流れ重視：データ・指示・フィードバックの流れに着目",
            "quality_gates": "品質ゲート重視：品質確認・検査のタイミングに着目"
        }
        return focus_descriptions.get(self.focus, "")


class CategorySet(BaseModel):
    """カテゴリセット（生成された複数のカテゴリをまとめる）"""

    name: str = Field(..., description="セット名")
    categories: List[FunctionalCategory] = Field(
        default_factory=list, description="カテゴリのリスト"
    )
    options: CategoryGenerationOptions = Field(
        ..., description="生成時のオプション"
    )
    generated_at: Optional[str] = Field(None, description="生成日時")

    def to_simple_list(self) -> List[str]:
        """シンプルな文字列リストに変換"""
        return [cat.name for cat in self.categories]

    def get_category_count(self) -> int:
        """カテゴリ数を取得"""
        return len(self.categories)


class IDEF0Node(BaseModel):
    """IDEF0形式のノード（Input-Mechanism-Output構造）"""

    category: str = Field(..., description="所属する機能カテゴリ名")
    inputs: List[str] = Field(
        default_factory=list,
        description="インプット（材料、部品、情報、半製品）"
    )
    mechanisms: List[str] = Field(
        default_factory=list,
        description="メカニズム（設備、道具、作業工程、スキル、手順）"
    )
    outputs: List[str] = Field(
        default_factory=list,
        description="アウトプット（加工品、組立品、データ、完成品）"
    )

    class Config:
        """Pydantic設定"""
        json_schema_extra = {
            "example": {
                "category": "材料を準備する",
                "inputs": ["材料発注情報", "部品リスト", "未検査部品"],
                "mechanisms": ["受入検査", "ノギス", "マイクロメーター", "寸法測定作業"],
                "outputs": ["検査済み材料", "不良品リスト", "検査記録"]
            }
        }


class Node(BaseModel):
    """ノードのデータモデル（後方互換性用）"""

    name: str = Field(..., description="ノード名", min_length=1, max_length=200)
    description: str = Field("", description="ノードの説明", max_length=500)
    node_type: str = Field(
        "process",
        description="ノードタイプ（process/resource/material/skill）"
    )
    related_categories: List[str] = Field(
        default_factory=list, description="関連カテゴリ"
    )


class Evaluation(BaseModel):
    """ノードペア評価のデータモデル"""

    from_node: str = Field(..., description="評価元ノード")
    to_node: str = Field(..., description="評価先ノード")
    score: int = Field(..., description="評価スコア（-9〜+9）", ge=-9, le=9)
    reason: str = Field(..., description="評価理由", min_length=1)

    class Config:
        """Pydantic設定"""
        json_schema_extra = {
            "example": {
                "from_node": "材料検査",
                "to_node": "組立工程",
                "score": 9,
                "reason": "材料検査の品質向上は組立工程の不良率を直接的に削減する"
            }
        }


class ProjectData(BaseModel):
    """プロジェクト全体のデータモデル"""

    process_name: str = Field("", description="プロセス名")
    process_description: str = Field("", description="プロセス説明")
    functional_categories: List[str] = Field(
        default_factory=list, description="機能カテゴリ（後方互換性）"
    )
    categories_with_metadata: List[FunctionalCategory] = Field(
        default_factory=list, description="メタ情報付きカテゴリ"
    )
    nodes: List[str] = Field(default_factory=list, description="ノードリスト")
    evaluations: List[Dict[str, Any]] = Field(
        default_factory=list, description="評価リスト"
    )
    adjacency_matrix: Optional[Any] = Field(None, description="隣接行列")

    def get_categories_simple(self) -> List[str]:
        """シンプルなカテゴリ名のリストを取得"""
        if self.categories_with_metadata:
            return [cat.name for cat in self.categories_with_metadata]
        return self.functional_categories

    def set_categories_from_metadata(
        self, categories: List[FunctionalCategory]
    ) -> None:
        """メタ情報付きカテゴリを設定"""
        self.categories_with_metadata = categories
        self.functional_categories = [cat.name for cat in categories]

    def set_categories_simple(self, categories: List[str]) -> None:
        """シンプルなカテゴリリストを設定"""
        self.functional_categories = categories
        self.categories_with_metadata = [
            FunctionalCategory(name=name) for name in categories
        ]
