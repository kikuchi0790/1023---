"""
Data models for Process Insight Modeler.
Pydanticを使用したデータモデル定義
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class FunctionalCategory(BaseModel):
    """機能カテゴリのデータモデル"""

    name: str = Field(..., description="カテゴリ名", min_length=1, max_length=100)
    description: str = Field(
        "", description="カテゴリの説明", max_length=500
    )
    importance: int = Field(
        3, description="重要度（1-5）", ge=1, le=5
    )
    perspective: str = Field(
        "balanced", description="観点（quality/cost/time/safety/balanced）"
    )
    examples: List[str] = Field(
        default_factory=list, description="具体例のリスト"
    )

    class Config:
        """Pydantic設定"""
        json_schema_extra = {
            "example": {
                "name": "品質管理",
                "description": "製品の品質を保証するための検査・測定活動",
                "importance": 5,
                "perspective": "quality",
                "examples": ["寸法検査", "外観検査", "性能試験"]
            }
        }


class CategoryGenerationOptions(BaseModel):
    """カテゴリ生成オプション"""

    focus: str = Field(
        "balanced",
        description="分析の焦点（balanced/quality/cost/time/safety/flexibility）"
    )
    granularity: str = Field(
        "standard",
        description="粒度（coarse/standard/detailed）"
    )
    count_min: int = Field(5, description="最小カテゴリ数", ge=3, le=20)
    count_max: int = Field(8, description="最大カテゴリ数", ge=3, le=20)
    include_description: bool = Field(
        True, description="説明文を含めるか"
    )

    def get_count_range(self) -> tuple[int, int]:
        """粒度に基づいたカテゴリ数の範囲を取得"""
        if self.granularity == "coarse":
            return (4, 5)
        elif self.granularity == "detailed":
            return (10, 12)
        else:  # standard
            return (6, 8)

    def get_focus_description(self) -> str:
        """焦点の説明を取得"""
        focus_descriptions = {
            "balanced": "バランス型：すべての観点を均等に考慮",
            "quality": "品質重視：製品・サービスの品質向上を最優先",
            "cost": "コスト重視：製造コストと効率性を最優先",
            "time": "時間重視：リードタイムと生産速度を最優先",
            "safety": "安全性重視：作業者の安全と環境配慮を最優先",
            "flexibility": "柔軟性重視：変化への対応力と適応性を最優先"
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


class Node(BaseModel):
    """ノードのデータモデル"""

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
