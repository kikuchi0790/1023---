"""
Settings and configuration for Process Insight Modeler.
設定と定数の管理
"""

from typing import Final


class Settings:
    """アプリケーション設定クラス"""

    APP_TITLE: Final[str] = "Process Insight Modeler (PIM)"
    APP_VERSION: Final[str] = "0.1.0"

    DEFAULT_PROCESS_NAME: Final[str] = "自動車エンジン組立工程"
    DEFAULT_PROCESS_DESCRIPTION: Final[str] = """エンジンブロックの準備から完成品の検査までを含む組立ラインです。

主要工程：
1. 材料・部品の受入と準備
2. シリンダーブロックへのピストン組み付け
3. クランクシャフト・カムシャフトの取り付け
4. シリンダーヘッドの組み付け
5. 補機類（ポンプ、オルタネーター等）の取り付け
6. 完成品の性能試験・検査
7. 梱包・出荷準備

使用設備：組立ライン、トルクレンチ、検査装置、性能試験機"""

    MIN_CATEGORIES: Final[int] = 5
    MAX_CATEGORIES: Final[int] = 8

    EVALUATION_SCALE_MIN: Final[int] = -9
    EVALUATION_SCALE_MAX: Final[int] = 9
    EVALUATION_SCALE_VALUES: Final[tuple] = (-9, -3, -1, 0, 1, 3, 9)

    OPENAI_MODEL: Final[str] = "gpt-5-2025-08-07"
    OPENAI_TEMPERATURE: Final[float] = 0.7
    OPENAI_MAX_RETRIES: Final[int] = 3
    OPENAI_TIMEOUT: Final[int] = 60

    HEATMAP_COLORMAP: Final[str] = "coolwarm"
    HEATMAP_LINEWIDTH: Final[float] = 0.5

    NETWORK_LAYOUT: Final[str] = "spring"
    NETWORK_NODE_SIZE: Final[int] = 3000
    NETWORK_FONT_SIZE: Final[int] = 10


settings = Settings()
