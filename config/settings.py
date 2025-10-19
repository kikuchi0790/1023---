"""
Settings and configuration for Process Insight Modeler.
設定と定数の管理
"""

from typing import Final


class Settings:
    """アプリケーション設定クラス"""

    APP_TITLE: Final[str] = "Process Insight Modeler (PIM)"
    APP_VERSION: Final[str] = "0.1.0"

    DEFAULT_PROCESS_NAME: Final[str] = "たまごやきの調理"
    DEFAULT_PROCESS_DESCRIPTION: Final[str] = """キッチンで人間がたまごやきを作る調理プロセスです。

主要工程：
1. 材料の準備（卵、調味料、油）
2. 卵を溶く（ボウルと菜箸で混ぜる）
3. 調味料の投入（砂糖、塩、だし汁）
4. フライパンの加熱と油引き
5. 卵液を流し込み焼き成形（巻く/折りたたむ）
6. 盛り付けと完成

使用器具：ボウル、菜箸、フライパン、フライ返し、まな板、包丁"""

    MIN_CATEGORIES: Final[int] = 5
    MAX_CATEGORIES: Final[int] = 8

    EVALUATION_SCALE_MIN: Final[int] = -9
    EVALUATION_SCALE_MAX: Final[int] = 9
    EVALUATION_SCALE_VALUES: Final[tuple] = (-9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9)

    OPENAI_MODEL: Final[str] = "gpt-5-2025-08-07"
    # OPENAI_TEMPERATURE: GPT-5 does not support custom temperature values (only default 1.0)
    OPENAI_MAX_RETRIES: Final[int] = 3
    OPENAI_TIMEOUT: Final[int] = 60

    HEATMAP_COLORMAP: Final[str] = "coolwarm"
    HEATMAP_LINEWIDTH: Final[float] = 0.5

    NETWORK_LAYOUT: Final[str] = "spring"
    NETWORK_NODE_SIZE: Final[int] = 3000
    NETWORK_FONT_SIZE: Final[int] = 10


settings = Settings()
