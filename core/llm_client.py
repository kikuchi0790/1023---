"""
OpenAI API client for Process Insight Modeler.
OpenAI APIクライアント統合
"""

import json
import time
from typing import List, Optional, Dict, Any
import streamlit as st
from openai import OpenAI, OpenAIError
from config.settings import settings
from core.data_models import FunctionalCategory, CategoryGenerationOptions, IDEF0Node


class LLMClient:
    """OpenAI APIクライアントクラス"""

    def __init__(self) -> None:
        """
        初期化
        Streamlit secretsからAPIキーを読み込む
        """
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
            self.client = OpenAI(api_key=api_key)
        except KeyError:
            raise ValueError(
                "OpenAI APIキーが設定されていません。"
                ".streamlit/secrets.toml に OPENAI_API_KEY を設定してください。"
            )

    def _call_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = settings.OPENAI_MAX_RETRIES,
    ) -> str:
        """
        リトライ機能付きAPI呼び出し (GPT-5: temperatureはデフォルト1.0のみサポート)

        Args:
            messages: メッセージリスト
            max_retries: 最大リトライ回数

        Returns:
            APIレスポンスのテキスト

        Raises:
            OpenAIError: API呼び出しエラー
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=messages,
                    timeout=settings.OPENAI_TIMEOUT,
                )
                return response.choices[0].message.content or ""

            except OpenAIError as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                    continue
                raise e

        raise OpenAIError("最大リトライ回数に達しました")

    def extract_functional_categories(self, process_description: str) -> List[str]:
        """
        生産プロセスから機能カテゴリを抽出

        Args:
            process_description: プロセスの説明文

        Returns:
            機能カテゴリのリスト

        Raises:
            ValueError: プロセス説明が空の場合
            json.JSONDecodeError: JSONパースエラー
            OpenAIError: API呼び出しエラー
        """
        if not process_description or not process_description.strip():
            raise ValueError("プロセスの概要を入力してください")

        system_prompt = """あなたは生産技術に20年以上従事するベテランのコンサルタントです。

与えられた生産プロセスの説明文を分析し、そのプロセスを評価するための重要な「機能カテゴリ」を5〜8個抽出してください。

機能カテゴリは以下の観点を含めてください：
- 品質（Quality）
- コスト（Cost）
- 時間（Time）
- 安全性（Safety）
- その他のプロセス特有の重要な観点

結果は必ずJSON形式のリスト（例: ["品質", "作業時間", "製造コスト", "安全性"]）として出力してください。
他の説明文は一切不要です。JSON配列のみを出力してください。"""

        user_prompt = f"""以下の生産プロセスについて、機能カテゴリを抽出してください。

【生産プロセス】
{process_description}

JSON形式の配列で出力してください。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_text = self._call_with_retry(messages)

        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            categories = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"LLMからの応答がJSON形式ではありません: {response_text}",
                e.doc,
                e.pos,
            )

        if not isinstance(categories, list):
            raise ValueError(f"期待されるリスト形式ではありません: {type(categories)}")

        if not all(isinstance(cat, str) for cat in categories):
            raise ValueError("すべてのカテゴリが文字列である必要があります")

        if not (settings.MIN_CATEGORIES <= len(categories) <= settings.MAX_CATEGORIES):
            categories = categories[: settings.MAX_CATEGORIES]

        return categories

    def extract_categories_advanced(
        self,
        process_name: str,
        process_description: str,
        options: CategoryGenerationOptions,
    ) -> List[FunctionalCategory]:
        """
        高度なプロンプトで機能カテゴリを抽出（メタ情報付き）

        Args:
            process_name: プロセス名
            process_description: プロセスの説明文
            options: 生成オプション

        Returns:
            メタ情報付き機能カテゴリのリスト

        Raises:
            ValueError: プロセス説明が空の場合
            json.JSONDecodeError: JSONパースエラー
            OpenAIError: API呼び出しエラー
        """
        if not process_description or not process_description.strip():
            raise ValueError("プロセスの概要を入力してください")

        count_min, count_max = options.get_count_range()
        focus_desc = options.get_focus_description()

        focus_guidelines = {
            "material_flow": """
【モノの流れ重視のガイドライン】
- 物理的な材料・部品がどのように変換されるか
- 各工程での形状・状態の変化
- 部品の結合・分解・加工
- 物理的な移動・搬送""",
            "information_flow": """
【情報の流れ重視のガイドライン】
- 作業指示・データの流れ
- 測定・検査データの生成
- フィードバック・確認のタイミング
- 情報処理・判断のプロセス""",
            "quality_gates": """
【品質ゲート重視のガイドライン】
- 品質確認・検査のタイミング
- 合否判定のポイント
- 不良品の検出・除去
- 品質保証のための確認工程""",
            "balanced": """
【バランス型のガイドライン】
- モノ・情報・品質の流れを総合的に分析
- プロセス全体のバランス
- インプット→変換→アウトプットの全体像"""
        }

        guideline = focus_guidelines.get(options.focus, focus_guidelines["balanced"])

        system_prompt = f"""あなたは生産技術に20年以上従事するベテランのコンサルタントです。
プロセス工学とシステム分析に精通しています。

【重要な定義】
「機能カテゴリ」= プロセスを構成する動的な変換機能
- 単なる評価軸（品質、コスト）ではなく、製品を作り上げる動的なプロセスです
- インプット→変換→アウトプットの流れを持つ機能です
- 例：「材料を準備する」「部品を加工する」「組み立てる」「検査する」

【分析するプロセス】
プロセス名: {process_name}
プロセス概要: {process_description}

【分析方針】
{focus_desc}

{guideline}

【分析手順（Chain-of-Thought）】

ステップ1: プロセスの全体フローを理解する
- プロセスの開始から終了までの流れは？
- 主要なインプットは何か？（材料、部品、情報）
- 最終的なアウトプットは何か？（製品、データ）
- プロセスはどのような段階に分かれるか？（準備→加工→組立→検査→完成）

ステップ2: 動的な変換ステップを特定する
- 各段階でどのような変換が起こるか？
  - 物理的変換：切削、成形、組立、分解
  - 状態変換：加熱、冷却、乾燥、硬化
  - 位置変換：搬送、移動、配置
  - 情報変換：測定、検査、記録、判定
- 各変換のインプットとアウトプットは何か？

ステップ3: MECE原則に基づく機能分解
- Mutually Exclusive（相互排他的）：重複のない機能
- Collectively Exhaustive（全体網羅的）：プロセス全体をカバー
- {count_min}〜{count_max}個の動的機能に分解

ステップ4: 各機能の詳細化
各機能について：
- 機能名：動詞を含む動的な名称（「〜を〜する」形式）
- 説明：この機能が何を変換・処理するか（50-100文字）
- transformation_type：変換タイプ
  - preparation: 準備（材料手配、段取り）
  - processing: 加工（切削、成形、処理）
  - assembly: 組立（結合、接続、取り付け）
  - inspection: 検査（測定、確認、判定）
  - adjustment: 調整（チューニング、較正、修正）
  - packaging: 梱包（保護、包装）
  - transfer: 移動（搬送、移送）
- inputs: インプット（2-3個）
- outputs: アウトプット（1-2個）
- process_phase: プロセスフェーズ
  - preparation: 準備段階
  - main_process: 主要プロセス
  - verification: 検証段階
  - completion: 完了段階
- importance: 重要度（1-5）
- examples: 具体的な作業工程の例（2-3個）

【出力形式（必須）】
以下のJSON形式で出力してください。他の説明は一切不要です：
```json
[
  {{
    "name": "材料を準備する",
    "description": "必要な材料・部品を手配し、作業に適した状態にする",
    "transformation_type": "preparation",
    "inputs": ["材料リスト", "部品発注情報"],
    "outputs": ["作業準備済み材料"],
    "process_phase": "preparation",
    "importance": 4,
    "examples": ["材料の受入検査", "部品のピッキング", "作業台への配置"]
  }},
  ...
]
```"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"「{process_name}」の機能カテゴリを{focus_desc}で{count_min}〜{count_max}個抽出してください。"
            }
        ]

        response_text = self._call_with_retry(messages)

        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            categories_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"LLMからの応答がJSON形式ではありません: {response_text}",
                e.doc,
                e.pos,
            )

        if not isinstance(categories_data, list):
            raise ValueError(f"期待されるリスト形式ではありません: {type(categories_data)}")

        categories = []
        for cat_data in categories_data:
            try:
                category = FunctionalCategory(**cat_data)
                categories.append(category)
            except Exception as e:
                print(f"カテゴリデータの解析エラー: {e}")
                category = FunctionalCategory(
                    name=cat_data.get("name", "未定義"),
                    description=cat_data.get("description", ""),
                    transformation_type=cat_data.get("transformation_type", "processing"),
                    inputs=cat_data.get("inputs", []),
                    outputs=cat_data.get("outputs", []),
                    process_phase=cat_data.get("process_phase", "main_process"),
                    importance=cat_data.get("importance", 3),
                    examples=cat_data.get("examples", [])
                )
                categories.append(category)

        return categories

    def evaluate_node_pair(
        self, process_name: str, from_node: str, to_node: str
    ) -> Dict[str, Any]:
        """
        ノードペア間の影響を評価

        Args:
            process_name: プロセス名
            from_node: 評価元ノード
            to_node: 評価先ノード

        Returns:
            評価結果の辞書 {"score": int, "reason": str}

        Raises:
            json.JSONDecodeError: JSONパースエラー
            OpenAIError: API呼び出しエラー
        """
        system_prompt = f"""あなたは生産技術に20年以上従事するベテランのコンサルタントです。
インダストリアル・エンジニアリング（IE）の手法に精通しています。

今、"{process_name}"という生産プロセスについて分析しています。

「{from_node}」という工程/要素の性能を向上させるための変更が、
「{to_node}」という工程/要素に与える影響を評価してください。

評価スケール（必ず以下の値のみを使用）：
+9: 強い正の相関（直接的かつ大幅に改善される）
+3: 正の相関（改善される傾向にある）
+1: 弱い正の相関
 0: 無関係
-1: 弱い負の相関（トレードオフ）
-3: 負の相関（一般的なトレードオフ）
-9: 強い負の相関（直接的かつ大幅に悪化する）

思考プロセス（必須）：
1. {from_node}と{to_node}の生産プロセスにおける関係性を説明する
2. {from_node}への変更が{to_node}に与える影響の方向性（正か負か）を判断する
3. 影響の度合いを考慮し、評価スケールから最も適切な値を選択する
4. なぜその値を選んだのか、生産技術の観点から具体的な理由を述べる

結果は必ず以下のJSON形式で出力してください。他の説明文は不要です：
{{"score": (評価スコア), "reason": "思考プロセスを含む評価の具体的な理由"}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"「{from_node}」から「{to_node}」への影響を評価してください。",
            },
        ]

        response_text = self._call_with_retry(messages)

        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"LLMからの応答がJSON形式ではありません: {response_text}",
                e.doc,
                e.pos,
            )

        if "score" not in result or "reason" not in result:
            raise ValueError(
                f"期待されるキー（score, reason）が含まれていません: {result.keys()}"
            )

        if not isinstance(result["score"], (int, float)):
            raise ValueError(f"scoreが数値ではありません: {type(result['score'])}")

        if not isinstance(result["reason"], str):
            raise ValueError(f"reasonが文字列ではありません: {type(result['reason'])}")

        result["score"] = int(result["score"])

        if result["score"] not in settings.EVALUATION_SCALE_VALUES:
            valid_values = settings.EVALUATION_SCALE_VALUES
            result["score"] = min(
                valid_values, key=lambda x: abs(x - result["score"])
            )

        return result

    def evaluate_node_pair_with_idef0_context(
        self,
        process_name: str,
        from_node: str,
        to_node: str,
        from_type: str,
        to_type: str,
        from_category: str,
        to_category: str,
        evaluation_phase: str,
        phase_description: str,
    ) -> Dict[str, Any]:
        """
        IDEF0コンテキストを含むノードペア影響評価（Zigzagging手法）
        
        文献に基づき、ノードのIDEF0タイプと評価フェーズを考慮した
        一貫性のある評価を行います。疎で階層的な行列を生成します。
        
        Args:
            process_name: プロセス名
            from_node: 評価元ノード
            to_node: 評価先ノード
            from_type: 評価元ノードタイプ ("output" | "mechanism" | "input")
            to_type: 評価先ノードタイプ
            from_category: 評価元カテゴリ
            to_category: 評価先カテゴリ
            evaluation_phase: 評価フェーズ
            phase_description: フェーズの説明
        
        Returns:
            評価結果の辞書 {"score": int, "reason": str}
        
        Raises:
            json.JSONDecodeError: JSONパースエラー
            OpenAIError: API呼び出しエラー
        """
        type_labels = {
            "output": "性能 (Output/FR)",
            "mechanism": "手段 (Mechanism/DP)",
            "input": "材料 (Input/DP)"
        }
        
        from_type_label = type_labels.get(from_type, from_type)
        to_type_label = type_labels.get(to_type, to_type)
        
        phase_context = {
            "perf_to_char": """
【評価の文脈】
「{from_node}」は性能（目的・評価指標）であり、「{to_node}」は特性（手段・材料）です。
この性能を達成するために、この特性がどれほど重要かを評価してください。

【重要】直接的で強い影響がある場合のみ、非ゼロの評価をしてください。
間接的・弱い影響は 0（無関係）として扱い、疎な行列を生成します。""",
            "char_to_perf": """
【評価の文脈】
「{from_node}」は特性（手段・材料）であり、「{to_node}」は性能（目的・評価指標）です。
この特性を改善すると、この性能がどれほど向上するかを評価してください。

【重要】直接的で強い影響がある場合のみ、非ゼロの評価をしてください。
間接的・弱い影響は 0（無関係）として扱い、疎な行列を生成します。""",
            "perf_to_perf": """
【評価の文脈】
「{from_node}」と「{to_node}」は両方とも性能（目的・評価指標）です。
ある性能を向上させると、他の性能との間にトレードオフや相乗効果が生じるかを評価してください。

【重要】明確なトレードオフ関係や相乗効果がある場合のみ、非ゼロの評価をしてください。
多くの性能間は独立であるため、デフォルトは 0（無関係）です。""",
            "char_to_char": """
【評価の文脈】
「{from_node}」と「{to_node}」は両方とも特性（手段・材料）です。
ある特性を変更すると、他の特性にどのような影響があるかを評価してください。

【重要】物理的・構造的に直接影響がある場合のみ、非ゼロの評価をしてください。
多くの特性間は独立であるため、デフォルトは 0（無関係）です。"""
        }
        
        context = phase_context.get(evaluation_phase, "").format(
            from_node=from_node, to_node=to_node
        )
        
        system_prompt = f"""あなたは生産技術に20年以上従事するベテランのコンサルタントです。
インダストリアル・エンジニアリング（IE）とIDEF0モデリングに精通しています。

【プロセス情報】
プロセス名: {process_name}

【評価対象】
評価元: 「{from_node}」
  - タイプ: {from_type_label}
  - カテゴリ: {from_category}

評価先: 「{to_node}」
  - タイプ: {to_type_label}
  - カテゴリ: {to_category}

【評価フェーズ】
{phase_description}

{context}

【評価スケール（必ず以下の値のみを使用）】
+9: 強い正の相関（直接的かつ大幅に改善される）
+3: 正の相関（改善される傾向にある）
+1: 弱い正の相関
 0: 無関係（デフォルト：多くの場合はこれ）
-1: 弱い負の相関（トレードオフ）
-3: 負の相関（一般的なトレードオフ）
-9: 強い負の相関（直接的かつ大幅に悪化する）

【思考プロセス（必須）】
1. IDEF0の観点から、{from_node}と{to_node}の論理的依存関係を分析する
2. この評価フェーズ（{phase_description}）の文脈で、直接的な影響があるかを判断する
3. 影響がある場合、その方向性（正か負か）と度合いを判定する
4. なぜその値を選んだのか、IDEF0の論理的依存関係の観点から具体的な理由を述べる

【重要な原則】
- Zigzagging手法に基づき、疎な行列（直接的で強い影響のみ）を生成します
- 間接的な影響、弱い影響は 0 として扱います
- これにより、設計の論理的依存関係が明確化され、後戻りの少ない設計順序の導出が可能になります

結果は必ず以下のJSON形式で出力してください。他の説明文は不要です：
{{"score": (評価スコア), "reason": "IDEF0の論理的依存関係に基づく評価理由"}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"「{from_node}」から「{to_node}」への影響を、IDEF0の論理的依存関係に基づいて評価してください。"
            }
        ]

        response_text = self._call_with_retry(messages)

        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"LLMからの応答がJSON形式ではありません: {response_text}",
                e.doc,
                e.pos,
            )

        if "score" not in result or "reason" not in result:
            raise ValueError(
                f"期待されるキー（score, reason）が含まれていません: {result.keys()}"
            )

        if not isinstance(result["score"], (int, float)):
            raise ValueError(f"scoreが数値ではありません: {type(result['score'])}")

        if not isinstance(result["reason"], str):
            raise ValueError(f"reasonが文字列ではありません: {type(result['reason'])}")

        result["score"] = int(result["score"])

        if result["score"] not in settings.EVALUATION_SCALE_VALUES:
            valid_values = settings.EVALUATION_SCALE_VALUES
            result["score"] = min(
                valid_values, key=lambda x: abs(x - result["score"])
            )

        return result

    def extract_nodes_from_chat(self, chat_history: List[Dict[str, str]]) -> List[str]:
        """
        チャット履歴からノードを抽出

        Args:
            chat_history: チャット履歴

        Returns:
            ノードのリスト

        Raises:
            json.JSONDecodeError: JSONパースエラー
            OpenAIError: API呼び出しエラー
        """
        system_prompt = """これまでの対話履歴を分析し、
抽出される全てのノード（作業工程、道具、材料、スキルなど）を、
重複なくPythonのリスト形式で出力してください。

ノード名は簡潔で一意な名詞句にしてください。

例: ["材料を混ぜる", "オーブンで焼く", "品質検査", "梱包"]

他の説明は不要です。JSON配列のみを出力してください。"""

        messages = [{"role": "system", "content": system_prompt}] + chat_history
        messages.append(
            {
                "role": "user",
                "content": "この対話から抽出される全てのノードをリスト形式で出力してください。",
            }
        )

        response_text = self._call_with_retry(messages)

        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```python"):
            response_text = response_text[9:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            nodes = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"LLMからの応答がJSON形式ではありません: {response_text}",
                e.doc,
                e.pos,
            )

        if not isinstance(nodes, list):
            raise ValueError(f"期待されるリスト形式ではありません: {type(nodes)}")

        if not all(isinstance(node, str) for node in nodes):
            raise ValueError("すべてのノードが文字列である必要があります")

        nodes = list(dict.fromkeys(nodes))

        return nodes

    def generate_expert_response(
        self,
        process_name: str,
        categories: List[str],
        chat_history: List[Dict[str, str]],
        user_message: str,
    ) -> str:
        """
        エキスパートAIの応答を生成（技術的な補足・提案）

        Args:
            process_name: プロセス名
            categories: 機能カテゴリのリスト
            chat_history: これまでのチャット履歴
            user_message: ユーザーの最新メッセージ

        Returns:
            エキスパートの応答

        Raises:
            OpenAIError: API呼び出しエラー
        """
        system_prompt = f"""あなたは生産技術のエキスパートAIです。ユーザーの発言に対して、技術的な補足や追加の提案を行います。

【分析対象プロセス】
{process_name}

【定義済みの動的機能】
{', '.join(categories)}

【あなたの役割】
1. **技術的な補足**: ユーザーが言及した要素について、専門的な視点から補足します
2. **抜け漏れの指摘**: ユーザーが見落としている可能性のある重要な要素を提案します
3. **具体化の支援**: 曖昧な表現を、より具体的なノード候補に変換します
4. **インプット・アウトプットの明確化**: 各要素の入出力を確認します

【応答スタイル】
- 簡潔に1-3文で応答してください
- ユーザーの発言を受けて、「〜も重要ですね」「〜は考慮されていますか？」という形で補足
- 新しいノードの候補を提案する場合は、具体的な名称で提示

【注意】
- ファシリテーターではないので、質問を投げかける必要はありません
- 技術的な補足と提案に徹してください
"""

        conversation_for_expert = []
        for msg in chat_history:
            role_map = {"facilitator": "assistant", "expert": "assistant", "user": "user"}
            conversation_for_expert.append({
                "role": role_map.get(msg["role"], "assistant"),
                "content": f"[{msg['role'].upper()}] {msg['content']}"
            })

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_for_expert)
        messages.append({"role": "user", "content": user_message})

        response = self._call_with_retry(messages)

        return response

    def generate_facilitator_response(
        self,
        process_name: str,
        categories: List[str],
        chat_history: List[Dict[str, str]],
    ) -> str:
        """
        ファシリテーターAIの応答を生成（対話の主導・次の質問）

        Args:
            process_name: プロセス名
            categories: 機能カテゴリのリスト
            chat_history: これまでのチャット履歴（expertの応答まで含む）

        Returns:
            ファシリテーターの応答

        Raises:
            OpenAIError: API呼び出しエラー
        """
        system_prompt = f"""あなたはZigzagging対話を主導するファシリテーターAIです。

【分析対象プロセス】
{process_name}

【定義済みの動的機能】
{', '.join(categories)}

これらは、製品を作り上げるための動的な変換機能（インプット→変換→アウトプット）です。

【あなたの役割】
1. **対話の主導**: プロセス全体を網羅的に洗い出すよう、ユーザーを導きます
2. **次の質問**: ユーザーとエキスパートのやり取りを踏まえ、次に掘り下げるべき点を質問します
3. **構造化**: 定期的に、これまでに出たノードを整理して提示します
4. **カテゴリの網羅**: すべての機能カテゴリについて、具体的な要素が洗い出されるようにします

【Zigzaggingの手法】
- **動的機能 → 実体**: 各カテゴリの具体的な作業工程・設備・材料を洗い出す
- **実体 → 下位機能**: さらに詳細なステップに分解する
- **インプット・アウトプット**: 各ノードの入出力を明確にする

【応答スタイル】
- ユーザーとエキスパートの発言を踏まえ、次の質問を投げかけます
- 2-4文程度の簡潔な応答
- 「では次に〜について教えてください」「〜はどうでしょうか？」という形式
- 定期的に「ここまでで〜というノードが出ましたね」と整理

【注意】
- エキスパートではないので、技術的な補足は不要です
- 対話の進行と質問に徹してください
"""

        conversation_for_facilitator = []
        for msg in chat_history:
            role_map = {"facilitator": "assistant", "expert": "assistant", "user": "user"}
            conversation_for_facilitator.append({
                "role": role_map.get(msg["role"], "assistant"),
                "content": f"[{msg['role'].upper()}] {msg['content']}"
            })

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_for_facilitator)

        response = self._call_with_retry(messages)

        return response

    def generate_initial_facilitator_message(
        self, process_name: str, categories: List[str]
    ) -> str:
        """
        Zigzagging対話の初期メッセージ（ファシリテーター）を生成

        Args:
            process_name: プロセス名
            categories: 機能カテゴリのリスト

        Returns:
            初期メッセージ
        """
        if not categories:
            return "機能カテゴリを先に定義してください。"

        first_category = categories[0]

        message = f"""こんにちは！私はファシリテーターAIです。🎯

「{process_name}」のプロセス分析をお手伝いします。

定義された{len(categories)}個の動的機能カテゴリに基づき、プロセスの具体的なノード（作業工程、設備、材料、スキルなど）を洗い出していきます。

私と一緒に、技術エキスパートAI 🔬 が対話をサポートします。
- 私たちが議論を主導します
- あなたは適宜、現場の知識を提供してください

「会話を進める」ボタンで、私たちの議論が展開されます。"""

        return message

    def generate_ai_discussion(
        self,
        process_name: str,
        categories: List[str],
        chat_history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        AI同士の議論を生成（ソクラテス式対話・全カテゴリ対応）
        
        Args:
            process_name: プロセス名
            categories: 全機能カテゴリのリスト
            chat_history: これまでのチャット履歴
            
        Returns:
            議論のメッセージリスト [{"role": "facilitator/expert", "content": "..."}]
        """
        categories_str = "、".join(categories)
        
        system_prompt = f"""あなたはファシリテーターとエキスパートの2つのAIの議論を生成します。

【プロセス】{process_name}
【全機能カテゴリ】{categories_str}

【議論の目的】
IDEF0の考え方（Input-Mechanism-Output）でプロセス全体を分析し、ユーザーの暗黙知を引き出します。
全カテゴリを俯瞰しながら、プロセスの具体的な要素を洗い出していきます。

【議論の流れ】
1. Facilitator: プロセス全体を見渡し、まだ詳しく議論していないカテゴリや重要な観点を提示
2. Expert: 具体的なインプット・メカニズム・アウトプットの例を挙げる
3. Facilitator: ユーザーに対して、現場の具体例や知識を引き出すソクラテス式の質問
4. カテゴリ間の関連性や全体のつながりにも言及

【応答スタイル】
- 各発言は2-3文で簡潔に
- 自然な対話口調
- ユーザーの思考を引き出す質問で締めくくる
- 会話の進捗に応じて、未議論のカテゴリや深掘りすべき点を提示

【出力形式（JSON）】
[
  {{"role": "facilitator", "content": "..."}},
  {{"role": "expert", "content": "..."}},
  ...
]
"""

        conversation_context = []
        for msg in chat_history[-10:]:
            role_map = {"facilitator": "assistant", "expert": "assistant", "user": "user"}
            conversation_context.append({
                "role": role_map.get(msg["role"], "assistant"),
                "content": f"[{msg['role'].upper()}] {msg['content']}"
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"プロセス全体（{categories_str}）について、AIたちの議論を生成してください。"}
        ]
        
        if conversation_context:
            messages.insert(1, {"role": "assistant", "content": "これまでの会話を踏まえます。"})
            messages.extend(conversation_context[-3:])

        response_text = self._call_with_retry(messages)
        
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            discussion = json.loads(response_text)
            if isinstance(discussion, list):
                return discussion
            else:
                return []
        except json.JSONDecodeError:
            return [
                {"role": "facilitator", "content": "プロセス全体を見渡して、具体的な要素を洗い出していきましょう。"},
                {"role": "expert", "content": "まず、各工程のインプットとアウトプットを明確にすることが重要ですね。"},
                {"role": "facilitator", "content": "あなたの現場では、どのような材料や情報が各工程に入ってきますか？"}
            ]

    def extract_all_idef0_nodes_from_chat(
        self,
        process_name: str,
        process_description: str,
        categories: List[str],
        chat_history: List[Dict[str, str]],
    ) -> Dict[str, IDEF0Node]:
        """
        会話から全カテゴリのIDEF0ノードを一括抽出
        
        Args:
            process_name: プロセス名
            process_description: プロセス概要
            categories: 全機能カテゴリのリスト
            chat_history: チャット履歴
            
        Returns:
            カテゴリ名をキーとしたIDEF0Nodeの辞書
        """
        recent_messages = chat_history[-15:] if len(chat_history) > 15 else chat_history
        
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in recent_messages
        ])

        categories_str = "、".join(categories)

        system_prompt = f"""あなたは生産プロセスをIDEF0で分析する専門家です。

【プロセス情報】
プロセス名: {process_name}
プロセス概要: {process_description}
全機能カテゴリ: {categories_str}

【IDEF0の3軸】
1. **Input（インプット）**: この機能に入ってくるもの（材料、部品、情報、データなど）
2. **Mechanism（メカニズム）**: この機能を実現する手段（設備、道具、作業工程、スキルなど）
3. **Output（アウトプット）**: この機能から出てくるもの（加工品、データ、記録など）

【指示】
会話から各カテゴリのInput/Mechanism/Outputを抽出してください。
- 各要素は簡潔な名詞句（10文字以内）
- 重複を避ける
- 会話に情報がないカテゴリは空配列でOK

【出力形式（JSON）】
{{
  "カテゴリ1": {{
    "category": "カテゴリ1",
    "inputs": ["要素1", "要素2", ...],
    "mechanisms": ["要素1", "要素2", ...],
    "outputs": ["要素1", "要素2", ...]
  }},
  "カテゴリ2": {{...}},
  ...
}}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"以下の会話から、全カテゴリのIDEF0ノードを抽出してください:\n\n{conversation_text}"}
        ]

        response_text = self._call_with_retry(messages)
        
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            all_idef0_data = json.loads(response_text)
            result = {}
            for cat_name in categories:
                if cat_name in all_idef0_data:
                    result[cat_name] = IDEF0Node(**all_idef0_data[cat_name])
                else:
                    result[cat_name] = IDEF0Node(
                        category=cat_name,
                        inputs=[],
                        mechanisms=[],
                        outputs=[]
                    )
            return result
        except (json.JSONDecodeError, Exception) as e:
            print(f"全カテゴリIDEF0抽出エラー: {e}")
            return {cat: IDEF0Node(category=cat, inputs=[], mechanisms=[], outputs=[]) for cat in categories}

    def extract_nodes_in_idef0_format(
        self,
        process_name: str,
        process_description: str,
        current_category: str,
        chat_history: List[Dict[str, str]],
        existing_idef0: Optional[IDEF0Node] = None,
    ) -> IDEF0Node:
        """
        会話からIDEF0形式（Input-Mechanism-Output）でノードを抽出
        
        Args:
            process_name: プロセス名
            process_description: プロセス概要
            current_category: 現在の機能カテゴリ
            chat_history: 最新のチャット履歴
            existing_idef0: 既存のIDEF0ノード（更新の場合）
            
        Returns:
            IDEF0Node: Input-Mechanism-Outputの構造化データ
        """
        recent_messages = chat_history[-10:] if len(chat_history) > 10 else chat_history
        
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in recent_messages
        ])

        existing_info = ""
        if existing_idef0:
            existing_info = f"""
【既存の抽出結果】
Input: {', '.join(existing_idef0.inputs) if existing_idef0.inputs else 'なし'}
Mechanism: {', '.join(existing_idef0.mechanisms) if existing_idef0.mechanisms else 'なし'}
Output: {', '.join(existing_idef0.outputs) if existing_idef0.outputs else 'なし'}
"""

        system_prompt = f"""あなたは生産プロセスをIDEF0の考え方で分析する専門家です。

【プロセス情報】
プロセス名: {process_name}
プロセス概要: {process_description}
現在の機能カテゴリ: {current_category}

【IDEF0の3軸】
1. **Input（インプット）**: この機能に入ってくるもの
   - 材料、部品、半製品
   - 情報、データ、指示書
   - 前工程からの産出物

2. **Mechanism（メカニズム）**: この機能を実現する手段
   - 設備、機械、装置
   - 道具、治具、測定器
   - 作業工程、手順
   - 作業者のスキル、知識
   - ソフトウェア、システム

3. **Output（アウトプット）**: この機能から出てくるもの
   - 加工品、組立品、完成品
   - 検査データ、記録
   - 次工程への引き渡し物
   - 不良品、廃棄物

{existing_info}

【指示】
以下の会話から、「{current_category}」のInput/Mechanism/Outputを抽出してください。
- 既存の抽出結果がある場合は、それに追加・更新する形で出力
- 各要素は簡潔な名詞句（10文字以内）
- 重複を避ける

【出力形式（JSON）】
{{
  "category": "{current_category}",
  "inputs": ["要素1", "要素2", ...],
  "mechanisms": ["要素1", "要素2", ...],
  "outputs": ["要素1", "要素2", ...]
}}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"以下の会話から、IDEF0形式でノードを抽出してください:\n\n{conversation_text}"}
        ]

        response_text = self._call_with_retry(messages)
        
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            idef0_data = json.loads(response_text)
            return IDEF0Node(**idef0_data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"IDEF0抽出エラー: {e}")
            return IDEF0Node(
                category=current_category,
                inputs=[],
                mechanisms=[],
                outputs=[]
            )

    def generate_diverse_idef0_nodes(
        self,
        process_name: str,
        process_description: str,
        current_category: str,
        num_perspectives: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Verbalized Samplingで多様なIDEF0ノードを生成
        
        Args:
            process_name: プロセス名
            process_description: プロセス概要
            current_category: 現在の機能カテゴリ
            num_perspectives: 生成する視点の数（デフォルト5）
            
        Returns:
            視点ごとのIDEF0ノードリスト
            [
              {
                "perspective": "品質重視",
                "probability": 0.25,
                "description": "品質管理と検査を最優先する視点",
                "idef0": {
                  "category": "...",
                  "inputs": [...],
                  "mechanisms": [...],
                  "outputs": [...]
                }
              },
              ...
            ]
        """
        system_prompt = f"""あなたは生産プロセスをIDEF0で分析する専門家です。

【Verbalized Sampling指示】
「{current_category}」という機能カテゴリについて、異なる{num_perspectives}つの思考モード・視点から
IDEF0ノード（Input-Mechanism-Output）を生成してください。

【プロセス情報】
プロセス名: {process_name}
プロセス概要: {process_description}
機能カテゴリ: {current_category}

【重要】
各視点は互いに異なる「重視点」「思考モード」を反映し、多様な解釈を提供すること。
AIの創造性と多様性を最大限に引き出してください。

【推奨される視点の例（これらに限定されない）】
1. 品質重視：品質管理・検査・精度を最優先
2. 効率重視：時間短縮・コスト削減・自動化を最優先
3. 安全性重視：作業者の安全・リスク管理を最優先
4. イノベーション重視：新技術・改善・最新手法を重視
5. 標準作業重視：確立された手順・マニュアル化を重視
6. 柔軟性重視：変更対応・カスタマイズを重視
7. トレーサビリティ重視：記録・追跡・可視化を重視

【各視点について】
1. 視点名（perspective）：簡潔な名称
2. 確率（probability）：この視点の妥当性・適用可能性（0.0-1.0）
3. 説明（description）：この視点が何を重視するか（30文字程度）
4. IDEF0ノード：
   - inputs: インプット（3-5個）
   - mechanisms: メカニズム（3-7個）
   - outputs: アウトプット（2-4個）

【出力形式（JSON）】
[
  {{
    "perspective": "品質重視",
    "probability": 0.25,
    "description": "品質管理と精密検査を最優先する視点",
    "idef0": {{
      "category": "{current_category}",
      "inputs": ["材料仕様書", "品質基準", "未検査部品"],
      "mechanisms": ["精密測定機", "全数検査", "品質記録システム"],
      "outputs": ["検査済み部品", "品質報告書", "不良品"]
    }}
  }},
  {{
    "perspective": "効率重視",
    "probability": 0.22,
    "description": "時間短縮と自動化を最優先する視点",
    "idef0": {{
      "category": "{current_category}",
      "inputs": ["部品リスト", "検査基準"],
      "mechanisms": ["自動検査装置", "サンプリング検査", "バーコードスキャン"],
      "outputs": ["合否判定", "検査データ"]
    }}
  }},
  ...
]
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"「{current_category}」について、{num_perspectives}つの異なる思考モードから多様なIDEF0ノードを生成してください。"
            }
        ]

        response_text = self._call_with_retry(messages)
        
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            perspectives = json.loads(response_text)
            if isinstance(perspectives, list):
                for persp in perspectives:
                    if "probability" not in persp:
                        persp["probability"] = 1.0 / num_perspectives
                return perspectives
            else:
                return []
        except json.JSONDecodeError as e:
            print(f"多様性生成エラー: {e}")
            return []

    def refine_idef0_with_zigzagging(
        self,
        process_name: str,
        category: str,
        current_idef0: Dict[str, List[str]],
        refinement_depth: int = 1,
    ) -> Dict[str, List[str]]:
        """
        Zigzagging手法でIDEF0ノードを細分化
        
        反復的な知識精緻化プロセスを支援するため、既存のIDEF0ノードを
        段階的に細かい粒度に分解します。
        
        Args:
            process_name: プロセス名
            category: カテゴリ名
            current_idef0: 現在のIDEF0ノード
                          {"inputs": [...], "mechanisms": [...], "outputs": [...]}
            refinement_depth: 細分化の深さ（1: 軽度, 2: 中程度, 3: 詳細）
        
        Returns:
            細分化後のIDEF0ノード
        
        Raises:
            json.JSONDecodeError: JSONパースエラー
            OpenAIError: API呼び出しエラー
        """
        depth_descriptions = {
            1: "各要素を2-3個の下位要素に分解",
            2: "各要素を3-5個の下位要素に詳細分解",
            3: "各要素を5-7個の下位要素に徹底的に分解"
        }
        
        depth_desc = depth_descriptions.get(refinement_depth, depth_descriptions[1])
        
        inputs_str = "、".join(current_idef0.get("inputs", []))
        mechanisms_str = "、".join(current_idef0.get("mechanisms", []))
        outputs_str = "、".join(current_idef0.get("outputs", []))
        
        system_prompt = f"""あなたは生産プロセス分析の専門家です。
Zigzagging手法により、IDEF0ノードの粒度を段階的に細かくします。

【プロセス情報】
プロセス名: {process_name}
カテゴリ: {category}

【現在のIDEF0ノード】
- Output（性能・成果物）: {outputs_str}
- Mechanism（手段・手順）: {mechanisms_str}
- Input（材料・情報）: {inputs_str}

【タスク: Zigzaggingによる細分化】
{depth_desc}してください。

**手順1: Output（性能）の細分化**
各Outputをさらに細かい性能指標・品質要素に分解してください。
- 問い: 「この成果物の品質を評価する際、どのような細かい指標がありますか？」
- 例: "焼き色の均一性" → ["表面の焼き色", "内部の火の通り具合", "層の密着度"]

**手順2: Mechanism（手段）の細分化**
各Mechanismをさらに細かい作業手順・ステップに分解してください。
- 問い: 「この手段を実行する際、具体的にどのような細かい作業が含まれますか？」
- 例: "フライパンで焼く" → ["フライパン予熱", "温度調整", "卵液投入", "火加減調整", "巻き操作"]

**手順3: Input（材料）の細分化**
各Inputをさらに細かい構成要素に分解してください。
- 問い: 「この材料・情報は、どのような細かい要素で構成されていますか？」
- 例: "卵液" → ["卵", "砂糖", "塩", "出汁"]

【重要な原則】
1. 粒度の一貫性: 同じ抽象度レベルで分解すること
2. MECE原則: 重複なく、漏れなく（Mutually Exclusive, Collectively Exhaustive）
3. 計測可能性: 各要素は観測・計測可能な具体的なものにすること
4. プロセスとの関連性: "{process_name}"のプロセスに直接関連する要素のみを含めること

【出力形式（JSON）】
{{
  "outputs": ["細分化後のOutput1", "Output2", ...],
  "mechanisms": ["細分化後のMechanism1", "Mechanism2", ...],
  "inputs": ["細分化後のInput1", "Input2", ...]
}}

他の説明は不要です。JSON形式のみを出力してください。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"「{category}」のIDEF0ノードをZigzagging手法で細分化してください（深さレベル: {refinement_depth}）。"
            }
        ]

        response_text = self._call_with_retry(messages)

        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            refined_idef0 = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"LLMからの応答がJSON形式ではありません: {response_text}",
                e.doc,
                e.pos,
            )

        if not isinstance(refined_idef0, dict):
            raise ValueError(f"期待される辞書形式ではありません: {type(refined_idef0)}")

        for key in ["inputs", "mechanisms", "outputs"]:
            if key not in refined_idef0:
                refined_idef0[key] = []
            if not isinstance(refined_idef0[key], list):
                refined_idef0[key] = []

        refined_idef0["category"] = category

        return refined_idef0

    def generate_single_perspective_idef0(
        self,
        process_name: str,
        process_description: str,
        categories: List[str],
        perspective_examples: str = """1. 品質重視：品質管理・検査・精度を最優先
2. 効率重視：時間短縮・コスト削減・自動化を最優先
3. 安全性重視：作業者の安全・リスク管理を最優先
4. イノベーション重視：新技術・改善・最新手法を重視
5. 標準作業重視：確立された手順・マニュアル化を重視""",
    ) -> Dict[str, Any]:
        """
        1つの視点から全カテゴリのIDEF0ノードを生成
        
        Args:
            process_name: プロセス名
            process_description: プロセス概要
            categories: 全機能カテゴリのリスト
            perspective_examples: 視点の例（プロンプト用）
            
        Returns:
            単一視点のIDEF0ノードデータ
            {
              "perspective": "品質重視",
              "probability": 0.25,
              "description": "品質管理と検査を最優先する視点",
              "idef0_nodes": {
                "カテゴリ1": {...},
                "カテゴリ2": {...},
                ...
              }
            }
        """
        categories_str = "、".join(categories)
        
        system_prompt = f"""あなたは生産プロセスをIDEF0で分析する専門家です。

【指示】
プロセス全体を1つの特定の思考モード・視点から分析し、
各カテゴリのIDEF0ノード（Input-Mechanism-Output）を一括生成してください。

【プロセス情報】
プロセス名: {process_name}
プロセス概要: {process_description}

【全機能カテゴリ】
{categories_str}

【重要】
- 明確な「重視点」を持つ視点を1つ選択すること
- 全カテゴリについて、その視点から見たIDEF0ノードを生成すること
- プロセス全体の一貫性を保つこと

【視点の例】
{perspective_examples}

【出力形式（JSON）】
{{
  "perspective": "品質重視",
  "probability": 0.2,
  "description": "品質管理と精密検査を最優先する視点",
  "idef0_nodes": {{
    "材料準備": {{
      "category": "材料準備",
      "inputs": ["材料仕様書", "品質基準"],
      "mechanisms": ["受入検査", "測定機器"],
      "outputs": ["検査済み材料", "検査記録"]
    }},
    "加工": {{
      "category": "加工",
      "inputs": ["検査済み材料", "加工指示"],
      "mechanisms": ["精密加工機", "品質モニタリング"],
      "outputs": ["加工品", "測定データ"]
    }}
  }}
}}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"全カテゴリ（{categories_str}）について、1つの明確な視点から一括生成してください。"
            }
        ]

        response_text = self._call_with_retry(messages)
        
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            perspective = json.loads(response_text)
            
            # カテゴリ名のマッピング（柔軟なマッチング）
            if "idef0_nodes" in perspective:
                idef0_nodes_remapped = {}
                for generated_cat_name, idef0_data in perspective["idef0_nodes"].items():
                    # 完全一致を優先
                    if generated_cat_name in categories:
                        idef0_nodes_remapped[generated_cat_name] = idef0_data
                    else:
                        # 部分一致を試行
                        matched = False
                        for actual_cat in categories:
                            if generated_cat_name in actual_cat or actual_cat in generated_cat_name:
                                idef0_nodes_remapped[actual_cat] = idef0_data
                                idef0_data["category"] = actual_cat
                                matched = True
                                print(f"  🔄 カテゴリ名マッピング: '{generated_cat_name}' → '{actual_cat}'")
                                break
                        
                        if not matched:
                            print(f"  ⚠️ カテゴリ名が一致しません: '{generated_cat_name}' (スキップ)")
                
                perspective["idef0_nodes"] = idef0_nodes_remapped
            
            if "probability" not in perspective:
                perspective["probability"] = 0.2
                
            return perspective
            
        except json.JSONDecodeError as e:
            print(f"\n❌ 単一視点生成エラー - JSONパースエラー")
            print(f"エラー詳細: {e}")
            return {}
        except Exception as e:
            print(f"\n❌ 単一視点生成エラー - 予期しないエラー: {e}")
            return {}

    def generate_diverse_idef0_nodes_all_categories(
        self,
        process_name: str,
        process_description: str,
        categories: List[str],
        num_perspectives: int = 3,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Verbalized Samplingで全カテゴリのIDEF0ノードを段階的生成（改善版）
        
        Args:
            process_name: プロセス名
            process_description: プロセス概要
            categories: 全機能カテゴリのリスト
            num_perspectives: 生成する視点の数（デフォルト3、推奨: 3-5）
            progress_callback: 進捗通知コールバック関数 callback(current, total, perspective_name)
            
        Returns:
            視点ごとの全カテゴリIDEF0ノードリスト
        """
        perspectives = []
        
        # 視点の例を定義
        perspective_examples = """1. 品質重視：品質管理・検査・精度を最優先
2. 効率重視：時間短縮・コスト削減・自動化を最優先
3. 安全性重視：作業者の安全・リスク管理を最優先
4. イノベーション重視：新技術・改善・最新手法を重視
5. 標準作業重視：確立された手順・マニュアル化を重視"""
        
        print(f"\n🎲 {num_perspectives}つの視点を段階的に生成します...")
        
        # 各視点を順次生成
        for i in range(num_perspectives):
            print(f"\n--- 視点 {i+1}/{num_perspectives} を生成中 ---")
            
            # プログレスコールバック呼び出し
            if progress_callback:
                progress_callback(i, num_perspectives, f"視点{i+1}")
            
            # 単一視点を生成
            perspective = self.generate_single_perspective_idef0(
                process_name=process_name,
                process_description=process_description,
                categories=categories,
                perspective_examples=perspective_examples,
            )
            
            if perspective and "idef0_nodes" in perspective:
                perspective_name = perspective.get("perspective", f"視点{i+1}")
                num_categories = len(perspective.get("idef0_nodes", {}))
                print(f"✅ {perspective_name}: {num_categories}カテゴリ生成完了")
                perspectives.append(perspective)
            else:
                print(f"⚠️ 視点{i+1}の生成に失敗しました（スキップ）")
        
        # 確率値を正規化
        if perspectives:
            for persp in perspectives:
                persp["probability"] = 1.0 / len(perspectives)
        
        print(f"\n✅ 合計{len(perspectives)}個の視点を生成しました")
        return perspectives

    def generate_diverse_category_sets(
        self,
        process_name: str,
        process_description: str,
        num_perspectives: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Verbalized Samplingで多様な機能カテゴリセットを生成
        
        Args:
            process_name: プロセス名
            process_description: プロセス概要
            num_perspectives: 生成する視点の数（デフォルト5）
            
        Returns:
            視点ごとの機能カテゴリセットリスト
            [
              {
                "perspective": "品質管理重視",
                "probability": 0.25,
                "description": "品質確保と検査プロセスに焦点を当てた分析",
                "categories": [
                  {"name": "材料受入検査", "description": "...", ...},
                  ...
                ]
              },
              ...
            ]
        """
        system_prompt = f"""あなたは生産プロセス分析の専門家です。

【Verbalized Sampling指示】
「{process_name}」について、異なる{num_perspectives}つの分析哲学・思考モードから
機能カテゴリ（動的な変換プロセス）を生成してください。

【プロセス情報】
プロセス名: {process_name}
プロセス概要: {process_description}

【重要】
各視点は互いに異なる「分析哲学」「重視する側面」を反映し、多様なカテゴリ分解を提供すること。
AIの創造性と多様性を最大限に引き出してください。

【推奨される分析哲学の例（これらに限定されない）】
1. 品質管理重視：検査・品質ゲート・トレーサビリティを中心にカテゴリ分解
2. リーン生産重視：ムダ排除・流れ最適化・JIT（ジャストインタイム）の視点
3. IoT/DX重視：データ収集・可視化・デジタルツイン・自動化の視点
4. 作業者中心：人の動き・スキル・安全性・エルゴノミクスの視点
5. サプライチェーン重視：前後工程との連携・物流・在庫管理の視点
6. 設備保全重視：機械・メンテナンス・稼働率の視点
7. コスト管理重視：原価・効率・ロス削減の視点

【機能カテゴリの定義】
「機能カテゴリ」= プロセスを構成する動的な変換機能（インプット→変換→アウトプット）
- 例：「材料を準備する」「部品を加工する」「組み立てる」「検査する」

【各視点について】
1. 視点名（perspective）：分析哲学の名称
2. 確率（probability）：この視点の妥当性・適用可能性（0.0-1.0）
3. 説明（description）：この視点が何を重視するか（40文字程度）
4. カテゴリ数：6-8個が標準
5. カテゴリリスト：各カテゴリの詳細情報
   - name: 動的プロセス名
   - description: 変換内容の説明
   - transformation_type: preparation/processing/assembly/inspection/adjustment/packaging/transfer
   - inputs: インプット（2-3個）
   - outputs: アウトプット（1-2個）
   - process_phase: preparation/main_process/verification/completion
   - importance: 重要度（1-5）
   - examples: 具体例（2-3個）

【出力形式（JSON）】
[
  {{
    "perspective": "品質管理重視",
    "probability": 0.28,
    "description": "品質確保と検査プロセスに焦点を当てた分析哲学",
    "categories": [
      {{
        "name": "材料受入検査",
        "description": "供給された材料の品質を確認する",
        "transformation_type": "inspection",
        "inputs": ["納品材料", "品質基準書"],
        "outputs": ["合格材料", "検査記録"],
        "process_phase": "preparation",
        "importance": 5,
        "examples": ["寸法測定", "材質確認", "外観検査"]
      }},
      ...
    ]
  }},
  {{
    "perspective": "リーン生産重視",
    "probability": 0.24,
    "description": "ムダ排除と流れ最適化を重視した分析哲学",
    "categories": [...]
  }},
  ...
]
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"「{process_name}」について、{num_perspectives}つの異なる分析哲学から多様な機能カテゴリセットを生成してください。"
            }
        ]

        response_text = self._call_with_retry(messages)
        
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            perspectives = json.loads(response_text)
            if isinstance(perspectives, list):
                for persp in perspectives:
                    if "probability" not in persp:
                        persp["probability"] = 1.0 / num_perspectives
                return perspectives
            else:
                return []
        except json.JSONDecodeError as e:
            print(f"カテゴリ多様性生成エラー: {e}")
            return []
    
    def evaluate_dsm_parameters(
        self,
        process_name: str,
        process_description: str,
        nodes: List[str],
        idef0_nodes: Dict[str, Dict[str, Any]],
        node_classifications: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        DSM最適化パラメータをLLMで評価
        
        Args:
            process_name: プロセス名
            process_description: プロセス概要
            nodes: 全ノードのリスト
            idef0_nodes: IDEF0ノードデータ（カテゴリごと）
            node_classifications: ノード分類（node_name -> "FR" or "DP"）
        
        Returns:
            {
                "parameters": {
                    "node_name": {
                        "cost": float (1-5),
                        "range": float (0.1-2.0),
                        "importance": float (1-5, FRのみ),
                        "structure": str (グループ名)
                    },
                    ...
                },
                "reasoning": str (評価の根拠)
            }
        """
        
        # ノード情報を整理
        fr_nodes = [n for n, t in node_classifications.items() if t == "FR"]
        dp_nodes = [n for n, t in node_classifications.items() if t == "DP"]
        
        # 各ノードのカテゴリを取得
        node_categories = {}
        for category_name, idef0_data in idef0_nodes.items():
            for output in idef0_data.get("outputs", []):
                node_categories[output] = category_name
            for mechanism in idef0_data.get("mechanisms", []):
                node_categories[mechanism] = category_name
            for input_node in idef0_data.get("inputs", []):
                node_categories[input_node] = category_name
        
        system_prompt = f"""あなたは生産技術に20年以上従事するベテランのコンサルタントであり、設計構造マトリクス（DSM）最適化の専門家です。

# タスク
「{process_name}」プロセスの各ノードについて、DSM最適化に必要な以下のパラメータを評価してください：

## 評価するパラメータ

### 1. Cost（コスト） - DP（設計パラメータ）のみ
**スケール: 1-5**
- **1**: 低コスト（簡単な調整、既存リソースで対応可能）
- **2**: やや低コスト（小規模な投資、軽微な訓練）
- **3**: 中コスト（一部投資、訓練必要、外部調達あり）
- **4**: やや高コスト（大規模投資、高度な技術必要）
- **5**: 高コスト（巨額投資、専門設備・技術、長期開発）

**考慮要素**: 材料費、人件費、設備投資、技術的難易度

### 2. Range（変動範囲） - DP（設計パラメータ）のみ
**スケール: 0.1-2.0**
- **0.1**: 非常に狭い（ほとんど変更不可、固定仕様）
- **0.5**: 狭い（限定的な調整幅）
- **1.0**: 標準（通常の調整幅）
- **1.5**: 広い（多様な選択肢）
- **2.0**: 非常に広い（非常に柔軟、多様な選択肢）

**考慮要素**: 変更の柔軟性、選択肢の多様性、技術的制約

### 3. Importance（重要度） - FR（機能要求）のみ
**スケール: 1-5**
- **1**: 低（オプション機能、美観、付加価値）
- **2**: やや低（利便性、効率の微改善）
- **3**: 中（品質、効率、顧客満足度）
- **4**: やや高（主要機能、競争力）
- **5**: 高（安全性、法規制、コア機能、存続に関わる）

**考慮要素**: ビジネス価値、安全性、品質への影響、法規制

### 4. Structure（構造グループ） - すべてのノード
**論理的なグループ名**
同じ部品、工程、システムに属するノードをグループ化します。

**例**: "加工系", "検査系", "材料系", "制御系", "搬送系"など

# プロセス情報
- **プロセス名**: {process_name}
- **概要**: {process_description}

# ノード情報

## FR（機能要求 - Outputノード）: {len(fr_nodes)}個
{chr(10).join([f"- {node} (カテゴリ: {node_categories.get(node, '不明')})" for node in fr_nodes])}

## DP（設計パラメータ - Mechanism + Inputノード）: {len(dp_nodes)}個
{chr(10).join([f"- {node} (カテゴリ: {node_categories.get(node, '不明')})" for node in dp_nodes])}

# 出力形式（厳守）
以下のJSON形式で出力してください：

```json
{{
  "parameters": {{
    "ノード名1": {{
      "cost": 3.5,  // DPのみ、FRの場合は省略
      "range": 1.2,  // DPのみ、FRの場合は省略
      "importance": 4,  // FRのみ、DPの場合は省略
      "structure": "グループ名"
    }},
    "ノード名2": {{...}},
    ...
  }},
  "reasoning": "評価の根拠と全体的な考察を200-400文字で記述"
}}
```

# 重要な注意事項
1. すべてのノードについて評価を提供してください
2. FRにはimportanceとstructureのみ、DPにはcost, range, structureを設定
3. スケールを厳守してください
4. reasoningには、プロセス全体の特性と評価の根拠を記述
5. 出力は必ずJSON形式のみ（他の説明文は不要）"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"「{process_name}」プロセスの全ノードについて、DSMパラメータを評価してください。"
            }
        ]
        
        response_text = self._call_with_retry(messages)
        
        # JSONパース
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            result = json.loads(response_text)
            
            # バリデーション
            if "parameters" not in result:
                raise ValueError("'parameters'キーが見つかりません")
            if "reasoning" not in result:
                result["reasoning"] = "評価根拠が提供されませんでした"
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"DSMパラメータ評価エラー: {e}")
            print(f"レスポンス: {response_text[:500]}")
            return {
                "parameters": {},
                "reasoning": f"JSONパースエラー: {str(e)}"
            }
    
    def evaluate_category_batch(
        self,
        category: str,
        idef0_data: Dict[str, Any],
        pairs: List[Dict[str, Any]],
        process_name: str
    ) -> List[Dict[str, Any]]:
        """
        1カテゴリ内の全評価ペアを一括評価
        
        同一カテゴリ内のノード間評価を一度のLLM呼び出しで実行。
        全体コンテキスト（inputs, mechanisms, outputs）を把握した上で、
        疎行列の原則に基づき強い影響のみを非ゼロとして評価。
        
        Args:
            category: カテゴリ名
            idef0_data: このカテゴリのIDEF0データ
                       {"function": str, "inputs": [...], "mechanisms": [...], "outputs": [...]}
            pairs: このカテゴリ内の評価ペアリスト
            process_name: プロセス名
        
        Returns:
            評価結果リスト
            [
                {
                    "from_node": str,
                    "to_node": str,
                    "score": int,
                    "reason": str
                },
                ...
            ]
        """
        if not pairs:
            return []
        
        # ペアリストをフェーズごとに整理
        pairs_by_phase: Dict[str, List[Dict]] = {}
        for pair in pairs:
            phase = pair.get("evaluation_phase", "unknown")
            if phase not in pairs_by_phase:
                pairs_by_phase[phase] = []
            pairs_by_phase[phase].append(pair)
        
        # システムプロンプト構築
        system_prompt = f"""あなたは生産プロセス分析に20年以上従事するベテランコンサルタントです。
IDEF0モデリングとZigzagging手法の専門家として、ノード間の影響を評価してください。

# 評価対象プロセス
プロセス名: {process_name}
カテゴリ: {category}
機能: {idef0_data.get('function', '')}

# IDEF0構造

## Outputs（性能・成果物）
{chr(10).join(f'- {o}' for o in idef0_data.get('outputs', []))}

## Mechanisms（手段・道具）
{chr(10).join(f'- {m}' for m in idef0_data.get('mechanisms', []))}

## Inputs（材料・情報）
{chr(10).join(f'- {i}' for i in idef0_data.get('inputs', []))}

# Zigzagging評価原則（疎で階層的な行列の生成）

あなたは公理的設計の専門家として、**Zigzagging思考プロセス**を適用してください。

## 評価の問いかけ（How推論）

各ペアについて、以下の問いかけで論理的な依存関係を判断してください：

### フェーズ1（性能→特性）
**問い**: 「この性能(Output)を達成するために、この特性(Mechanism/Input)は【どのように(How)】貢献するか？」

### フェーズ2（特性→性能）
**問い**: 「この特性(Mechanism/Input)を改善すると、この性能(Output)は【どのように(How)】向上するか？」

### フェーズ3（性能間）
**問い**: 「この性能を向上させると、他の性能との間に【どのような(How)】トレードオフが生じるか？」

### フェーズ4（特性間）
**問い**: 「この特性を変更すると、他の特性に【どのような(How)】影響があるか？」

## スコアリング基準（Zigzaggingの原理）

### ✅ 論理的なHow関係が明確 → 非ゼロ評価
- **±7～±9**: 決定的な因果関係（この要素なしでは成立しない）
  - 例: 「攪拌回数」→「卵液の均一性」: +8（直接的な物理的因果）
- **±4～±6**: 顕著な因果関係（品質や効率に明確な差）
  - 例: 「フライパン材質」→「焼き色の均一性」: +5（熱伝導の影響）
- **±1～±3**: 軽微な因果関係（改善余地はあるが主要因ではない）
  - 例: 「作業環境の温度」→「巻き速度」: +2（作業性への軽微な影響）

### ❌ How関係が不明確・間接的 → 0評価
- 論理的な因果関係が説明できない
- 他の要因を経由する間接的な影響のみ
- 例: 「卵の温度」→「層の密着度」: 0（間接的、他の要因経由）

### 負の影響（トレードオフ）
- 一方を改善すると他方が悪化する **明確な**トレードオフ関係
- 例: 「加工速度↑」→「精度↓」: -7（速度と精度の物理的トレードオフ）

## 疎行列の厳守（文献の原理）

**重要**: 設計の論理的依存関係（親子関係）に沿った、**直接的で強い影響のみ**を抽出してください。

- 間接的な関係や弱い相関は0としてください
- 「疎で階層的」な構造を維持してください
- 総当たり的な評価ではなく、論理的な依存関係に基づく評価を行ってください

# タスク
以下の評価ペアについて、Zigzagging思考プロセス（How推論）に基づき、スコアと理由を提示してください。
"""

        # ユーザープロンプト構築
        pair_descriptions = []
        for idx, pair in enumerate(pairs, 1):
            phase_name = pair.get("phase_name", "")
            phase_desc = pair.get("phase_description", "")
            from_node = pair["from_node"]
            to_node = pair["to_node"]
            
            pair_descriptions.append(
                f"{idx}. 【{phase_name}】\n"
                f"   評価元: {from_node}\n"
                f"   評価先: {to_node}\n"
                f"   観点: {phase_desc}"
            )
        
        user_prompt = f"""# 評価ペアリスト（全{len(pairs)}件）

{chr(10).join(pair_descriptions)}

# 出力形式（必ずJSON形式で）

{{
  "evaluations": [
    {{
      "from_node": "ノード名",
      "to_node": "ノード名",
      "score": -9～+9の整数
    }},
    ...
  ]
}}

# 重要
- 全{len(pairs)}件について必ず評価してください
- 疎行列の原則: 強い影響がない場合はscore=0
- スコアのみ出力（理由は不要、高速化のため）
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response_text = self._call_with_retry(messages)
            
            # JSON抽出
            import re
            json_match = re.search(r'\{[\s\S]*"evaluations"[\s\S]*\}', response_text)
            if json_match:
                response_json = json.loads(json_match.group(0))
            else:
                response_json = json.loads(response_text)
            
            evaluations = response_json.get("evaluations", [])
            
            # 結果を標準化
            results = []
            for eval_item in evaluations:
                results.append({
                    "from_node": eval_item.get("from_node", ""),
                    "to_node": eval_item.get("to_node", ""),
                    "score": int(eval_item.get("score", 0)),
                    "reason": "",  # 高速化のため理由は保存しない
                    "auto_assigned": False
                })
            
            return results
        
        except json.JSONDecodeError as e:
            # JSONパースエラー時のフォールバック: 全て0とする
            print(f"⚠️ JSONパースエラー: {str(e)}")
            return [
                {
                    "from_node": pair["from_node"],
                    "to_node": pair["to_node"],
                    "score": 0,
                    "reason": "",
                    "auto_assigned": True
                }
                for pair in pairs
            ]
        except Exception as e:
            # その他のエラー
            print(f"⚠️ LLM評価エラー: {str(e)}")
            return [
                {
                    "from_node": pair["from_node"],
                    "to_node": pair["to_node"],
                    "score": 0,
                    "reason": "",
                    "auto_assigned": True
                }
                for pair in pairs
            ]
    
    def zigzagging_inference_for_distant_pairs(
        self,
        distant_pairs: List[Dict[str, Any]],
        idef0_nodes: Dict[str, Dict[str, Any]],
        process_name: str,
        max_pairs_per_batch: int = 30
    ) -> List[Dict[str, Any]]:
        """
        離れた工程間の論理的な依存関係をZigzagging推論で探索
        
        論理ルールベースで「距離が遠いから0」と判定されたペアの中から、
        Zigzagging思考プロセス（How推論）により論理的な依存関係を発見する。
        
        Args:
            distant_pairs: デフォルト0と判定されたペアリスト
            idef0_nodes: IDEF0ノードデータ（全カテゴリ）
            process_name: プロセス名
            max_pairs_per_batch: 1回のLLM呼び出しで処理するペア数上限
        
        Returns:
            論理的な依存関係が見つかったペアの評価結果リスト
        """
        if not distant_pairs:
            return []
        
        # ペアを適切なサイズにバッチ分割
        batches = [
            distant_pairs[i:i + max_pairs_per_batch]
            for i in range(0, len(distant_pairs), max_pairs_per_batch)
        ]
        
        all_results = []
        
        for batch in batches:
            # システムプロンプト構築
            system_prompt = f"""あなたは公理的設計とZigzagging手法の専門家です。

# タスク: 離れた工程間の論理的依存関係の探索

**プロセス名**: {process_name}

## Zigzagging思考プロセスの適用

離れた工程間のペアについて、**論理的な依存関係（How関係）**が存在するかを判断してください。

### 評価の問いかけ

**重要**: 以下の問いかけに対して、**明確な答え**がある場合のみ、非ゼロ評価してください。

#### Output → Mechanism/Input の場合
「離れた工程Aの性能Xを達成するために、工程Bの特性Yは【どのように(How)】貢献する可能性がありますか？」

#### Mechanism/Input → Output の場合
「工程Aの特性Xを改善すると、離れた工程Bの性能Yは【どのように(How)】向上する可能性がありますか？」

### スコアリング基準（厳格な適用）

#### ✅ 論理的なHow関係が明確 → 非ゼロ評価
- **±4～±6**: 離れた工程間でも明確な因果関係が説明できる
- **±1～±3**: 離れた工程間だが、論理的な影響経路が特定できる

#### ❌ How関係が不明確 → 0評価（デフォルトのまま）
- 論理的な因果関係を説明できない
- 「おそらく影響する」「間接的に関係する」などの推測のみ
- **重要**: 疑わしい場合は0としてください（疎行列の原則）

### 疎行列の厳守

**原則**: 「疎で階層的」な構造を維持してください。

- 間接的な関係は除外
- 離れた工程間でも、**直接的で明確な**因果関係のみを抽出
"""
            
            # ユーザープロンプト構築
            pair_descriptions = []
            for idx, pair in enumerate(batch, 1):
                from_node = pair["from_node"]
                to_node = pair["to_node"]
                from_cat = pair.get("from_category", "")
                to_cat = pair.get("to_category", "")
                phase = pair.get("phase_name", "")
                
                pair_descriptions.append(
                    f"{idx}. 【{phase}】\n"
                    f"   評価元: {from_node} (カテゴリ: {from_cat})\n"
                    f"   評価先: {to_node} (カテゴリ: {to_cat})\n"
                    f"   工程間距離: {pair.get('category_distance', '不明')}"
                )
            
            user_prompt = f"""# 評価ペアリスト（全{len(batch)}件）

{chr(10).join(pair_descriptions)}

# 出力形式（必ずJSON形式で）

{{
  "evaluations": [
    {{
      "from_node": "ノード名",
      "to_node": "ノード名",
      "score": -6～+6の整数（0を含む）
    }},
    ...
  ]
}}

# 重要
- 全{len(batch)}件について必ず評価
- 論理的なHow関係が明確な場合のみ非ゼロ
- 疑わしい場合は0（疎行列の原則）
- スコアのみ出力（理由は不要、高速化のため）
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            try:
                response_text = self._call_with_retry(messages)
                
                # JSON抽出
                import re
                json_match = re.search(r'\{[\s\S]*"evaluations"[\s\S]*\}', response_text)
                if json_match:
                    response_json = json.loads(json_match.group(0))
                else:
                    response_json = json.loads(response_text)
                
                evaluations = response_json.get("evaluations", [])
                
                # 非ゼロのペアのみを結果に追加
                for eval_item in evaluations:
                    score = int(eval_item.get("score", 0))
                    if score != 0:
                        all_results.append({
                            "from_node": eval_item.get("from_node", ""),
                            "to_node": eval_item.get("to_node", ""),
                            "score": score,
                            "reason": "",  # 高速化のため理由は保存しない
                            "zigzagging_inference": True,
                            "auto_assigned": False
                        })
            
            except json.JSONDecodeError as e:
                print(f"Zigzagging推論エラー（JSONパース失敗）: {e}")
                continue
            except Exception as e:
                print(f"Zigzagging推論エラー: {e}")
                continue
        
        return all_results
    
    def evaluate_matrix_with_knowledge(
        self,
        from_category: str,
        to_category: str,
        from_nodes: List[str],
        to_nodes: List[str],
        idef0_from: Dict[str, Any],
        idef0_to: Dict[str, Any],
        process_name: str,
        knowledge: List[Dict[str, Any]],
        distance: int
    ) -> List[List[int]]:
        """
        ナレッジを活用した行列形式の評価生成
        
        Args:
            from_category: 評価元カテゴリ名
            to_category: 評価先カテゴリ名
            from_nodes: 評価元ノードリスト（行）
            to_nodes: 評価先ノードリスト（列）
            idef0_from: 評価元カテゴリのIDEF0データ
            idef0_to: 評価先カテゴリのIDEF0データ
            process_name: プロセス名
            knowledge: 参考評価リスト（前フェーズの非ゼロ評価）
            distance: カテゴリ間距離（0=同一、1=隣接、2+=遠距離）
        
        Returns:
            評価行列（2次元リスト）
            matrix[i][j] = from_nodes[i] → to_nodes[j] のスコア
        """
        n = len(from_nodes)
        m = len(to_nodes)
        
        if distance == 0:
            phase_name = "同一カテゴリ内評価"
            phase_desc = "内部依存関係の評価"
            constraint_desc = "対角線成分（自己影響）は必ず0"
        elif distance == 1:
            phase_name = "隣接カテゴリ間評価"
            phase_desc = f"カテゴリ '{from_category}' → '{to_category}' の影響"
            constraint_desc = "前工程の成果物が次工程に与える影響を評価"
        else:
            phase_name = "遠距離カテゴリ間評価"
            phase_desc = f"カテゴリ '{from_category}' → '{to_category}' の間接的影響"
            constraint_desc = "中間パスを経由する論理的な依存関係を評価"
        
        knowledge_text = self._format_knowledge_for_prompt(knowledge)
        
        system_prompt = f"""プロセス '{process_name}' の{phase_name}を評価してください。

# 評価ルール
1. スコア: ±0, ±1, ±3, ±5, ±7, ±9 のみ使用
2. 正(+): 改善方向、負(-): トレードオフ
3. How推論: 「この要素を改善すると、あの要素はどう変化するか？」
4. 対角線（同じノード間）は必ず0

# 参考評価
{knowledge_text}

# タスク
{n}×{m}の評価行列を生成（各成分は上記スコア）
"""

        row_list = "\n".join(f"{i+1}. {node}" for i, node in enumerate(from_nodes))
        col_list = "\n".join(f"{i+1}. {node}" for i, node in enumerate(to_nodes))
        
        user_prompt = f"""# 行（{n}個）
{row_list}

# 列（{m}個）
{col_list}

# 出力（JSON）
{{"matrix": [[各{m}個の数値を省略なく列挙]×{n}行]}}

例: {{"matrix": [[0,3,0,5,0,0,7,0,0,1,0,0],[7,0,0,3,0,0,0,0,0,0,0,5],...{n}行]}}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response_text = self._call_with_retry(messages)
            
            import re
            json_match = re.search(r'\{[\s\S]*"matrix"[\s\S]*\}', response_text)
            if json_match:
                response_json = json.loads(json_match.group(0))
            else:
                response_json = json.loads(response_text)
            
            matrix = response_json.get("matrix", [])
            
            if len(matrix) != n:
                print(f"⚠️ 行列の行数が不正: 期待{n}、実際{len(matrix)}")
                return [[0] * m for _ in range(n)]
            
            for i, row in enumerate(matrix):
                if len(row) != m:
                    print(f"⚠️ 行列の列数が不正（行{i+1}）: 期待{m}、実際{len(row)}")
                    return [[0] * m for _ in range(n)]
            
            for i in range(n):
                for j in range(m):
                    score = int(matrix[i][j])
                    abs_score = abs(score)
                    
                    if abs_score not in [0, 1, 3, 5, 7, 9]:
                        print(f"⚠️ 不正なスコア値[{i}][{j}]: {score} → 0に補正")
                        matrix[i][j] = 0
                    
                    if distance == 0 and i == j:
                        if score != 0:
                            print(f"⚠️ 対角線成分が非ゼロ[{i}][{j}]: {score} → 0に補正")
                            matrix[i][j] = 0
            
            return matrix
        
        except json.JSONDecodeError as e:
            print(f"⚠️ JSONパースエラー: {str(e)}")
            return [[0] * m for _ in range(n)]
        except Exception as e:
            print(f"⚠️ LLM評価エラー: {str(e)}")
            return [[0] * m for _ in range(n)]
    
    def _format_knowledge_for_prompt(self, knowledge: List[Dict[str, Any]]) -> str:
        """
        ナレッジをLLMプロンプト用にフォーマット
        
        Args:
            knowledge: ナレッジリスト
        
        Returns:
            フォーマット済み文字列
        """
        if not knowledge:
            return "参考評価なし（初回評価）"
        
        lines = []
        for i, item in enumerate(knowledge, 1):
            sign = "+" if item["score"] > 0 else ""
            lines.append(
                f"{i}. {item['from_node']} → {item['to_node']}: {sign}{item['score']} "
                f"（{item.get('source_category', '不明')}内）"
            )
        
        return "\n".join(lines)
