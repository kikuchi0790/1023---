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
from core.data_models import FunctionalCategory, CategoryGenerationOptions


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
        temperature: float = settings.OPENAI_TEMPERATURE,
        max_retries: int = settings.OPENAI_MAX_RETRIES,
    ) -> str:
        """
        リトライ機能付きAPI呼び出し

        Args:
            messages: メッセージリスト
            temperature: 温度パラメータ
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
                    temperature=temperature,
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

        response_text = self._call_with_retry(messages, temperature=0.3)

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
            "quality": """
【品質重視のガイドライン】
- 製品・サービスの品質を保証する要素
- 検査・測定・管理のプロセス
- 品質基準の達成度
- 不良率・歩留まりに影響する要因""",
            "cost": """
【コスト重視のガイドライン】
- 製造コスト・運用コストに影響する要素
- 資源の効率的利用
- ムダ・ロスの削減
- 原価に直結する要因""",
            "time": """
【時間重視のガイドライン】
- リードタイム・サイクルタイムに影響する要素
- 作業速度・処理能力
- 待ち時間・段取り時間
- スループットに影響する要因""",
            "safety": """
【安全性重視のガイドライン】
- 作業者の安全を守る要素
- 危険作業・リスク管理
- 環境への配慮
- 安全基準の遵守""",
            "flexibility": """
【柔軟性重視のガイドライン】
- 変化への対応力
- 多品種対応・カスタマイゼーション
- 拡張性・適応性
- 変動への耐性""",
            "balanced": """
【バランス型のガイドライン】
- すべての観点を均等に考慮
- QCD+S（品質・コスト・納期・安全性）のバランス
- プロセス全体の最適化
- 総合的な評価軸"""
        }

        guideline = focus_guidelines.get(options.focus, focus_guidelines["balanced"])

        system_prompt = f"""あなたは生産技術に20年以上従事するベテランのコンサルタントです。
生産プロセスの体系的な分析と評価基準の策定に精通しています。

【分析するプロセス】
プロセス名: {process_name}
プロセス概要: {process_description}

【分析方針】
{focus_desc}

{guideline}

【分析手順（Chain-of-Thought）】
ステップ1: プロセスの本質理解
- このプロセスの主要な目的は何か？
- 主要な成果物は何か？
- 誰がステークホルダーか？
- 主要な制約条件は何か？

ステップ2: 評価軸の特定
以下の視点から重要な評価軸を考えてください：
- インプット視点：何が必要か（材料、情報、リソース）
- プロセス視点：どのように変換されるか（作業、手順、技術）
- アウトプット視点：何が生まれるか（製品、品質、データ）
- 制約視点：何が制限要因か（時間、コスト、能力）

ステップ3: MECE原則に基づくカテゴリ化
- Mutually Exclusive（相互排他的）：重複のないカテゴリ
- Collectively Exhaustive（全体網羅的）：プロセス全体をカバー
- {count_min}〜{count_max}個のカテゴリに整理

ステップ4: カテゴリの詳細化
各カテゴリについて：
- 明確で具体的な名称
- カテゴリが何を評価するかの説明（50-100文字）
- 重要度（1-5、5が最重要）
- 具体的な評価項目の例（2-3個）

【出力形式（必須）】
以下のJSON形式で出力してください。他の説明は一切不要です：
```json
[
  {{
    "name": "カテゴリ名",
    "description": "このカテゴリが評価する内容の説明",
    "importance": 5,
    "perspective": "{options.focus}",
    "examples": ["具体例1", "具体例2", "具体例3"]
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

        response_text = self._call_with_retry(messages, temperature=0.5)

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
                    importance=cat_data.get("importance", 3),
                    perspective=options.focus,
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

        response_text = self._call_with_retry(messages, temperature=0.5)

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

        response_text = self._call_with_retry(messages, temperature=0.3)

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

    def chat_zigzagging(
        self,
        process_name: str,
        categories: List[str],
        chat_history: List[Dict[str, str]],
        user_message: str,
    ) -> str:
        """
        Zigzagging対話を実行

        Args:
            process_name: プロセス名
            categories: 機能カテゴリのリスト
            chat_history: これまでのチャット履歴
            user_message: ユーザーの新しいメッセージ

        Returns:
            アシスタントの応答

        Raises:
            OpenAIError: API呼び出しエラー
        """
        system_prompt = f"""あなたは生産技術コンサルタントであり、ユーザーとの対話を通じて生産プロセスの「ノード」を定義する専門家です。あなたの役割は「Zigzagging」と呼ばれるプロセスを主導することです。

【分析対象プロセス】
{process_name}

【機能カテゴリ】
{', '.join(categories)}

【Zigzaggingの手法】
1. **機能から実体へ:** ユーザーに質問し、機能カテゴリを実現するための具体的な実体（作業工程、道具、材料、スキルなど）を洗い出させます。
2. **実体から機能へ:** ユーザーが挙げた実体について、さらに深掘りする質問をします。「その作業はさらにどのような細かい工程に分けられますか？」「その道具を使う目的は何ですか？」のように問いかけ、実体を下位の機能やノードに分解させます。
3. **ノードの提案:** ユーザーの回答が曖昧な場合、具体的なノードの候補を提案してください。
4. **構造化:** 対話の最後に、洗い出されたノードをリスト形式で要約して提示し、ユーザーに確認を求めてください。出力するノード名は、簡潔で一意な名詞句にしてください。

常に対話の主導権を握り、プロセス全体の解像度が上がるようにユーザーを導いてください。"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})

        response = self._call_with_retry(messages, temperature=0.7)

        return response

    def generate_initial_message(
        self, process_name: str, categories: List[str]
    ) -> str:
        """
        Zigzagging対話の初期メッセージを生成

        Args:
            process_name: プロセス名
            categories: 機能カテゴリのリスト

        Returns:
            初期メッセージ
        """
        if not categories:
            return "機能カテゴリを先に定義してください。"

        first_category = categories[0]

        message = f"""こんにちは！生産技術コンサルタントです。

「{process_name}」のプロセス分析をお手伝いします。

定義された{len(categories)}個の機能カテゴリに基づき、プロセスの具体的なノード（作業工程、道具、材料、スキルなど）を洗い出していきましょう。

まずは「{first_category}」について、どのような具体的な要素がありますか？"""

        return message
