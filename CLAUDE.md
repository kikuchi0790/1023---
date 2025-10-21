# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリで作業する際のガイダンスを提供します。

## プロジェクト概要

**名称:** Process Insight Modeler (PIM)

**目的:** 専門家の持つ暗黙知を、対話を通じて形式知に変換。プロセスを「機能」と「実体（ノード）」に分解し、ノード間の相互作用を有向行列とネットワークで定量化し、DSM（設計構造マトリクス）として最適化する。

**核心思想:** **反復的な知識精緻化プロセス**
- 一度作って終わりではなく、分析→フィードバック→改善のループを支援
- 分析結果（ヒートマップ、PageRank、DSM最適化）から粒度調整の必要性を発見
- Zigzagging手法でノードの粒度を段階的に細分化
- 細分化後に再評価・再分析し、知識の解像度を高める

**使用言語:** 日本語（ドキュメントとUIテキスト）

## 技術スタック

### バックエンド
- **UIフレームワーク:** Streamlit（タブベースUI）
- **プログラミング言語:** Python 3.10+
- **LLM統合:** OpenAI API (gpt-4o)
- **データ処理:** Pandas, NumPy
- **ネットワーク分析:** NetworkX
- **可視化:** Matplotlib, Seaborn
- **最適化:** NSGA-II（多目的遺伝的アルゴリズム）

### フロントエンド（カスタムStreamlitコンポーネント）
- **3D可視化:** NetworkMaps (Three.js)
- **2D可視化:** Cytoscape.js
- **開発言語:** React, TypeScript
- **ビルドツール:** Webpack

## アプリケーション構造

### 9ステップのタブ形式UI（反復的プロセス）

アプリケーションは`app_tabs.py`で実装され、9つのタブで構成されます：

1. **ステップ1: プロセス定義** - プロセス名と概要の入力
2. **ステップ2: 機能カテゴリ** - LLMによる自動抽出（多様性生成オプション）
3. **ステップ3: ノード定義** - IDEF0形式（AI主導対話 / 多様性生成 / **Zigzagging粒度調整**）
4. **ステップ4: ノード影響評価** - Zigzagging手法による-9～+9スケール評価
5. **ステップ5: 行列分析** - 隣接行列生成とヒートマップ可視化
6. **ステップ6: ネットワーク可視化** - 3D/2D可視化
7. **ステップ7: ネットワーク分析** - PageRank、中心性指標
8. **ステップ8: DSM最適化** - NSGA-IIによる多目的最適化 + **パーティショニング**
9. **ステップ9: 高度な分析** - 博士課程レベルの7つの分析手法（Shapley Value, Transfer Entropy, etc.）

**反復的フィードバックループ:**
- ステップ5-8の分析結果から「粒度が粗い」ノードを発見
- ステップ3の「Zigzagging粒度調整」モードで細分化
- ステップ4で再評価 → ステップ5-8で再分析
- このループを繰り返し、知識の解像度を段階的に高める

### ファイル構成

```
pim/
├── app_tabs.py                    # メインアプリケーション（9タブUI）
├── app.py.backup                  # 旧単一ファイル版（バックアップ）
├── config/
│   └── settings.py                # アプリケーション設定
├── core/
│   ├── session_manager.py         # セッションステート管理
│   ├── llm_client.py              # OpenAI API統合
│   └── data_models.py             # Pydanticデータモデル
├── utils/
│   ├── networkmaps_bridge.py      # PIM → NetworkMaps変換
│   ├── cytoscape_bridge.py        # PIM → Cytoscape変換
│   ├── idef0_classifier.py        # IDEF0ノード分類とZigzaggingペア生成
│   ├── evaluation_filter.py       # 論理ルールベース評価フィルタリング
│   ├── dsm_optimizer.py           # DSM最適化（NSGA-II）
│   ├── dsm_partitioning.py        # DSMパーティショニング（モジュール化・デザインシーケンス）
│   ├── shapley_analysis.py        # Shapley Value分析（協力貢献度 + 連携安定性）
│   ├── information_theory_analysis.py  # Transfer Entropy（情報フロー分析）
│   ├── statistical_testing.py     # Bootstrap統計検定
│   ├── bayesian_analysis.py       # Bayesian Inference（不確実性定量化）
│   ├── causal_inference.py        # Pearl's Causal Inference（因果推論）
│   ├── graph_embedding.py         # Graph Embedding + Community Detection
│   ├── fisher_information.py      # Fisher Information Matrix（感度分析）
│   └── analytics_export.py        # 高度な分析結果のエクスポート（Excel/JSON）
├── components/
│   ├── networkmaps_viewer/        # 3D可視化コンポーネント
│   │   ├── __init__.py
│   │   ├── component.py
│   │   └── frontend/              # React/TypeScript
│   └── cytoscape_viewer/          # 2D可視化コンポーネント
│       ├── __init__.py
│       ├── component.py
│       └── frontend/              # React/TypeScript
├── step3.md, step4.md, ...        # 各ステップの実装仕様書
├── CLAUDE.md                      # このファイル
├── ROADMAP.md                     # 開発ロードマップ
└── PROGRESS.json                  # 進捗管理データ
```

## 主要コマンド

```bash
# 依存関係のインストール
pip install streamlit openai pandas numpy networkx matplotlib seaborn openpyxl

# メインアプリケーションの実行
NETWORKMAPS_RELEASE=true streamlit run app_tabs.py

# 旧バージョンの実行（参照用）
streamlit run app.py

# フロントエンドコンポーネントのビルド（開発時）
cd components/networkmaps_viewer/frontend && npm run build
cd components/cytoscape_viewer/frontend && npm run build
```

## コアデータ構造

### セッションステート管理

`SessionManager`クラスが`st.session_state`を抽象化：

```python
# プロセス情報
SessionManager.get_process_name() -> str
SessionManager.get_process_description() -> str

# 機能カテゴリ
SessionManager.get_functional_categories() -> List[str]

# IDEF0ノード（カテゴリごと）
SessionManager.get_all_idef0_nodes() -> Dict[str, Dict]
# 形式: {category: {"function": str, "inputs": List[str], 
#                   "mechanisms": List[str], "outputs": List[str]}}

# 全ノードリスト
SessionManager.get_nodes() -> List[str]

# 対話履歴
SessionManager.get_messages() -> List[Dict]

# 隣接行列
st.session_state.adjacency_matrix -> np.ndarray
```

### IDEF0形式

各機能カテゴリは以下の構造を持ちます：

- **Function**: 機能名（カテゴリ名と同じ）
- **Input**: 材料、情報（リスト）
- **Mechanism**: 手段、道具、スキル（リスト）
- **Output**: 成果物（リスト）

## LLM統合

### 主要な生成タスク

1. **機能カテゴリ抽出** (`generate_functional_categories`)
   - 入力: プロセス名、概要、オプション（分析視点、粒度）
   - 出力: `CategorySet`（Pydanticモデル）

2. **多様性生成** (`generate_categories_with_diversity`)
   - Verbalized Samplingを使用
   - 5つの異なる思考モード（分析哲学）から生成
   - 各代替案に確率スコア付与

3. **AI主導対話** (`generate_ai_discussion`)
   - ファシリテーターとエキスパートの2つのペルソナ
   - ソクラテス式対話でノードを深掘り

4. **IDEF0ノード抽出** (`extract_all_idef0_nodes_from_chat`)
   - 対話履歴から全カテゴリのIDEF0構造を抽出

5. **Zigzagging粒度調整** (`refine_idef0_with_zigzagging`) ⭐ NEW
   - 既存のIDEF0ノードを段階的に細分化
   - Output → Sub-Output、Mechanism → Sub-Mechanism、Input → Sub-Input
   - 細分化の深さ（1: 軽度, 2: 中程度, 3: 詳細）を指定可能
   - 反復的な知識精緻化プロセスの核心機能

6. **ノード影響評価（IDEF0コンテキスト付き）** (`evaluate_node_pair_with_idef0_context`)
   - ノードペアごとに-9～+9スケールで評価
   - ノードタイプ（Output/Mechanism/Input）を考慮
   - Zigzagging手法による評価フェーズ（性能→特性、特性→性能、性能間、特性間）
   - 疎で階層的な行列を生成（直接的で強い影響のみ）

7. **カテゴリバッチ評価** (`evaluate_category_batch`) ⭐ Zigzagging統合版
   - 1カテゴリ内の全評価ペアを一括評価
   - 全体コンテキスト（inputs, mechanisms, outputs）を把握
   - **How推論ベースの評価プロンプト**:
     - フェーズごとの明確な問いかけ
     - 「どのような経路・メカニズムで影響するか？」
     - 疎行列の厳守原則を強調（直接的で強い影響のみ）
   - 効率化: 複数ペアを1回のLLM呼び出しで処理

8. **Zigzagging推論 - 離れたペアの論理的依存関係探索** (`zigzagging_inference_for_distant_pairs`) ⭐ NEW
   - 距離2+の離れた工程間ペアに対して論理的な依存関係を探索
   - バッチ処理（30ペア/回）で効率化
   - How推論ベース: 「どのような経路・メカニズムで影響するか？」
   - フェーズごとの明確な問いかけ:
     - Output → Mechanism/Input
     - Mechanism/Input → Output
     - Output → Output（性能間の複合影響）
     - Mechanism/Input間（リソース競合、制約共有）
   - 低温度設定（temperature=0.2）で精度重視
   - 疎行列の厳守原則を強調
   - 非ゼロ結果のみ返却（疎行列維持）
   - タブ4でオプション機能として実装（チェックボックス有効化）

9. **DSMパラメータ評価** (`evaluate_dsm_parameters`) ⭐ NEW
   - タブ8のDSM最適化パラメータをLLMが自動評価
   - Cost（1-5）、Range（0.1-2.0）、Importance（1-5）、Structure（グループ名）
   - 20年以上の生産技術コンサルタントのペルソナ
   - プロセスコンテキストを考慮した一貫性のある評価

### システムプロンプト設計原則

- **ペルソナ設定**: 生産技術コンサルタント、20年以上の経験
- **出力形式の厳守**: JSON形式で構造化データを要求
- **思考プロセスの明示**: Chain-of-Thoughtで推論過程を記述
- **エラー処理**: JSONパースエラーに対するリトライとフォールバック

## ネットワーク可視化

### 3D可視化（NetworkMaps）

**変換**: `utils/networkmaps_bridge.py`

```python
def convert_pim_to_networkmaps(
    nodes: List[str],
    adjacency_matrix: np.ndarray,
    categories: List[str],
    idef0_data: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    # PIMデータをNetworkMaps JSON形式に変換
    # IDEF0構造に基づき階層的配置
    # X軸: カテゴリ（時系列）
    # Y軸: 層（Function > Output > Mechanism > Input）
    # Z軸: 奥行き（層間の視覚的分離）
```

**特徴**:
- Three.jsベースの3D空間
- IDEF0層ごとに背景プレーン
- ノード色: Output=緑、Mechanism=青、Input=オレンジ
- エッジ色/太さ: スコアに基づく

### 2D可視化（Cytoscape）

**変換**: `utils/cytoscape_bridge.py`

```python
def convert_pim_to_cytoscape(
    nodes: List[str],
    adjacency_matrix: np.ndarray,
    categories: List[str],
    idef0_data: Dict[str, Dict[str, Any]],
    threshold: float = 2.0,
    use_hierarchical_layout: bool = False
) -> Dict[str, Any]:
    # 閾値フィルタリング
    # 階層的座標計算（オプション）
```

**レイアウトオプション**:
- **hierarchical**: 3D構造準拠（プリセット座標）
- **cose**: 力学モデル
- **breadthfirst**: 階層的（自動）
- **circle**: 円形
- **grid**: グリッド

## DSM最適化（未実装）

### STEP-1: 設計パラメータ選択

**目的関数**:
1. コスト最小化: 同一構造内の最大コストの合計
2. 自由度最大化: 各FRの調整能力比の総和

### STEP-2: 依存関係方向決定

**目的関数**:
1. 調整困難度最小化: αパターン + γパターン
2. 競合困難度最小化: 列への複数影響の相乗効果
3. ループ困難度最小化: 閉路の累積影響

**アルゴリズム**: NSGA-II（Non-dominated Sorting Genetic Algorithm II）

## データエクスポート/インポート

### Excel処理

**使用ライブラリ**: Pandas + openpyxl

```python
import pandas as pd

# エクスポート
df = pd.DataFrame(data)
df.to_excel('export.xlsx', index=False)

# インポート
df = pd.read_excel('import.xlsx')
```

**注意**: Claude Skills（claude.aiのWeb UI機能）はClaude Code（CLI）では使用不可。PythonライブラリでExcel処理を実装。

## 開発ワークフロー

### タブごとの段階的実装

各タブは前のタブの完了を前提条件としてチェック：

```python
def tab2_functional_categories():
    if not (process_name and process_description):
        st.warning("⚠️ 先にタブ1でプロセスを定義してください")
        return
    # ...
```

### 前提条件チェックフロー

- タブ1 → タブ2: プロセス名・概要が入力済み
- タブ2 → タブ3: カテゴリが1つ以上定義済み
- タブ3 → タブ4: ノードが2つ以上定義済み
- タブ4 → タブ5: 評価データが存在
- タブ5 → タブ6,7,8: 隣接行列が生成済み

### テストアプローチ

1. **UIテスト**: 各タブで正常にレンダリングされるか確認
2. **状態管理テスト**: タブ切替時にセッションステートが保持されるか
3. **LLM統合テスト**: API呼び出しとレスポンス解析の動作確認
4. **可視化テスト**: 3D/2D可視化が正しく表示されるか
5. **エラーハンドリングテスト**: 異常系での挙動確認

## 重要な実装上の注意点

### APIキー管理

```python
# .streamlit/secrets.tomlに設定
OPENAI_API_KEY = "sk-..."

# コード内でアクセス
import streamlit as st
api_key = st.secrets["OPENAI_API_KEY"]
```

### エラー処理

```python
try:
    llm_client = LLMClient()
    result = llm_client.some_method()
except OpenAIError as e:
    st.error(f"❌ OpenAI APIエラー: {str(e)}")
except json.JSONDecodeError as e:
    st.error(f"❌ JSONパースエラー: 行 {e.lineno}, 列 {e.colno}")
except Exception as e:
    st.error(f"❌ エラー: {str(e)}")
```

### データ検証

LLMからのJSON応答は必ずPydanticモデルで検証：

```python
from core.data_models import CategorySet

response_json = json.loads(response_text)
category_set = CategorySet(**response_json)  # バリデーション
```

### UIレイアウト

- **サイドバー**: プロジェクト情報サマリー（読み取り専用）
- **タブ**: 各ステップの完全なUI
- **進捗表示**: 各タブで次ステップへの誘導

## カスタムStreamlitコンポーネント開発

### ディレクトリ構成

```
components/example_viewer/
├── __init__.py                 # Pythonエントリーポイント
├── component.py                # Streamlit統合
└── frontend/
    ├── package.json            # npm依存関係
    ├── tsconfig.json           # TypeScript設定
    ├── webpack.config.js       # ビルド設定
    ├── public/index.html       # HTMLテンプレート
    └── src/
        ├── types.ts            # 型定義
        ├── Graph.tsx           # メインコンポーネント
        └── index.tsx           # Streamlit接続
```

### ビルドコマンド

```bash
cd components/example_viewer/frontend
npm install
npm run build
# → build/bundle.js生成
```

### Python側の使用

```python
from components.example_viewer import example_viewer

result = example_viewer(
    graph_data=data,
    layout="cose",
    height=700,
    key="unique_key"
)
```

## 進捗管理

### 進捗管理ファイル

- **`ROADMAP.md`**: 8週間の開発ロードマップ
- **`PROGRESS.json`**: 構造化された進捗データ
- **`DAILY_LOG.md`**: 日次作業記録
- **`CHECKLIST.md`**: 品質保証チェックリスト
- **`progress_tracker.py`**: 進捗追跡自動化

### 主要コマンド

```bash
# 進捗レポート生成
python progress_tracker.py --report --format markdown

# タスクステータス更新
python progress_tracker.py --update-task p1_t1 --status completed --hours 5.5

# 品質メトリクスチェック
python progress_tracker.py --quality-check
```

### 品質基準

| 指標 | 目標値 |
|------|--------|
| テストカバレッジ | 90%以上 |
| Pylintスコア | 9.0以上 |
| 型チェック（mypy） | エラー0件 |
| コードフォーマット | Black適用済み |
| ドキュメント | 全関数にDocstring |

## トラブルシューティング

### LLM生成エラー

**症状**: "視点の生成に失敗しました"

**原因**:
1. JSONフォーマット不正
2. プロセス概要が不明瞭
3. カテゴリ数過多（推奨: 5-8個）

**対処**:
- `debug_logs/`ディレクトリのログファイル確認
- プロセス概要をより具体的に記述
- temperature値調整（デフォルト: 0.7）

### カスタムコンポーネントビルドエラー

**症状**: TypeScriptコンパイルエラー

**対処**:
```bash
cd components/xxx_viewer/frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### 可視化が表示されない

**確認項目**:
1. `NETWORKMAPS_RELEASE=true`環境変数が設定されているか
2. フロントエンドがビルド済みか (`build/`ディレクトリ存在)
3. 隣接行列データが存在するか
4. ブラウザのコンソールエラーを確認

## 実装済み機能

### フェーズ1: 基本ワークフロー（完了）

- [x] タブ1: プロセス定義
- [x] タブ2: 機能カテゴリ抽出（多様性生成含む）
- [x] タブ3: IDEF0ノード定義（AI主導対話、多様性生成、**Zigzagging粒度調整**）
- [x] タブ4: **論理ルールベース評価システム**（カテゴリ間距離フィルタリング + LLMバッチ評価 + Zigzagging推論）
  - 3ステップワークフロー: ペア生成→フィルタリング→バッチ評価→結果確認
  - カテゴリ間距離による事前フィルタリング（80-90%削減）
  - 同一カテゴリ内ペアのLLMバッチ評価
  - 疎行列の厳格な適用
  - **オプション: Zigzagging推論**（ステップ2.5）
    - 離れた工程間の論理的依存関係探索
    - バッチ処理（30ペア/回）、低温度設定（0.2）
    - 非ゼロ結果のみ返却（疎行列維持）
- [x] タブ5: 行列分析・**ヒートマップ可視化** + **粒度調整提案**
  - matplotlib font設定による日本語文字化け修正
  - ラベル英語化（"Node Influence Heatmap", "To Node", "From Node"）
- [x] タブ6: 3D/2D可視化
- [x] タブ7: NetworkX分析（PageRank、中心性） + **粒度調整提案**
- [x] タブ8: DSM最適化（NSGA-II） + **LLMパラメータ評価** + **クラッシュ対策**
  - STEP-1: 設計パラメータ選択（コスト vs 自由度）
  - STEP-2: 依存関係方向決定（調整困難度 vs 競合困難度 vs ループ困難度）
  - パレートフロント2D/3D可視化（日本語フォント対応）
  - 最適化DSMヒートマップ
  - **LLM自動パラメータ評価**: Cost, Range, Importance, Structureを文脈に応じて自動評価
  - **軽量モード**: デフォルト100個体×50世代（STEP-1）、100個体×30世代（STEP-2）
  - **リアルタイム進捗表示**: プログレスバー、推定残り時間、完了メッセージ
  - **チェックポイント機能**: クラッシュ時の復元対応（`utils/dsm_optimizer.py`）
  - **同期実行**: シンプルで確実な進捗表示

### フェーズ2: 反復的知識精緻化（完了）

- [x] `refine_idef0_with_zigzagging()`: 粒度調整LLMメソッド
- [x] タブ3「Zigzagging粒度調整」モード: 手動で細分化選択
- [x] タブ5: 次数ベースの粒度調整提案（高次数ノード検出）
- [x] タブ7: PageRank/Betweennessベースの粒度調整提案
- [x] タブ3との連携: `selected_refinement_node`セッションステート

### フェーズ3: データ管理・永続化（完了） ✅ NEW

- [x] **サイドバーUI**: エクスポート/インポート機能
  - 折りたたみ式エクスパンダー
  - 形式選択（Excel/JSON/CSV）
  - ダウンロードボタン動的生成
  - ファイルアップローダー
- [x] **Excelエクスポート/インポート** (`utils/data_io.py`)
  - 6シート構成（プロジェクト情報、カテゴリ、IDEF0、行列、評価、ノード）
  - Pandas + openpyxl使用
  - タイムスタンプ付きファイル名
- [x] **JSONエクスポート/インポート**
  - 完全なセッション状態保存
  - バージョン管理（v1.0.0）
  - メタデータ（エクスポート日時）
- [x] **CSVサポート**
  - 隣接行列のみエクスポート/インポート
  - 他ツールとの互換性
- [x] **データ検証とエラーハンドリング**
  - バージョンチェック
  - データ型検証
  - インポート後自動リロード

### Week 5: 高度な分析（タブ9）（完了） ✅

- [x] **7つの博士課程レベル分析手法**
  - Shapley Value（協力貢献度分析）⭐ 推奨
  - Transfer Entropy（情報フロー分析）⭐ 推奨
  - Bootstrap統計検定⭐ 推奨
  - Bayesian Inference（不確実性定量化）
  - Causal Inference（因果推論）
  - Graph Embedding（潜在構造発見）
  - Fisher Information（感度分析）

- [x] **Shapley Value 連携安定性分析** (`utils/shapley_analysis.py:compute_shapley_coalition_stability()`)
  - 上位25%ノード間の連携強度分析
  - 密結合ペアTop10の特定
  - NetworkX spring layoutネットワーク図可視化
  - **UI要素**: 7つの可視化（他の分析手法を上回る）
    1. 解釈文（expander）
    2. メトリクス（4カラム）
    3. 貢献度ランキング（DataFrame上位20）
    4. 貢献度分布（横棒グラフ、上位15）
    5. 累積貢献度（折れ線グラフ、80%ライン）
    6. カテゴリ別平均貢献度（IDEF0連携）
    7. **連携安定性分析**（上位貢献者 + 密結合ペア + ネットワーク図）

- [x] **高度な分析結果エクスポート** (`utils/analytics_export.py`)
  - **Excel形式**: 各分析ごとに複数シート
    - Shapley: 5シート（Values, Cumulative, Categories, TopContributors, DensePairs）
    - Transfer Entropy: 3シート（Matrix, Flows, Comparison）
    - Bootstrap: 2シート（CI, Groups）
    - Bayesian Inference: 2シート（CredibleIntervals, HighUncertainty）
    - Causal Inference: 3シート（InterventionEffects, TopTargets, Confounders）
    - Graph Embedding: 3シート（Communities, Positions2D, Similarity）
    - Fisher Information: 2シート（SensitivityScores, CramerRaoBounds）
  - **JSON形式**: 完全な分析結果の構造化データ
  - サイドバーからワンクリックエクスポート

- [x] **統一UI設計パターン**（全7手法）
  1. パラメータ設定（sliders, selectors）
  2. 実行ボタン + 進捗トラッカー
  3. 結果の解釈（expander）
  4. メトリクス（4カラム）
  5. 可視化（matplotlib charts）
  6. データテーブル（pandas DataFrames）

- [x] **ユーザーガイド** (`docs/USER_GUIDE_TAB9.md`)
  - 各分析の詳細説明（何がわかるか、どう使うか、結果の見方）
  - 7手法の比較表（目的、計算時間、推奨度、主な用途）
  - 推奨ワークフロー（5ステップ）
  - FAQ（よくある質問）

### オプション実装候補（Phase 2, 3）

- [ ] **Shapley Value収束曲線分析**（1.5時間）
  - サンプル数を変えて複数回実行
  - 上位ノードの収束曲線可視化
  - 収束判定メトリクス
  
- [ ] **Shapley Value貢献度分解マトリックス**（2時間）
  - N×N 貢献度マトリックス（ノードi → ノードj への限界貢献）
  - ヒートマップ可視化
  - 上位貢献ペアランキング

## タブ4: 論理ルールベース評価システム（詳細）

### 評価の課題

**従来の問題:**
- 37ノード → 1332ペア（37×36）の全組み合わせ評価
- 1ペアずつ手動評価は非現実的
- UIが進まない

### データ構造に基づく論理ルール

#### ルール1: カテゴリ間距離フィルタリング

```
距離0（同一カテゴリ）: 強い影響の可能性 → LLM評価必須
距離1（隣接カテゴリ）: 中程度の影響 → 評価推奨
距離2+（離れたカテゴリ）: 影響微弱 → デフォルト0
```

#### ルール2: IDEF0論理による構造的制約

**フェーズ1（Output → Mechanism/Input）:**
- 同一カテゴリ内: 強依存（必ず評価）
- 異カテゴリ: 距離ルール適用

**フェーズ2（Mechanism/Input → Output）:**
- 同一カテゴリ内: 強依存
- 次カテゴリのOutput: 中程度（前工程の成果物が次工程の性能に影響）

**フェーズ3（Output ↔ Output）:**
- 同一カテゴリ内: トレードオフの可能性
- 隣接カテゴリ: 前工程の性能が次工程の性能に影響
- 距離2+: ほぼ無関係

**フェーズ4（Mechanism/Input間）:**
- 同一カテゴリ内: リソース競合、制約共有の可能性
- 異カテゴリ: ほぼ無関係

### 3ステップワークフロー（+ オプション: Zigzagging推論）

**ステップ1: 評価ペア生成 + フィルタリング**
- Zigzagging手法で全ペア生成
- 論理ルールで分類:
  - 必須評価（同一カテゴリ内）
  - 推奨評価（隣接カテゴリ）
  - デフォルト0（離れたカテゴリ）
- 削減率表示（通常80-90%削減）

**ステップ2: LLMバッチ評価実行**
- **3フェーズ段階的評価**:
  - フェーズ1: 同一カテゴリ内（距離0）
  - フェーズ2: 隣接カテゴリ間（距離1）
  - フェーズ3: 遠距離（距離2+、オプション）
- **UIの工夫**（2024-01-19修正）:
  - バッチ評価ボタンをexpander外に配置
  - フェーズヘッダーと「全て評価」ボタンを常に表示
  - 個別評価のみexpander内に配置
  - 未完了の最初のフェーズを自動展開
  - ナレッジベース件数を表示（前フェーズの知識継承を可視化）
- カテゴリごとにバッチ処理
- 各バッチでIDEF0構造全体を把握
- プログレスバー表示
- デフォルト0のペアは自動で0を設定

**ステップ2.5（オプション）: Zigzagging推論 - 離れたペアの論理的依存関係探索**
- チェックボックスで有効化（デフォルト: 無効）
- 距離2+の離れた工程間ペアに対して論理的な依存関係を探索
- **メソッド:** `zigzagging_inference_for_distant_pairs()`
- **処理内容:**
  - バッチ処理（30ペア/回）で効率化
  - How推論ベース: 「どのような経路・メカニズムで影響するか？」
  - フェーズごとの明確な問いかけ:
    - Output → Mechanism/Input
    - Mechanism/Input → Output
    - Output → Output（性能間の複合影響）
    - Mechanism/Input間（リソース競合、制約共有）
  - 低温度設定（temperature=0.2）で精度重視
  - 疎行列の厳守原則を強調
  - 非ゼロ結果のみ返却（疎行列維持）
- **使用例:** 
  - 離れた工程間の隠れた依存関係の発見
  - 長期的な品質影響の検出
  - サプライチェーン全体の連鎖反応の分析
- **注意:** LLM呼び出しコストが増加（推奨: 37ノード以下）

**ステップ3: 評価結果確認**
- 非ゼロペア数と疎行列率の表示
- 高スコアペア（|score| ≥ 5）の一覧
- タブ5への誘導

### 効果

**37ノード（7カテゴリ）の例:**
- 全ペア: 1332件
- 必須評価: 約100-200件
- LLM呼び出し: 7回（カテゴリ数分）
- 削減率: **80-90%**

## タブ8: DSM最適化 LLMパラメータ評価（詳細）

### パラメータ評価モード

**3つのモード:**
1. **🤖 LLMによる自動評価（推奨）**: プロセスコンテキストを考慮した自動評価
2. **📊 固定デフォルト値**: Cost=1, Range=1, Importance=1, Structure=カテゴリ名
3. **⚙️ 手動カスタム設定（上級者向け）**: 将来実装予定

### LLM評価パラメータ

**Cost（コスト） - DP（設計パラメータ）のみ:**
- スケール: 1-5
- 評価観点: 調整にかかるコスト（投資、訓練、設備）

**Range（変動範囲） - DP（設計パラメータ）のみ:**
- スケール: 0.1-2.0
- 評価観点: パラメータの調整幅、柔軟性

**Importance（重要度） - FR（機能要求）のみ:**
- スケール: 1-5
- 評価観点: 性能の重要性（安全性、法規制、コア機能）

**Structure（構造グループ） - すべてのノード:**
- 論理的なグループ名
- 同じ部品、工程、システムに属するノードをグループ化

### LLMプロンプト設計

**ペルソナ:**
- 生産技術に20年以上従事するベテランのコンサルタント
- 設計構造マトリクス（DSM）最適化の専門家

**評価プロセス:**
1. プロセス名、概要、全ノードのIDEF0構造を把握
2. 各ノードのタイプ（FR/DP）を考慮
3. パラメータごとに評価基準に基づき数値化
4. JSON形式で評価結果と根拠を出力

**出力形式:**
```json
{
  "parameters": {
    "ノード名": {
      "cost": 3,
      "range": 1.2,
      "importance": 4,
      "structure": "組立工程"
    },
    ...
  },
  "reasoning": "評価の根拠..."
}
```

### UI統合

**DSM設定セクション（タブ8）:**
1. パラメータ設定方法をラジオボタンで選択
2. LLM評価ボタンをクリック
3. 評価結果をDataFrameで表示（ノード名、タイプ、パラメータ値）
4. 評価の根拠をテキスト表示
5. STEP-1実行時に選択されたモードのパラメータを使用

### NSGA-II最適化のパフォーマンス改善（2025-10-21実装）

**課題:**
- STEP-2は大規模データで40分以上かかることがある
- Streamlitサーバーがメモリ不足でクラッシュ（Exit code 137: OOM Killer）
- プログレスバーが更新されず、フリーズしているように見える

**解決策:**

#### 1. 軽量モード（デフォルト有効）
- **STEP-1**: 100個体 × 50世代（従来: 200×100）
- **STEP-2**: 100個体 × 30世代（従来: 200×50）
- チェックボックスで切替可能
- メモリ使用量を50%削減

#### 2. リアルタイム進捗表示
```python
# 進捗コールバック（推定残り時間付き）
def progress_callback(gen: int, pareto_size: int):
    progress_pct = gen / total_gen
    eta_seconds = (elapsed / gen) * (total_gen - gen)
    progress_placeholder.progress(
        progress_pct,
        text=f"世代 {gen}/{total_gen} | 推定残り時間: {eta_min}分{eta_sec}秒"
    )
```

- **初期メッセージ**: ボタン押下直後に「開始しました」を表示
- **スピナー**: 画面上部に回転アイコン
- **プログレスバー**: 毎世代ごとに更新（`save_every=1`）
- **推定残り時間**: 平均世代時間から自動計算
- **完了メッセージ**: 実行時間を分秒で表示

#### 3. チェックポイント機能
- 世代ごとに途中結果を`checkpoints/`に保存
- クラッシュ時に最新チェックポイントから復元可能
- 最適化完了後に自動削除

#### 4. 同期実行（非同期から変更）
- 非同期処理はプログレスバー更新が難しい
- 同期処理でシンプルかつ確実な進捗表示
- `st.spinner()`と`st.progress()`の組み合わせ

#### 5. 日本語フォント設定
```python
# パレートフロント図の文字化け対策
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

**実行時間の目安（37ノード、軽量モード）:**
- STEP-1: 30-60秒
- STEP-2: 20-40秒（ノード数により変動）

**推奨設定:**
- 軽量モードを有効（メモリ不足防止）
- ノード数が50以上の場合は個体数を50に削減
- 長時間処理（10分以上）はサーバーを監視

## 今後の実装予定

### 短期（機能強化）

- [ ] `utils/refinement_analyzer.py`: 分析結果からの粒度調整提案ロジックを共通化
- [ ] タブ8: 困難度指標ベースの粒度調整提案（STEP-2結果から検出）
- [ ] タブ4: 推奨評価（隣接カテゴリ）のオプション実装

### 中期（バージョン管理）

- [x] スナップショット機能（プロジェクト状態の保存・読み込み） ✅ 完了
- [ ] バージョン比較ビュー（ノード数、行列密度の比較）
- [ ] Undo/Redo機能

### データ管理機能 ✅ 完了

- [x] **Excelエクスポート（Pandas + openpyxl）** - `utils/data_io.py:export_to_excel()`
  - 6シート構成: プロジェクト情報、機能カテゴリ、IDEF0ノード、隣接行列、評価詳細、ノードリスト
  - タイムスタンプ付きファイル名
  - サイドバーからワンクリックエクスポート
- [x] **Excelインポート** - `utils/data_io.py:import_from_excel()`
  - 各シートから段階的復元
  - データ型検証とエラーハンドリング
  - インポート後自動リロード
- [x] **プロジェクトセッション保存/読み込み（JSON）** - `utils/data_io.py:export_to_json(), import_from_json()`
  - 完全なセッション状態の保存
  - バージョン管理（v1.0.0）
  - メタデータ（エクスポート日時）
- [x] **CSV形式サポート** - `utils/data_io.py:export_adjacency_matrix_to_csv(), import_adjacency_matrix_from_csv()`
  - 隣接行列のみエクスポート/インポート
  - 他ツールとの互換性

## 反復的知識精緻化プロセスの詳細

### ワークフローの循環構造

```
ステップ1-3: ノード定義（初期粒度）
    ↓
ステップ4: 評価
    ↓
ステップ5-8: 分析・可視化・最適化
    ↓
「粒度が粗い」と判断
    ↓
ステップ3: Zigzagging粒度調整 ← フィードバック
    ↓
ステップ4: 再評価
    ↓
...（ループ）
```

### Zigzagging粒度調整の原理

**Suh (2001) の公理的設計におけるZigzagging:**
- 新規設計: 機能要求（FR）と設計変数（DP）を交互に具体化
- PIMでの応用: **既存設計の知識を階層的に整理**

**手順:**
1. Output（性能）の細分化: 「この成果物の品質を評価する細かい指標は？」
2. Mechanism（手段）の細分化: 「この手段を実行する具体的な作業は？」
3. Input（材料）の細分化: 「この材料はどのような要素で構成される？」

**原則:**
- MECE原則（Mutually Exclusive, Collectively Exhaustive）
- 粒度の一貫性（同じ抽象度レベル）
- 計測可能性（観測・計測可能な要素）

### 分析結果からの粒度調整提案（実装済み）

| 分析タブ | 検出基準 | 提案理由 | ステータス |
|---------|---------|---------|----------|
| タブ5（ヒートマップ） | High degree、平均スコア高 | 多くのノードと関係 → 粒度が粗い | ✅ 実装済み |
| タブ7（PageRank） | PageRank高、Betweenness高 | 重要・ボトルネック → 細分化の価値 | ✅ 実装済み |
| タブ8（DSM） | 調整難易度・コンフリクト度高 | 設計困難 → 粒度調整が必要 | ⚡ 将来実装 |

## 参考資料

- ステップ実装仕様: `step3.md`, `step4.md`, `step5.md`, `step6.md`, `step7.md`
- NSGA-II数式: ユーザー提供のLaTeX文書参照
- IDEF0 + Zigzagging文献: ユーザー提供のLLMによるSAM自動生成フレームワーク
- Streamlitドキュメント: https://docs.streamlit.io/
- NetworkMaps: 既存実装（NetworkMaps/pim統合プロジェクト）
- Cytoscape.js: https://js.cytoscape.org/
- 公理的設計: Suh (2001) - Zigzagging手法の原典
