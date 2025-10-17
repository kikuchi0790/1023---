# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリで作業する際のガイダンスを提供します。

## プロジェクト概要

**名称:** Process Insight Modeler (PIM)

**目的:** 専門家の持つ暗黙知を、対話を通じて形式知に変換。プロセスを「機能」と「実体（ノード）」に分解し、ノード間の相互作用を有向行列とネットワークで定量化する。

**使用言語:** 日本語（ドキュメントとUIテキスト）

## 技術スタック

- **UIフレームワーク:** Streamlit
- **プログラミング言語:** Python
- **LLM統合:** OpenAI API (gpt-4o)
- **データ処理:** Pandas, NumPy
- **ネットワーク分析:** NetworkX
- **可視化:** Matplotlib, Seaborn

## 開発ワークフロー

7つの段階的なステップによる実装アプローチ：

1. **ステップ1:** 基本UIとセッション管理
2. **ステップ2:** 機能カテゴリ自動抽出のためのOpenAI API統合
3. **ステップ3:** チャットベースのノード定義（"Zigzagging"）
4. **ステップ4:** 評価用ノードペアの生成
5. **ステップ5:** LLMによるノード関係評価
6. **ステップ6:** 隣接行列生成とヒートマップ可視化
7. **ステップ7:** ネットワーク分析とグラフ可視化

各ステップは前のステップを基に構築され、すべてのコードは単一の`app.py`ファイルに集約されます。

## 主要コマンド

```bash
# 依存関係のインストール
pip install streamlit openai pandas numpy networkx matplotlib seaborn

# アプリケーションの実行
streamlit run app.py
```

## プロジェクトアーキテクチャ

### コアデータ構造

アプリケーションは`st.session_state.project_data`で状態を管理：
- `process_name`: 分析対象の生産プロセス名
- `process_description`: プロセスの詳細説明
- `functional_categories`: 機能カテゴリのリスト（品質、コスト、時間、安全性など）
- `nodes`: プロセスノードのリスト（作業工程、道具、材料、スキル）
- `evaluations`: スコアと理由を含むノードペア評価のリスト
- `adjacency_matrix`: ノード関係を表すNumPy配列

### 主要な設計パターン

1. **セッションステート管理:** セッション期間中、すべてのデータが`st.session_state`に永続化
2. **段階的開発:** 各ステップが既存機能を壊さずに新機能を追加
3. **LLMプロンプティング:** 構造化されたシステムプロンプトがOpenAI APIを一貫性のある解析可能な出力に誘導
4. **インタラクティブワークフロー:** 各段階で明確なUIセクションを持つユーザーガイド型プロセス

### 重要な実装上の注意点

- **APIキー管理:** OpenAI APIキーはStreamlitシークレット（`st.secrets["OPENAI_API_KEY"]`）に保存
- **エラー処理:** すべてのAPI呼び出しはtry-exceptブロックでラップし、ユーザーフレンドリーなエラーメッセージを表示
- **データ検証:** LLMからのJSONレスポンスは慎重に解析し、フォールバックエラー処理を実装
- **UIレイアウト:** サイドバーに入力/制御、メインエリアに表示/インタラクション

## LLM統合ガイドライン

### システムプロンプト

異なるタスクのための慎重に作成されたシステムプロンプト：
- **カテゴリ抽出:** 5～8個の機能カテゴリを抽出する生産コンサルタントのペルソナ
- **Zigzagging対話:** プロセスをノードに反復的に分解する促進
- **関係評価:** -9から+9のスケールで理由付きでノード間相互作用を定量化

### 出力フォーマット

構造化データにはLLMから常にJSON形式を要求：
- カテゴリ: `["カテゴリ1", "カテゴリ2", ...]`
- 評価: `{"score": X, "reason": "..."}`

## テストアプローチ

StreamlitアプリケーションとOpenAI API統合のため、手動テストを推奨：
1. 各ステップでUIコンポーネントが正しくレンダリングされることを確認
2. インタラクション間でセッションステートの永続性をテスト
3. APIレスポンスとエラー処理を検証
4. 可視化出力（ヒートマップ、ネットワークグラフ）を確認

## 進捗管理

### 進捗管理ファイル

プロジェクトの進捗を体系的に管理するための以下のファイルを使用：

- **`ROADMAP.md`**: 8週間の開発ロードマップとマイルストーン定義
- **`PROGRESS.json`**: 構造化された進捗データ（自動更新用）
- **`DAILY_LOG.md`**: 日次作業記録と振り返り
- **`CHECKLIST.md`**: 各フェーズの品質保証チェックリスト
- **`progress_tracker.py`**: 進捗追跡・レポート生成の自動化スクリプト

### 進捗管理ワークフロー

#### 1. 作業開始時
- `DAILY_LOG.md`に当日の作業計画を記録
- `progress_tracker.py`でタスクステータスを"in_progress"に更新：
  ```bash
  python progress_tracker.py --update-task p1_t1 --status in_progress
  ```

#### 2. 機能実装完了時
- `CHECKLIST.md`の該当項目をチェック
- テストを実行し、カバレッジを確認
- タスクを完了として更新：
  ```bash
  python progress_tracker.py --update-task p1_t1 --status completed --hours 5.5
  ```

#### 3. 日次終了時
- `DAILY_LOG.md`に完了タスクと課題を記録
- 進捗レポートを生成：
  ```bash
  python progress_tracker.py --report --format markdown
  ```

#### 4. 週次レビュー
- 週間サマリーを`DAILY_LOG.md`に追加
- 品質メトリクスをチェック：
  ```bash
  python progress_tracker.py --quality-check
  ```
- `ROADMAP.md`のマイルストーンを確認・更新

### 自動化コマンド

```bash
# 進捗レポート生成（テキスト形式）
python progress_tracker.py --report

# 進捗レポート生成（Markdown形式）
python progress_tracker.py --report --format markdown

# 進捗レポート生成（JSON形式）
python progress_tracker.py --report --format json

# タスクステータス更新
python progress_tracker.py --update-task <TASK_ID> --status <STATUS> [--hours <HOURS>]

# 品質メトリクスチェック
python progress_tracker.py --quality-check

# 完了予定日の推定
python progress_tracker.py --estimate
```

### 品質基準

各フェーズ完了時に以下の品質基準を満たすこと：

| 指標 | 目標値 |
|------|--------|
| テストカバレッジ | 90%以上 |
| Pylintスコア | 9.0以上 |
| 型チェック（mypy） | エラー0件 |
| コードフォーマット | Black適用済み |
| ドキュメント | 全関数にDocstring |

### タスクID体系

タスクIDは以下の形式で管理：
- `p1_t1`: フェーズ1のタスク1
- `p2_t3`: フェーズ2のタスク3

詳細は`PROGRESS.json`を参照。

### 重要な注意事項

1. **毎日の更新**: `DAILY_LOG.md`は毎日更新し、作業の記録を残す
2. **リアルタイム更新**: タスク開始・完了時は即座に`progress_tracker.py`で更新
3. **品質チェック**: 各タスク完了前に`CHECKLIST.md`の項目を確認
4. **週次レビュー**: 毎週金曜日に進捗確認と次週計画の調整