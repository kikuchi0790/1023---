# Process Insight Modeler (PIM)

生産プロセスの暗黙知を形式知に変換するStreamlitアプリケーション

## 🎯 プロジェクト概要

このプロジェクトは、専門家が持つ生産プロセスに関する暗黙知を、対話を通じて形式知に変換し、ネットワーク分析によって可視化するツールです。

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. OpenAI APIキーの設定

`.streamlit/secrets.toml` ファイルを作成し、APIキーを設定：

```toml
OPENAI_API_KEY = "your-api-key-here"
```

### 3. アプリケーションの起動

```bash
streamlit run app.py
```

ブラウザで http://localhost:8501 が自動的に開きます。

## 📁 プロジェクト構造

```
.
├── app.py                  # メインアプリケーション
├── config/                 # 設定管理
│   ├── __init__.py
│   └── settings.py
├── core/                   # コア機能
│   ├── __init__.py
│   └── session_manager.py
├── components/             # UIコンポーネント
├── utils/                  # ユーティリティ
├── tests/                  # テスト
├── requirements.txt        # 依存関係
├── pyproject.toml         # 開発ツール設定
├── ROADMAP.md             # 開発ロードマップ
├── CHECKLIST.md           # 品質チェックリスト
├── DAILY_LOG.md           # 開発日誌
├── PROGRESS.json          # 進捗データ
└── progress_tracker.py    # 進捗管理ツール
```

## 📊 開発進捗の確認

```bash
# 進捗レポートの表示
python progress_tracker.py --report

# Markdown形式で出力
python progress_tracker.py --report --format markdown
```

## 🧪 テスト実行

```bash
# テストの実行
pytest

# カバレッジ付きテスト
pytest --cov=. --cov-report=html
```

## 🔧 開発ツール

### コードフォーマット

```bash
# Black（フォーマッター）
black .

# isort（インポート整理）
isort .
```

### コード品質チェック

```bash
# 型チェック
mypy .

# リンター
pylint **/*.py
```

## 📖 ドキュメント

- [ROADMAP.md](ROADMAP.md) - 開発ロードマップ
- [CHECKLIST.md](CHECKLIST.md) - 品質チェックリスト
- [CLAUDE.md](CLAUDE.md) - Claude Code用ガイド

## 🎓 研究用途

このプロジェクトは博士課程の研究用に開発されています。以下の品質基準を維持：

- テストカバレッジ: 90%以上
- Pylintスコア: 9.0以上
- 型チェック: エラー0件
- 全関数にDocstring記載

## 📝 ライセンス

研究用プロジェクト

## 👥 開発者

博士課程研究プロジェクト
