# Process Insight Modeler (PIM)

**プロセスの暗黙知を形式知に変換し、データ駆動で最適化する対話型分析ツール**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-Research-green.svg)](LICENSE)

---

## 📖 目次

- [概要](#-概要)
- [主な機能](#-主な機能)
- [デモ動画](#-デモ動画)
- [クイックスタート](#-クイックスタート)
- [9ステップのワークフロー](#-9ステップのワークフロー)
- [高度な分析機能](#-高度な分析機能タブ9)
- [プロジェクト構造](#-プロジェクト構造)
- [開発者向け情報](#-開発者向け情報)
- [貢献とサポート](#-貢献とサポート)

---

## 🎯 概要

**Process Insight Modeler (PIM)** は、製造業・サービス業などの複雑なプロセスを持つ組織が、専門家の持つ**暗黙知**を**形式知**に変換し、科学的に分析・最適化するためのツールです。

### なぜPIMが必要か？

- **🧠 暗黙知の可視化**: ベテラン社員の頭の中にある知識を、AIとの対話で引き出す
- **📊 データ駆動の意思決定**: 直感ではなく、統計・ネットワーク分析に基づく改善提案
- **🔄 反復的な精緻化**: 一度作って終わりではなく、分析結果をフィードバックして知識を深化
- **🎓 博士課程レベルの分析**: Shapley Value、Transfer Entropy、因果推論など、最先端の分析手法を搭載

### 誰のためのツール？

- **生産技術者・プロセス改善担当者**: ボトルネック特定、工程最適化
- **品質管理者**: 品質に影響する要因の科学的分析
- **コンサルタント**: クライアントの業務プロセス可視化と改善提案
- **研究者**: プロセスマイニング、ネットワーク分析の研究基盤

---

## ✨ 主な機能

### 1. **AI主導の対話型ノード定義**
OpenAI GPT-4oが専門家と対話し、プロセスを「機能」と「実体（ノード）」に分解します。

### 2. **IDEF0形式のプロセス分解**
- **Input（入力）**: 材料、情報
- **Mechanism（手段）**: 道具、スキル、設備
- **Output（出力）**: 成果物、性能指標
- **Function（機能）**: 各工程の役割

### 3. **Zigzagging手法による段階的細分化**
- 粗い粒度から始め、分析結果を見ながら重要なノードを細分化
- 無駄な詳細化を避け、効率的に知識を深化

### 4. **多層ネットワーク可視化**
- **3D可視化**: Three.js + NetworkMapsでインタラクティブな3D表示
- **2D可視化**: Cytoscape.jsで階層的レイアウト

### 5. **7つの博士課程レベル分析手法**
- Shapley Value（協力貢献度分析）
- Transfer Entropy（情報フロー分析）
- Bootstrap統計検定
- Bayesian Inference（不確実性定量化）
- Causal Inference（因果推論）
- Graph Embedding（潜在構造発見）
- Fisher Information（感度分析）

### 6. **DSM最適化（NSGA-II）**
- 設計構造マトリクス（DSM）を多目的遺伝的アルゴリズムで最適化
- パレートフロント可視化で複数の最適解を探索

---

## 🎬 デモ動画

*(ここにGIF動画やスクリーンショットを追加予定)*

---

## 🚀 クイックスタート

### 前提条件

- **Python 3.10以上**
- **OpenAI APIキー**（GPT-4o使用）
- macOS / Linux / Windows対応

### インストール手順

#### 1. リポジトリのクローン

```bash
git clone https://github.com/your-username/networkmaps.git
cd networkmaps/pim
```

#### 2. 仮想環境のセットアップ（推奨）

```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
```

#### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

**主な依存パッケージ:**
- `streamlit` - UIフレームワーク
- `openai` - LLM統合
- `pandas`, `numpy` - データ処理
- `networkx` - ネットワーク分析
- `matplotlib`, `seaborn` - 可視化
- `scikit-learn`, `scipy` - 機械学習・統計

#### 4. OpenAI APIキーの設定

`.streamlit/secrets.toml` ファイルを作成：

```toml
OPENAI_API_KEY = "sk-..."
```

**セキュリティ注意:** このファイルは`.gitignore`に含まれています。GitHubにpushされません。

#### 5. アプリケーションの起動

```bash
NETWORKMAPS_RELEASE=true streamlit run app_tabs.py
```

ブラウザで **http://localhost:8501** が開きます。

---

## 📋 9ステップのワークフロー

PIMは段階的なワークフローで、初心者でも迷わず使えます：

### **ステップ1: プロセス定義**
プロセス名と概要を入力（例: 「新製品開発プロセス」）

### **ステップ2: 機能カテゴリ抽出**
AIが自動でプロセスを5-8個の機能カテゴリに分類
- 📊 多様性生成オプション: 5つの異なる視点から候補を生成

### **ステップ3: ノード定義（IDEF0）**
3つのモード:
- **🤖 AI主導対話**: ソクラテス式対話でノードを深掘り
- **🎲 多様性生成**: 5つの思考モードで代替案を生成
- **🔄 Zigzagging粒度調整**: 既存ノードを細分化

### **ステップ4: ノード影響評価**
- Zigzagging手法による段階的評価
- 論理ルールベースフィルタリング（80-90%削減）
- LLMバッチ評価で効率化

### **ステップ5: 行列分析**
- 隣接行列生成
- ヒートマップ可視化
- **粒度調整提案**: 高次数ノードを自動検出

### **ステップ6: ネットワーク可視化**
- 3D可視化（Three.js）
- 2D可視化（Cytoscape.js）

### **ステップ7: ネットワーク分析**
- PageRank、次数中心性、媒介中心性
- **粒度調整提案**: 重要ノードを自動検出

### **ステップ8: DSM最適化**
- NSGA-IIによる多目的最適化
- パレートフロント可視化
- **軽量モード**でクラッシュ対策

### **ステップ9: 高度な分析**
7つの博士課程レベル分析手法（詳細は下記）

---

## 🧬 高度な分析機能（タブ9）

### 1. **Shapley Value（協力貢献度分析）** ⭐ 推奨
**何がわかるか**: 各ノードの「真の貢献度」を公平に評価

**使用例**:
- 投資優先順位の決定
- 見えにくい「縁の下の力持ち」の発見
- **連携安定性分析**: 上位ノード同士の協力効果を可視化

### 2. **Transfer Entropy（情報フロー分析）** ⭐ 推奨
**何がわかるか**: 「誰が誰に何bit情報を伝えているか」を定量化

**使用例**:
- 真のボトルネック特定
- コミュニケーション設計
- 見かけの相関 vs 真の因果関係

### 3. **Bootstrap統計検定** ⭐ 推奨
**何がわかるか**: 分析結果の統計的信頼性

**使用例**:
- 経営層への説明（95%信頼区間付き）
- 再評価箇所の特定
- グループ間比較

### 4. **Bayesian Inference（不確実性定量化）**
**何がわかるか**: LLM評価の信頼性を数値化

**使用例**:
- 再評価箇所の優先順位付け
- リスク評価
- シナリオ分析

### 5. **Causal Inference（因果推論）**
**何がわかるか**: 「もし工程Aを改善したら、何が起こるか」をシミュレーション

**使用例**:
- 介入効果の予測
- 改善施策の優先順位
- 交絡因子の特定

### 6. **Graph Embedding（潜在構造発見）**
**何がわかるか**: ネットワークの潜在的なコミュニティ構造

**使用例**:
- モジュール分割
- 類似工程の発見
- 階層的プロセス理解

### 7. **Fisher Information（感度分析）**
**何がわかるか**: どのエッジが不正確だと全体が歪むか

**使用例**:
- 再評価の優先順位決定
- データ収集計画
- 信頼性評価

---

## 📂 プロジェクト構造

```
pim/
├── app_tabs.py                    # ⭐ メインアプリケーション（9タブUI）
├── config/
│   └── settings.py                # 設定管理
├── core/
│   ├── session_manager.py         # セッションステート管理
│   ├── llm_client.py              # OpenAI API統合
│   └── data_models.py             # Pydanticデータモデル
├── utils/
│   ├── shapley_analysis.py        # Shapley Value + 連携安定性
│   ├── information_theory_analysis.py  # Transfer Entropy
│   ├── statistical_testing.py     # Bootstrap統計検定
│   ├── bayesian_analysis.py       # Bayesian Inference
│   ├── causal_inference.py        # 因果推論
│   ├── graph_embedding.py         # Graph Embedding
│   ├── fisher_information.py      # Fisher Information
│   ├── analytics_export.py        # Excel/JSONエクスポート
│   ├── dsm_optimizer.py           # NSGA-II最適化
│   ├── dsm_partitioning.py        # DSMパーティショニング
│   ├── data_io.py                 # データI/O
│   └── ...
├── components/
│   ├── networkmaps_viewer/        # 3D可視化コンポーネント
│   └── cytoscape_viewer/          # 2D可視化コンポーネント
├── tests/                         # テストスイート（10ファイル）
├── docs/
│   └── USER_GUIDE_TAB9.md         # ユーザーガイド
├── README.md                      # このファイル
├── CLAUDE.md                      # 開発者向けガイド
└── requirements.txt               # 依存関係
```

---

## 🛠️ 開発者向け情報

### テスト実行

```bash
# 全テスト実行
pytest

# カバレッジ付き
pytest --cov=utils --cov-report=html

# 特定のテストのみ
pytest tests/test_shapley_analysis.py -v
```

**テストカバレッジ**: 82%以上（主要分析モジュール）

### コード品質

```bash
# フォーマット
black .
isort .

# 型チェック
mypy utils/

# リンター
pylint utils/
```

### 進捗管理

```bash
# 進捗レポート
python progress_tracker.py --report

# タスク更新
python progress_tracker.py --update-task task_id --status completed
```

### カスタムコンポーネント開発

```bash
cd components/networkmaps_viewer/frontend
npm install
npm run build
```

---

## 📚 ドキュメント

| ファイル | 内容 |
|---------|------|
| **[README.md](README.md)** | このファイル（概要・クイックスタート） |
| **[CLAUDE.md](CLAUDE.md)** | 開発者向け技術ガイド（アーキテクチャ、実装詳細） |
| **[docs/USER_GUIDE_TAB9.md](docs/USER_GUIDE_TAB9.md)** | タブ9高度な分析の使い方ガイド |
| **[ROADMAP.md](ROADMAP.md)** | 開発ロードマップ |

---

## 🤝 貢献とサポート

### バグ報告・機能要望

GitHubの[Issues](https://github.com/your-username/networkmaps/issues)でお知らせください。

### プルリクエスト

1. このリポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m '✨ Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

### コントリビューションガイドライン

- テストカバレッジ90%以上
- Black + isortでフォーマット
- Docstring必須
- 日本語UIテキスト

---

## 📄 ライセンス

研究用プロジェクト - 学術・教育目的での使用を想定

---

## 🙏 謝辞

- **OpenAI GPT-4o**: AI主導対話・分析解釈生成
- **Streamlit**: 高速プロトタイピング
- **NetworkX**: ネットワーク分析基盤
- **scikit-learn, SciPy**: 統計・機械学習アルゴリズム

---

## 📞 連絡先

博士課程研究プロジェクト

**開発**: Claude Code (claude.ai/code) + Human-in-the-loop

---

## 🌟 スター・フォローをお願いします！

このプロジェクトが役立つと思ったら、GitHubでスター⭐をお願いします！

[![GitHub stars](https://img.shields.io/github/stars/your-username/networkmaps.svg?style=social&label=Star)](https://github.com/your-username/networkmaps)
