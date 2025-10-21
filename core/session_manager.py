"""
Session state management for Streamlit application.
Streamlitアプリケーションのセッション状態管理
"""

from typing import Any, Dict, List, Optional
import streamlit as st
from config.settings import settings


class SessionManager:
    """Streamlitセッション状態管理クラス"""

    @staticmethod
    def initialize() -> None:
        """
        セッションステートの初期化
        アプリケーション起動時に一度だけ呼び出される
        """
        if "project_data" not in st.session_state:
            st.session_state.project_data = {
                "process_name": settings.DEFAULT_PROCESS_NAME,
                "process_description": settings.DEFAULT_PROCESS_DESCRIPTION,
                "functional_categories": [],
                "nodes": [],
                "idef0_nodes": {},
                "evaluations": [],
                "adjacency_matrix": None,
            }

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "evaluation_index" not in st.session_state:
            st.session_state.evaluation_index = 0

    @staticmethod
    def get_project_data() -> Dict[str, Any]:
        """
        プロジェクトデータの取得

        Returns:
            プロジェクトデータの辞書
        """
        return st.session_state.project_data

    @staticmethod
    def update_process_info(name: str, description: str) -> None:
        """
        プロセス情報の更新

        Args:
            name: プロセス名
            description: プロセス説明
        """
        st.session_state.project_data["process_name"] = name
        st.session_state.project_data["process_description"] = description

    @staticmethod
    def get_process_name() -> str:
        """プロセス名の取得"""
        return st.session_state.project_data["process_name"]

    @staticmethod
    def get_process_description() -> str:
        """プロセス説明の取得"""
        return st.session_state.project_data["process_description"]

    @staticmethod
    def set_functional_categories(categories: List[str]) -> None:
        """
        機能カテゴリの設定

        Args:
            categories: 機能カテゴリのリスト
        """
        st.session_state.project_data["functional_categories"] = categories

    @staticmethod
    def get_functional_categories() -> List[str]:
        """機能カテゴリの取得"""
        return st.session_state.project_data["functional_categories"]

    @staticmethod
    def set_nodes(nodes: List[str]) -> None:
        """
        ノードリストの設定

        Args:
            nodes: ノードのリスト
        """
        st.session_state.project_data["nodes"] = nodes

    @staticmethod
    def get_nodes() -> List[str]:
        """ノードリストの取得"""
        return st.session_state.project_data["nodes"]

    @staticmethod
    def set_idef0_node(category: str, idef0_data: Dict[str, Any]) -> None:
        """
        IDEF0ノードの設定
        
        Args:
            category: 機能カテゴリ名
            idef0_data: IDEF0データ（inputs, mechanisms, outputs）
        """
        if "idef0_nodes" not in st.session_state.project_data:
            st.session_state.project_data["idef0_nodes"] = {}
        
        st.session_state.project_data["idef0_nodes"][category] = idef0_data
        
        SessionManager._update_nodes_from_idef0()
    
    @staticmethod
    def _update_nodes_from_idef0() -> None:
        """
        IDEF0ノードデータから統合ノードリストを自動構築
        内部メソッド: set_idef0_node()から自動的に呼ばれる
        """
        if "idef0_nodes" not in st.session_state.project_data:
            return
        
        all_nodes = set()
        
        for category, idef0_data in st.session_state.project_data["idef0_nodes"].items():
            if idef0_data.get("inputs"):
                all_nodes.update(idef0_data["inputs"])
            
            if idef0_data.get("mechanisms"):
                all_nodes.update(idef0_data["mechanisms"])
            
            if idef0_data.get("outputs"):
                all_nodes.update(idef0_data["outputs"])
        
        nodes_list = sorted(list(all_nodes))
        
        st.session_state.project_data["nodes"] = nodes_list

    @staticmethod
    def get_idef0_node(category: str) -> Optional[Dict[str, Any]]:
        """
        IDEF0ノードの取得
        
        Args:
            category: 機能カテゴリ名
            
        Returns:
            IDEF0データ、存在しない場合はNone
        """
        if "idef0_nodes" not in st.session_state.project_data:
            st.session_state.project_data["idef0_nodes"] = {}
        
        return st.session_state.project_data["idef0_nodes"].get(category)

    @staticmethod
    def get_all_idef0_nodes() -> Dict[str, Dict[str, Any]]:
        """全IDEF0ノードの取得"""
        if "idef0_nodes" not in st.session_state.project_data:
            st.session_state.project_data["idef0_nodes"] = {}
        
        return st.session_state.project_data["idef0_nodes"]

    @staticmethod
    def add_evaluation(
        from_node: str, to_node: str, score: int, reason: str = ""
    ) -> None:
        """
        評価の追加

        Args:
            from_node: 評価元ノード
            to_node: 評価先ノード
            score: 評価スコア
            reason: 評価理由（オプション、高速化のため通常は空文字列）
        """
        evaluations = st.session_state.project_data["evaluations"]

        for eval_item in evaluations:
            if (
                eval_item["from_node"] == from_node
                and eval_item["to_node"] == to_node
            ):
                eval_item["score"] = score
                eval_item["reason"] = reason
                return

        evaluations.append(
            {"from_node": from_node, "to_node": to_node, "score": score, "reason": reason}
        )

    @staticmethod
    def save_evaluation_matrix(
        from_nodes: List[str],
        to_nodes: List[str],
        matrix: List[List[int]],
        from_category: str,
        to_category: str
    ) -> None:
        """
        評価行列を保存（行列形式）
        
        Args:
            from_nodes: 評価元ノードリスト（行）
            to_nodes: 評価先ノードリスト（列）
            matrix: n×mの評価行列（2次元リスト）
            from_category: 評価元カテゴリ名
            to_category: 評価先カテゴリ名
        """
        if "evaluation_matrices" not in st.session_state:
            st.session_state.evaluation_matrices = []
        
        # 既存の行列を検索（上書き対応）
        for existing in st.session_state.evaluation_matrices:
            if (existing["from_category"] == from_category and
                existing["to_category"] == to_category):
                existing["from_nodes"] = from_nodes
                existing["to_nodes"] = to_nodes
                existing["matrix"] = matrix
                return
        
        # 新規追加
        st.session_state.evaluation_matrices.append({
            "from_category": from_category,
            "to_category": to_category,
            "from_nodes": from_nodes,
            "to_nodes": to_nodes,
            "matrix": matrix
        })

    @staticmethod
    def get_evaluation_matrices() -> List[Dict[str, Any]]:
        """
        評価行列リストを取得
        
        Returns:
            評価行列のリスト
        """
        if "evaluation_matrices" not in st.session_state:
            st.session_state.evaluation_matrices = []
        return st.session_state.evaluation_matrices

    @staticmethod
    def get_evaluations() -> List[Dict[str, Any]]:
        """
        評価リストの取得（後方互換性のため）
        
        行列形式から動的にリスト形式を生成します。
        非ゼロの評価のみを返します（疎行列最適化）。
        
        Returns:
            評価リスト
        """
        # まず行列形式から生成を試みる
        if "evaluation_matrices" in st.session_state and st.session_state.evaluation_matrices:
            evaluations = []
            matrices = st.session_state.evaluation_matrices
            
            for matrix_data in matrices:
                from_nodes = matrix_data["from_nodes"]
                to_nodes = matrix_data["to_nodes"]
                matrix = matrix_data["matrix"]
                
                for i, from_node in enumerate(from_nodes):
                    for j, to_node in enumerate(to_nodes):
                        score = matrix[i][j]
                        if score != 0:  # 非ゼロのみ（疎行列）
                            evaluations.append({
                                "from_node": from_node,
                                "to_node": to_node,
                                "score": score,
                                "reason": ""
                            })
            
            return evaluations
        
        # フォールバック: 旧形式の評価リスト
        return st.session_state.project_data.get("evaluations", [])

    @staticmethod
    def get_evaluation(from_node: str, to_node: str) -> Optional[Dict[str, Any]]:
        """
        特定のノードペアの評価を取得

        Args:
            from_node: 評価元ノード
            to_node: 評価先ノード

        Returns:
            評価データ、存在しない場合はNone
        """
        evaluations = st.session_state.project_data["evaluations"]
        for eval_item in evaluations:
            if (
                eval_item["from_node"] == from_node
                and eval_item["to_node"] == to_node
            ):
                return eval_item
        return None

    @staticmethod
    def set_adjacency_matrix(matrix: Any) -> None:
        """
        隣接行列の設定

        Args:
            matrix: NumPy配列または行列データ
        """
        st.session_state.project_data["adjacency_matrix"] = matrix

    @staticmethod
    def get_adjacency_matrix() -> Optional[Any]:
        """隣接行列の取得"""
        return st.session_state.project_data["adjacency_matrix"]

    @staticmethod
    def add_message(role: str, content: str) -> None:
        """
        チャットメッセージの追加

        Args:
            role: メッセージの役割（"user" or "assistant"）
            content: メッセージ内容
        """
        st.session_state.messages.append({"role": role, "content": content})

    @staticmethod
    def get_messages() -> List[Dict[str, str]]:
        """チャットメッセージの取得"""
        return st.session_state.messages

    @staticmethod
    def clear_messages() -> None:
        """チャットメッセージのクリア"""
        st.session_state.messages = []

    @staticmethod
    def get_evaluation_index() -> int:
        """評価インデックスの取得"""
        return st.session_state.evaluation_index

    @staticmethod
    def set_evaluation_index(index: int) -> None:
        """
        評価インデックスの設定

        Args:
            index: 新しいインデックス
        """
        st.session_state.evaluation_index = index
