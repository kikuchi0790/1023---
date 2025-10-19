"""
Data Export/Import utilities for PIM application.
PIMアプリケーションのデータエクスポート/インポート機能
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from io import BytesIO
import json

import pandas as pd
import numpy as np
import streamlit as st

from core.session_manager import SessionManager


def export_to_json() -> Dict[str, Any]:
    """
    完全なセッション状態をJSONとしてエクスポート
    
    Returns:
        エクスポートデータ（辞書）
    """
    adj_matrix_data = None
    if "adj_matrix_df" in st.session_state and st.session_state.adj_matrix_df is not None:
        adj_matrix_data = st.session_state.adj_matrix_df.to_dict()
    
    export_data = {
        "version": "1.0.0",
        "exported_at": datetime.now().isoformat(),
        "project_data": SessionManager.get_project_data(),
        "messages": SessionManager.get_messages(),
        "adj_matrix_df": adj_matrix_data,
        "evaluation_pairs": st.session_state.get("evaluation_pairs"),
        "filtered_pairs": st.session_state.get("filtered_pairs"),
        "evaluation_index": SessionManager.get_evaluation_index(),
    }
    
    return export_data


def export_to_excel() -> BytesIO:
    """
    Excelファイル（複数シート）としてエクスポート
    
    Returns:
        BytesIO: Excelファイルのバイナリストリーム
    """
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        project_info_df = pd.DataFrame([{
            "プロセス名": SessionManager.get_process_name(),
            "プロセス概要": SessionManager.get_process_description(),
            "エクスポート日時": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "カテゴリ数": len(SessionManager.get_functional_categories()),
            "ノード数": len(SessionManager.get_nodes()),
        }])
        project_info_df.to_excel(writer, sheet_name='プロジェクト情報', index=False)
        
        categories = SessionManager.get_functional_categories()
        if categories:
            categories_df = pd.DataFrame({
                "カテゴリ名": categories
            })
            categories_df.to_excel(writer, sheet_name='機能カテゴリ', index=False)
        
        idef0_nodes = SessionManager.get_all_idef0_nodes()
        if idef0_nodes:
            idef0_rows = []
            for category, data in idef0_nodes.items():
                idef0_rows.append({
                    "カテゴリ": category,
                    "機能": data.get("function", category),
                    "Inputs": "; ".join(data.get("inputs", [])),
                    "Mechanisms": "; ".join(data.get("mechanisms", [])),
                    "Outputs": "; ".join(data.get("outputs", [])),
                })
            idef0_df = pd.DataFrame(idef0_rows)
            idef0_df.to_excel(writer, sheet_name='IDEF0ノード', index=False)
        
        if "adj_matrix_df" in st.session_state and st.session_state.adj_matrix_df is not None:
            st.session_state.adj_matrix_df.to_excel(writer, sheet_name='隣接行列')
        
        evaluations = SessionManager.get_evaluations()
        if evaluations:
            evaluations_df = pd.DataFrame(evaluations)
            evaluations_df.to_excel(writer, sheet_name='評価詳細', index=False)
        
        nodes_rows = []
        for category, idef0_data in SessionManager.get_all_idef0_nodes().items():
            for node_type, node_list in [
                ("Input", idef0_data.get("inputs", [])),
                ("Mechanism", idef0_data.get("mechanisms", [])),
                ("Output", idef0_data.get("outputs", []))
            ]:
                for node in node_list:
                    nodes_rows.append({
                        "ノード名": node,
                        "カテゴリ": category,
                        "タイプ": node_type
                    })
        
        if nodes_rows:
            nodes_df = pd.DataFrame(nodes_rows)
            nodes_df.to_excel(writer, sheet_name='ノードリスト', index=False)
    
    buffer.seek(0)
    return buffer


def export_adjacency_matrix_to_csv() -> str:
    """
    隣接行列のみをCSVとしてエクスポート
    
    Returns:
        CSV文字列
        
    Raises:
        ValueError: 隣接行列が生成されていない場合
    """
    if "adj_matrix_df" in st.session_state and st.session_state.adj_matrix_df is not None:
        return st.session_state.adj_matrix_df.to_csv()
    else:
        raise ValueError("隣接行列が生成されていません。タブ5で隣接行列を生成してください。")


def import_from_json(json_data: Dict[str, Any]) -> bool:
    """
    JSONデータからセッション状態を復元
    
    Args:
        json_data: インポートするJSONデータ
        
    Returns:
        成功したかどうか
    """
    try:
        version = json_data.get("version", "0.0.0")
        if not version.startswith("1."):
            raise ValueError(f"非対応のバージョン: {version}")
        
        if "project_data" in json_data:
            st.session_state.project_data = json_data["project_data"]
        
        if "messages" in json_data:
            st.session_state.messages = json_data["messages"]
        
        if "adj_matrix_df" in json_data and json_data["adj_matrix_df"]:
            st.session_state.adj_matrix_df = pd.DataFrame(json_data["adj_matrix_df"])
            st.session_state.adjacency_matrix = st.session_state.adj_matrix_df.values
        
        if "evaluation_pairs" in json_data:
            st.session_state.evaluation_pairs = json_data["evaluation_pairs"]
        
        if "filtered_pairs" in json_data:
            st.session_state.filtered_pairs = json_data["filtered_pairs"]
        
        if "evaluation_index" in json_data:
            st.session_state.evaluation_index = json_data["evaluation_index"]
        
        return True
        
    except Exception as e:
        st.error(f"JSONインポートエラー: {str(e)}")
        return False


def import_from_excel(excel_file) -> bool:
    """
    Excelファイルからデータをインポート
    
    Args:
        excel_file: アップロードされたExcelファイル
        
    Returns:
        成功したかどうか
    """
    try:
        xl = pd.ExcelFile(excel_file)
        
        if 'プロジェクト情報' in xl.sheet_names:
            project_df = pd.read_excel(xl, sheet_name='プロジェクト情報')
            if not project_df.empty:
                SessionManager.update_process_info(
                    str(project_df.iloc[0]['プロセス名']),
                    str(project_df.iloc[0]['プロセス概要'])
                )
        
        if '機能カテゴリ' in xl.sheet_names:
            categories_df = pd.read_excel(xl, sheet_name='機能カテゴリ')
            categories = categories_df['カテゴリ名'].tolist()
            SessionManager.set_functional_categories(categories)
        
        if 'IDEF0ノード' in xl.sheet_names:
            idef0_df = pd.read_excel(xl, sheet_name='IDEF0ノード')
            for _, row in idef0_df.iterrows():
                category = row['カテゴリ']
                idef0_data = {
                    "function": row.get('機能', category),
                    "inputs": row['Inputs'].split('; ') if pd.notna(row['Inputs']) else [],
                    "mechanisms": row['Mechanisms'].split('; ') if pd.notna(row['Mechanisms']) else [],
                    "outputs": row['Outputs'].split('; ') if pd.notna(row['Outputs']) else [],
                }
                SessionManager.set_idef0_node(category, idef0_data)
        
        if '隣接行列' in xl.sheet_names:
            adj_df = pd.read_excel(xl, sheet_name='隣接行列', index_col=0)
            st.session_state.adj_matrix_df = adj_df
            st.session_state.adjacency_matrix = adj_df.values
        
        if '評価詳細' in xl.sheet_names:
            eval_df = pd.read_excel(xl, sheet_name='評価詳細')
            if not eval_df.empty:
                st.session_state.project_data["evaluations"] = eval_df.to_dict('records')
        
        return True
        
    except Exception as e:
        st.error(f"Excelインポートエラー: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False


def import_adjacency_matrix_from_csv(csv_file) -> bool:
    """
    CSVファイルから隣接行列をインポート
    
    Args:
        csv_file: アップロードされたCSVファイル
        
    Returns:
        成功したかどうか
    """
    try:
        csv_df = pd.read_csv(csv_file, index_col=0)
        st.session_state.adj_matrix_df = csv_df
        st.session_state.adjacency_matrix = csv_df.values
        return True
        
    except Exception as e:
        st.error(f"CSVインポートエラー: {str(e)}")
        return False
