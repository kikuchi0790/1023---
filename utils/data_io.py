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
    
    dsm_optimization = None
    if "dsm_llm_params" in st.session_state:
        dsm_optimization = {
            "llm_params": st.session_state.dsm_llm_params,
            "step1_results": None,
            "step1_selected_idx": st.session_state.get("step1_selected_idx"),
            "step2_results": None,
            "step2_selected_idx": st.session_state.get("step2_selected_idx"),
            "optimized_dsm": None
        }
        
        if "step1_results" in st.session_state and st.session_state.step1_results:
            step1_serializable = []
            for r in st.session_state.step1_results:
                step1_serializable.append({
                    "cost": r["cost"],
                    "freedom": r["freedom"],
                    "removed_count": r["removed_count"],
                    "removed_nodes": r["removed_nodes"]
                })
            dsm_optimization["step1_results"] = step1_serializable
        
        if "step2_results" in st.session_state and st.session_state.step2_results:
            step2_serializable = []
            for r in st.session_state.step2_results:
                matrix_list = r["matrix"].tolist() if hasattr(r["matrix"], "tolist") else r["matrix"]
                step2_serializable.append({
                    "matrix": matrix_list,
                    "adjustment": r["adjustment"],
                    "conflict": r["conflict"],
                    "loop": r["loop"]
                })
            dsm_optimization["step2_results"] = step2_serializable
        
        if "optimized_dsm" in st.session_state and st.session_state.optimized_dsm is not None:
            dsm_optimization["optimized_dsm"] = st.session_state.optimized_dsm.tolist() if hasattr(st.session_state.optimized_dsm, "tolist") else st.session_state.optimized_dsm
    
    export_data = {
        "version": "1.0.0",
        "exported_at": datetime.now().isoformat(),
        "project_data": SessionManager.get_project_data(),
        "messages": SessionManager.get_messages(),
        "adj_matrix_df": adj_matrix_data,
        "evaluation_matrices": SessionManager.get_evaluation_matrices(),
        "evaluation_pairs": st.session_state.get("evaluation_pairs"),
        "filtered_pairs": st.session_state.get("filtered_pairs"),
        "evaluation_index": SessionManager.get_evaluation_index(),
        "dsm_optimization": dsm_optimization,
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
        
        # 評価行列シート（行列形式）
        eval_matrices = SessionManager.get_evaluation_matrices()
        nodes = SessionManager.get_nodes()
        
        if eval_matrices and nodes:
            n = len(nodes)
            full_matrix = np.zeros((n, n), dtype=int)
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            # 各カテゴリ間の行列を統合
            for matrix_data in eval_matrices:
                from_nodes = matrix_data["from_nodes"]
                to_nodes = matrix_data["to_nodes"]
                matrix = matrix_data["matrix"]
                
                for i, from_node in enumerate(from_nodes):
                    if from_node not in node_to_idx:
                        continue
                    for j, to_node in enumerate(to_nodes):
                        if to_node not in node_to_idx:
                            continue
                        fi = node_to_idx[from_node]
                        ti = node_to_idx[to_node]
                        full_matrix[fi][ti] = matrix[i][j]
            
            # DataFrameに変換
            eval_matrix_df = pd.DataFrame(
                full_matrix,
                index=nodes,
                columns=nodes
            )
            
            eval_matrix_df.to_excel(writer, sheet_name='評価行列')
        
        if "adj_matrix_df" in st.session_state and st.session_state.adj_matrix_df is not None:
            st.session_state.adj_matrix_df.to_excel(writer, sheet_name='隣接行列')
        
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
        
        if "dsm_llm_params" in st.session_state and st.session_state.dsm_llm_params:
            llm_params = st.session_state.dsm_llm_params
            params_data = llm_params.get("parameters", {})
            
            if params_data:
                from utils.idef0_classifier import classify_node_type, NodeType
                all_idef0 = SessionManager.get_all_idef0_nodes()
                
                dsm_params_rows = []
                for node_name in nodes:
                    node_params = params_data.get(node_name, {})
                    node_type, _ = classify_node_type(node_name, all_idef0)
                    
                    row = {
                        "ノード名": node_name,
                        "タイプ": "FR" if node_type == NodeType.OUTPUT else "DP",
                        "コスト": node_params.get("cost", "-") if node_type != NodeType.OUTPUT else "-",
                        "変動範囲": node_params.get("range", "-") if node_type != NodeType.OUTPUT else "-",
                        "重要度": node_params.get("importance", "-") if node_type == NodeType.OUTPUT else "-",
                        "構造グループ": node_params.get("structure", "-")
                    }
                    dsm_params_rows.append(row)
                
                dsm_params_df = pd.DataFrame(dsm_params_rows)
                dsm_params_df.to_excel(writer, sheet_name='DSMパラメータ', index=False)
        
        if "optimized_dsm" in st.session_state and st.session_state.optimized_dsm is not None:
            optimized_dsm = st.session_state.optimized_dsm
            
            if "dsm_data" in st.session_state and st.session_state.dsm_data:
                dsm_data = st.session_state.dsm_data
                remaining_nodes = dsm_data.reordered_nodes
                
                if "step1_selected_idx" in st.session_state and "step1_results" in st.session_state:
                    selected = st.session_state.step1_results[st.session_state.step1_selected_idx]
                    removed_nodes = selected.get("removed_nodes", [])
                    remaining_nodes = [n for n in dsm_data.reordered_nodes if n not in removed_nodes]
                
                optimized_df = pd.DataFrame(
                    optimized_dsm,
                    index=remaining_nodes,
                    columns=remaining_nodes
                )
                optimized_df.to_excel(writer, sheet_name='最適化DSM')
    
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
        
        if "evaluation_matrices" in json_data:
            st.session_state.evaluation_matrices = json_data["evaluation_matrices"]
        
        if "evaluation_index" in json_data:
            st.session_state.evaluation_index = json_data["evaluation_index"]
        
        if "dsm_optimization" in json_data and json_data["dsm_optimization"]:
            dsm_opt = json_data["dsm_optimization"]
            
            if dsm_opt.get("llm_params"):
                st.session_state.dsm_llm_params = dsm_opt["llm_params"]
            
            if dsm_opt.get("step1_results"):
                st.session_state.step1_results = dsm_opt["step1_results"]
            
            if dsm_opt.get("step1_selected_idx") is not None:
                st.session_state.step1_selected_idx = dsm_opt["step1_selected_idx"]
            
            if dsm_opt.get("step2_results"):
                step2_with_arrays = []
                for r in dsm_opt["step2_results"]:
                    step2_with_arrays.append({
                        "matrix": np.array(r["matrix"]),
                        "adjustment": r["adjustment"],
                        "conflict": r["conflict"],
                        "loop": r["loop"]
                    })
                st.session_state.step2_results = step2_with_arrays
            
            if dsm_opt.get("step2_selected_idx") is not None:
                st.session_state.step2_selected_idx = dsm_opt["step2_selected_idx"]
            
            if dsm_opt.get("optimized_dsm"):
                st.session_state.optimized_dsm = np.array(dsm_opt["optimized_dsm"])
        
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
        
        if '評価行列' in xl.sheet_names:
            eval_matrix_df = pd.read_excel(xl, sheet_name='評価行列', index_col=0)
            nodes = eval_matrix_df.index.tolist()
            matrix = eval_matrix_df.values.tolist()
            
            # 行列形式で保存
            SessionManager.save_evaluation_matrix(
                from_nodes=nodes,
                to_nodes=nodes,
                matrix=matrix,
                from_category="全体",
                to_category="全体"
            )
        
        if 'DSMパラメータ' in xl.sheet_names:
            dsm_params_df = pd.read_excel(xl, sheet_name='DSMパラメータ')
            
            parameters = {}
            for _, row in dsm_params_df.iterrows():
                node_name = row['ノード名']
                node_type = row['タイプ']
                
                params = {
                    "structure": str(row['構造グループ']) if pd.notna(row['構造グループ']) else ""
                }
                
                if node_type == "FR":
                    if pd.notna(row['重要度']) and row['重要度'] != "-":
                        params["importance"] = int(row['重要度'])
                else:
                    if pd.notna(row['コスト']) and row['コスト'] != "-":
                        params["cost"] = int(row['コスト'])
                    if pd.notna(row['変動範囲']) and row['変動範囲'] != "-":
                        params["range"] = float(row['変動範囲'])
                
                parameters[node_name] = params
            
            st.session_state.dsm_llm_params = {
                "parameters": parameters,
                "reasoning": "Excelからインポートされました"
            }
        
        if '最適化DSM' in xl.sheet_names:
            optimized_df = pd.read_excel(xl, sheet_name='最適化DSM', index_col=0)
            st.session_state.optimized_dsm = optimized_df.values
        
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
