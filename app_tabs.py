"""
Process Insight Modeler (PIM) - タブ形式UI
生産プロセスの暗黙知を形式知に変換するアプリケーション
"""

import json
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAIError
from config.settings import settings
from core.session_manager import SessionManager
from core.llm_client import LLMClient
from utils.matrix_evaluator import MatrixEvaluator
from core.data_models import (
    FunctionalCategory,
    CategoryGenerationOptions,
    CategorySet
)
from utils.analytics_progress import AnalyticsProgressTracker, create_simple_callback
from utils.analytics_export import add_analytics_export_to_sidebar


def render_sidebar():
    """サイドバー: プロジェクト情報サマリー"""
    with st.sidebar:
        st.header("📊 プロジェクト情報")
        
        process_name = SessionManager.get_process_name()
        if process_name:
            st.success(f"**プロセス**: {process_name}")
        else:
            st.warning("プロセス未定義")
        
        categories = SessionManager.get_functional_categories()
        if categories:
            st.info(f"**カテゴリ数**: {len(categories)}")
        else:
            st.warning("カテゴリ未定義")
        
        nodes = SessionManager.get_nodes()
        if nodes:
            st.info(f"**ノード数**: {len(nodes)}")
        else:
            st.warning("ノード未定義")
        
        if st.session_state.get("adjacency_matrix") is not None:
            st.info("**隣接行列**: 生成済み")
        else:
            st.warning("隣接行列未生成")
        
        st.divider()
        
        st.caption("💾 データ管理")
        
        with st.expander("📤 エクスポート", expanded=False):
            st.markdown("プロジェクトデータを保存します")
            
            export_format = st.radio(
                "エクスポート形式",
                options=["Excel (.xlsx)", "JSON (.json)", "CSV (.csv)"],
                help="Excel: 全データ、JSON: 完全な復元用、CSV: 隣接行列のみ",
                key="export_format_radio"
            )
            
            if st.button("エクスポート実行", use_container_width=True, type="primary", key="export_button"):
                try:
                    from utils.data_io import export_to_excel, export_to_json, export_adjacency_matrix_to_csv
                    from datetime import datetime
                    import json
                    
                    if export_format == "Excel (.xlsx)":
                        buffer = export_to_excel()
                        st.download_button(
                            label="📥 Excelファイルをダウンロード",
                            data=buffer,
                            file_name=f"pim_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="download_excel"
                        )
                        st.success("✅ Excelエクスポート完了")
                    elif export_format == "JSON (.json)":
                        json_data = export_to_json()
                        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                        st.download_button(
                            label="📥 JSONファイルをダウンロード",
                            data=json_str,
                            file_name=f"pim_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True,
                            key="download_json"
                        )
                        st.success("✅ JSONエクスポート完了")
                    else:
                        csv_str = export_adjacency_matrix_to_csv()
                        st.download_button(
                            label="📥 CSVファイルをダウンロード",
                            data=csv_str,
                            file_name=f"adjacency_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_csv"
                        )
                        st.success("✅ CSVエクスポート完了")
                except Exception as e:
                    st.error(f"❌ エクスポートエラー: {str(e)}")
        
        with st.expander("📥 インポート", expanded=False):
            st.markdown("保存したプロジェクトデータを読み込みます")
            st.warning("⚠️ 現在のデータは上書きされます")
            
            uploaded_file = st.file_uploader(
                "ファイルを選択",
                type=["json", "xlsx", "csv"],
                help="JSON, Excel, CSVファイルに対応",
                key="import_file_uploader"
            )
            
            if uploaded_file is not None:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                st.info(f"📄 ファイル: {uploaded_file.name}")
                
                if st.button("インポート実行", use_container_width=True, type="primary", key="import_button"):
                    try:
                        from utils.data_io import import_from_json, import_from_excel, import_adjacency_matrix_from_csv
                        import json
                        
                        if file_extension == "json":
                            json_data = json.load(uploaded_file)
                            if import_from_json(json_data):
                                st.success("✅ JSONインポート成功")
                                st.info("🔄 ページを再読み込みしています...")
                                st.rerun()
                        elif file_extension == "xlsx":
                            if import_from_excel(uploaded_file):
                                st.success("✅ Excelインポート成功")
                                st.info("🔄 ページを再読み込みしています...")
                                st.rerun()
                        elif file_extension == "csv":
                            if import_adjacency_matrix_from_csv(uploaded_file):
                                st.success("✅ CSV（隣接行列）インポート成功")
                                st.info("🔄 ページを再読み込みしています...")
                                st.rerun()
                    except Exception as e:
                        st.error(f"❌ インポートエラー: {str(e)}")
                        import traceback
                        with st.expander("エラー詳細"):
                            st.code(traceback.format_exc())
        
        # 高度な分析結果のエクスポート
        add_analytics_export_to_sidebar()


def tab1_process_definition():
    """タブ1: プロセス定義"""
    st.header("📝 ステップ1: プロセス定義")
    
    st.markdown("""
    分析対象の生産プロセスを定義します。プロセス名と詳細な概要を入力してください。
    """)
    
    process_name = st.text_input(
        "生産プロセス名",
        value=SessionManager.get_process_name(),
        placeholder="例: 自動車エンジン組立工程",
        help="分析対象の生産プロセスの名称を入力してください",
    )
    
    process_description = st.text_area(
        "プロセスの概要",
        value=SessionManager.get_process_description(),
        height=300,
        placeholder="プロセスの詳細な説明を記述してください...",
        help="プロセスの詳細、主要な工程、使用する設備などを記述してください",
    )
    
    SessionManager.update_process_info(process_name, process_description)
    
    st.divider()
    
    if process_name and process_description:
        st.success("✅ プロセス定義が完了しました")
        st.info("次のタブ「機能カテゴリ」に進んでください")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("プロセス名")
            st.info(process_name)
        with col2:
            st.subheader("プロセス概要")
            st.text_area(
                "概要",
                value=process_description,
                height=150,
                disabled=True,
                label_visibility="collapsed",
            )
    else:
        st.warning("⚠️ プロセス名と概要を入力してください")


def tab2_functional_categories():
    """タブ2: 機能カテゴリ定義"""
    st.header("🎯 ステップ2: 機能カテゴリ定義")
    
    process_name = SessionManager.get_process_name()
    process_description = SessionManager.get_process_description()
    
    if not (process_name and process_description):
        st.warning("⚠️ 先にタブ1でプロセスを定義してください")
        return
    
    st.markdown("""
    プロセスを構成する「機能カテゴリ」を抽出します。
    機能カテゴリとは、プロセスの動的な変換機能（インプット→変換→アウトプット）です。
    """)
    
    with st.expander("🎯 プロセス機能の抽出設定", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analysis_focus = st.selectbox(
                "分析の視点",
                [
                    "balanced",
                    "material_flow",
                    "information_flow",
                    "quality_gates"
                ],
                format_func=lambda x: {
                    "balanced": "バランス型（推奨）",
                    "material_flow": "モノの流れ重視",
                    "information_flow": "情報の流れ重視",
                    "quality_gates": "品質ゲート重視"
                }[x],
                help="プロセス分析の視点を選択します"
            )
        
        with col2:
            granularity = st.selectbox(
                "プロセスの分解レベル",
                ["standard", "high_level", "detailed"],
                format_func=lambda x: {
                    "high_level": "高レベル（4-5個の大工程）",
                    "standard": "標準（6-8個の中工程）",
                    "detailed": "詳細（10-12個の作業工程）"
                }[x],
                help="プロセスをどのレベルまで分解するか"
            )
        
        with col3:
            use_verbalized_sampling = st.checkbox(
                "多様性生成（Verbalized Sampling）",
                value=False,
                help="5つの異なる分析哲学から生成し、比較できます（推奨）"
            )
    
    col_btn1, col_btn2 = st.columns([2, 1])
    
    with col_btn1:
        if st.button(
            "🎯 カテゴリを自動抽出",
            type="primary",
            use_container_width=True,
            help="AIが機能カテゴリを自動的に抽出します"
        ):
            try:
                llm_client = LLMClient()
                options = CategoryGenerationOptions(
                    analysis_focus=analysis_focus,
                    granularity=granularity
                )
                
                if use_verbalized_sampling:
                    with st.spinner("🎯 多様な視点から生成中..."):
                        alternatives = llm_client.generate_diverse_category_sets(
                            process_name, process_description, num_perspectives=5
                        )
                        st.session_state["category_alternatives"] = alternatives
                        if alternatives:
                            st.success(f"✅ {len(alternatives)}つの代替案を生成しました！")
                        else:
                            st.error("❌ 生成に失敗しました")
                else:
                    with st.spinner("🎯 機能カテゴリを抽出中..."):
                        categories = llm_client.extract_categories_advanced(
                            process_name, process_description, options
                        )
                        
                        if categories:
                            SessionManager.set_functional_categories(
                                [cat.name for cat in categories]
                            )
                            
                            categories_metadata = {}
                            for cat in categories:
                                categories_metadata[cat.name] = {
                                    "description": cat.description,
                                    "inputs": cat.inputs,
                                    "outputs": cat.outputs,
                                    "examples": cat.examples,
                                    "importance": cat.importance
                                }
                            st.session_state["categories_metadata"] = categories_metadata
                            
                            st.success(f"✅ {len(categories)}個のカテゴリを抽出しました！")
                        else:
                            st.error("❌ カテゴリの抽出に失敗しました")
                
            except OpenAIError as e:
                st.error(f"❌ OpenAI APIエラー: {str(e)}")
            except Exception as e:
                st.error(f"❌ エラーが発生しました: {str(e)}")
    
    with col_btn2:
        if st.button("🔄 リセット", use_container_width=True):
            SessionManager.set_functional_categories([])
            st.session_state.pop("categories_metadata", None)
            st.session_state.pop("category_alternatives", None)
            st.rerun()
    
    if use_verbalized_sampling and "category_alternatives" in st.session_state:
        st.divider()
        st.subheader("生成された案の比較")
        
        alternatives = st.session_state["category_alternatives"]
        for idx, alt in enumerate(alternatives, 1):
            with st.expander(f"📋 案{idx}: {alt['perspective']}", expanded=(idx == 1)):
                st.markdown(f"**哲学**: {alt['philosophy']}")
                st.markdown(f"**カテゴリ数**: {len(alt['categories'])}")
                st.markdown(f"**カテゴリ**: {', '.join(alt['categories'])}")
                
                if st.button(f"この案を採用", key=f"adopt_alt_{idx}"):
                    SessionManager.set_functional_categories(alt['categories'])
                    st.success("カテゴリを設定しました！")
                    st.rerun()
    
    st.divider()
    
    categories = SessionManager.get_functional_categories()
    if categories:
        st.header("機能カテゴリ一覧")
        
        categories_metadata = st.session_state.get("categories_metadata", {})
        
        if categories_metadata:
            st.info(
                f"抽出された{len(categories)}個のカテゴリ（詳細情報付き）を確認できます。"
            )
            
            with st.expander("📋 プロセス機能の詳細情報", expanded=False):
                for cat_name in categories:
                    if cat_name in categories_metadata:
                        meta = categories_metadata[cat_name]
                        with st.container():
                            col_name, col_imp = st.columns([3, 1])
                            with col_name:
                                st.markdown(f"### {cat_name}")
                            with col_imp:
                                importance = meta.get("importance", 3)
                                st.markdown(f"重要度: {'⭐' * importance}")
                            
                            if meta.get("description"):
                                st.markdown(f"**説明:** {meta['description']}")
                            
                            if meta.get("inputs") or meta.get("outputs"):
                                col_in, col_out = st.columns(2)
                                with col_in:
                                    if meta.get("inputs"):
                                        inputs_str = "、".join(meta["inputs"][:3])
                                        st.caption(f"📥 **インプット:** {inputs_str}")
                                with col_out:
                                    if meta.get("outputs"):
                                        outputs_str = "、".join(meta["outputs"][:2])
                                        st.caption(f"📤 **アウトプット:** {outputs_str}")
                            
                            if meta.get("examples"):
                                examples_str = "、".join(meta["examples"][:3])
                                st.caption(f"🔧 **具体例:** {examples_str}")
                            
                            st.divider()
                    else:
                        st.markdown(f"### {cat_name}")
                        st.caption("（詳細情報なし）")
                        st.divider()
        
        st.subheader("カテゴリの編集")
        st.info("行の追加・削除・編集が可能です。")
        
        df = pd.DataFrame({"カテゴリ名": categories})
        
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "カテゴリ名": st.column_config.TextColumn(
                    "カテゴリ名",
                    help="プロセスの機能的な側面を表すカテゴリ",
                    max_chars=50,
                    required=True,
                )
            },
        )
        
        updated_categories = edited_df["カテゴリ名"].dropna().tolist()
        updated_categories = [cat.strip() for cat in updated_categories if cat.strip()]
        
        if updated_categories != categories:
            SessionManager.set_functional_categories(updated_categories)
        
        if updated_categories:
            st.success(f"✅ 現在のカテゴリ数: {len(updated_categories)}")
            st.info("次のタブ「ノード定義」に進んでください")
    else:
        st.warning("⚠️ カテゴリを抽出してください")


def tab3_node_definition():
    """タブ3: ノード定義（IDEF0）"""
    st.header("🔧 ステップ3: ノード定義（IDEF0形式）")
    
    categories = SessionManager.get_functional_categories()
    
    if not categories:
        st.warning("⚠️ 先にタブ2で機能カテゴリを定義してください")
        return
    
    st.markdown("""
    各機能カテゴリに対して、具体的なノード（工程、道具、材料、スキルなど）を定義します。
    IDEF0形式: **Input（材料・情報）** → **Mechanism（手段）** → **Output（成果物）**
    """)
    
    generation_mode = st.radio(
        "生成モード",
        ["AI主導対話", "多様性生成（Verbalized Sampling）", "Zigzagging粒度調整"],
        horizontal=True,
        help="AI主導対話：全カテゴリをソクラテス式対話で生成 / 多様性生成：複数の異なる視点から一度に生成 / Zigzagging粒度調整：既存ノードを段階的に細分化"
    )
    
    st.divider()
    
    col_main, col_nodes = st.columns([2, 1])
    
    with col_nodes:
        st.subheader("📋 抽出されたノード (IDEF0形式)")
        
        all_idef0 = SessionManager.get_all_idef0_nodes()
        
        if all_idef0:
            selected_cat = st.selectbox(
                "カテゴリを選択",
                options=list(all_idef0.keys()),
                key="idef0_category_selector"
            )
            
            if selected_cat and selected_cat in all_idef0:
                idef0_data = all_idef0[selected_cat]
                with st.container(border=True):
                    st.markdown(f"**{selected_cat}**")
                    
                    if idef0_data.get("inputs"):
                        st.markdown("**📥 Input:**")
                        for inp in idef0_data["inputs"]:
                            st.write(f"  • {inp}")
                    
                    if idef0_data.get("mechanisms"):
                        st.markdown("**🔧 Mechanism:**")
                        for mech in idef0_data["mechanisms"]:
                            st.write(f"  • {mech}")
                    
                    if idef0_data.get("outputs"):
                        st.markdown("**📤 Output:**")
                        for out in idef0_data["outputs"]:
                            st.write(f"  • {out}")
                    
                    if not any([idef0_data.get("inputs"), idef0_data.get("mechanisms"), idef0_data.get("outputs")]):
                        st.caption("まだ抽出されていません")
        else:
            st.info("会話が進むと、IDEF0形式でノードが自動的に抽出されます")
    
    with col_main:
        st.info(f"💡 全{len(categories)}個のカテゴリを一括で議論・生成します")
        
        if generation_mode == "AI主導対話":
            st.caption("🎯🔬👤 ソクラテス式AI対話（全カテゴリ一括）")
            
            messages = SessionManager.get_messages()

            if not messages:
                llm_client = LLMClient()
                initial_message = llm_client.generate_initial_facilitator_message(
                    SessionManager.get_process_name(), categories
                )
                SessionManager.add_message("facilitator", initial_message)
                st.rerun()

            with st.container(height=400):
                for message in messages:
                    role = message["role"]
                    
                    if role == "facilitator":
                        with st.chat_message("assistant", avatar="🎯"):
                            st.markdown(f"**[ファシリテーター]**\n\n{message['content']}")
                    elif role == "expert":
                        with st.chat_message("assistant", avatar="🔬"):
                            st.markdown(f"**[エキスパート]**\n\n{message['content']}")
                    else:
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(message['content'])

            st.divider()
            
            col_btn1, col_btn2 = st.columns([3, 1])
            
            with col_btn1:
                if st.button("💭 会話を進める", type="primary", use_container_width=True, help="AIたちが全カテゴリを議論します"):
                    try:
                        llm_client = LLMClient()
                        
                        with st.spinner("🎯🔬 AIたちが議論中..."):
                            discussion = llm_client.generate_ai_discussion(
                                process_name=SessionManager.get_process_name(),
                                categories=categories,
                                chat_history=messages,
                            )
                        
                        for msg in discussion:
                            SessionManager.add_message(msg["role"], msg["content"])
                        
                        with st.spinner("📋 IDEF0形式でノードを自動抽出中..."):
                            all_idef0_nodes = llm_client.extract_all_idef0_nodes_from_chat(
                                process_name=SessionManager.get_process_name(),
                                process_description=SessionManager.get_process_description(),
                                categories=categories,
                                chat_history=SessionManager.get_messages(),
                            )
                            
                            for cat_name, idef0_node in all_idef0_nodes.items():
                                SessionManager.set_idef0_node(cat_name, idef0_node.model_dump())
                        
                        st.rerun()
                    
                    except OpenAIError as e:
                        st.error(f"OpenAI APIエラー: {str(e)}")
                    except Exception as e:
                        st.error(f"エラー: {str(e)}")
            
            with col_btn2:
                if st.button("🔄 リセット", use_container_width=True, help="対話をリセット"):
                    SessionManager.clear_messages()
                    if "categories_changed" in st.session_state:
                        st.session_state.categories_changed = False
                    st.rerun()
            
            user_input = st.chat_input("💬 あなたの知識や意見を入力してください（任意）...")
            
            if user_input:
                SessionManager.add_message("user", user_input)

                try:
                    llm_client = LLMClient()
                    
                    with st.spinner("🔬 エキスパートAIが応答中..."):
                        expert_response = llm_client.generate_expert_response(
                            process_name=SessionManager.get_process_name(),
                            categories=categories,
                            chat_history=messages,
                            user_message=user_input,
                        )

                    SessionManager.add_message("expert", expert_response)

                    messages_with_expert = SessionManager.get_messages()

                    with st.spinner("🎯 ファシリテーターAIが応答中..."):
                        facilitator_response = llm_client.generate_facilitator_response(
                            process_name=SessionManager.get_process_name(),
                            categories=categories,
                            chat_history=messages_with_expert,
                        )

                    SessionManager.add_message("facilitator", facilitator_response)
                    
                    with st.spinner("📋 IDEF0形式でノードを自動抽出中..."):
                        all_idef0_nodes = llm_client.extract_all_idef0_nodes_from_chat(
                            process_name=SessionManager.get_process_name(),
                            process_description=SessionManager.get_process_description(),
                            categories=categories,
                            chat_history=SessionManager.get_messages(),
                        )
                        
                        for cat_name, idef0_node in all_idef0_nodes.items():
                            SessionManager.set_idef0_node(cat_name, idef0_node.model_dump())

                    st.rerun()

                except OpenAIError as e:
                    st.error(f"OpenAI APIエラー: {str(e)}")
                except Exception as e:
                    st.error(f"エラー: {str(e)}")
        
        elif generation_mode == "多様性生成（Verbalized Sampling）":
            st.caption("🎲 Verbalized Sampling - 全カテゴリ一括生成（段階的生成）")
            
            # 視点数選択スライダー
            num_perspectives = st.slider(
                "生成する視点の数",
                min_value=1,
                max_value=5,
                value=3,
                help="1視点: 最速（約30秒）、3視点: 推奨バランス、5視点: 最大多様性"
            )
            
            st.info(f"💡 {num_perspectives}つの視点を順次生成します。各視点の進捗が表示されます。")
            
            if st.button("🎲 多様な視点で生成", type="primary", use_container_width=True, help=f"{num_perspectives}つの異なる思考モードから全カテゴリを生成"):
                try:
                    llm_client = LLMClient()
                    
                    # プログレスバー用のプレースホルダー
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    # プログレスコールバック関数
                    def update_progress(current, total, perspective_name):
                        progress = (current + 1) / total
                        progress_bar.progress(progress)
                        status_text.text(f"🎲 視点 {current + 1}/{total} ({perspective_name}) を生成中...")
                    
                    # 段階的生成
                    perspectives = llm_client.generate_diverse_idef0_nodes_all_categories(
                        process_name=SessionManager.get_process_name(),
                        process_description=SessionManager.get_process_description(),
                        categories=categories,
                        num_perspectives=num_perspectives,
                        progress_callback=update_progress,
                    )
                    
                    # 完了
                    progress_bar.progress(1.0)
                    status_text.text(f"✅ {num_perspectives}つの視点の生成が完了しました")
                    
                    if perspectives:
                        st.session_state.diverse_perspectives_all = perspectives
                        st.success(f"{len(perspectives)}つの異なる視点を生成しました！")
                    else:
                        st.error("視点の生成に失敗しました。")
                        st.warning("💡 ターミナルログを確認してください。詳細なエラー情報が出力されています。")
                        with st.expander("🔍 トラブルシューティング"):
                            st.markdown("""
                            **考えられる原因:**
                            1. LLMモデルが期待するJSON形式で応答していない
                            2. プロセス概要が不足している、または複雑すぎる
                            3. カテゴリ数が多すぎる（推奨: 5-8個）
                            
                            **対処方法:**
                            - ターミナルでStreamlitを起動した場所で詳細ログを確認
                            - プロセス概要をより具体的に記述
                            - カテゴリ数を減らす
                            - モデルを変更（gpt-4o、gpt-4-turboなど）
                            """)
                
                except OpenAIError as e:
                    st.error(f"OpenAI APIエラー: {str(e)}")
                    with st.expander("詳細を表示"):
                        st.exception(e)
                except Exception as e:
                    st.error(f"エラー: {str(e)}")
                    with st.expander("詳細を表示"):
                        st.exception(e)
            
            if "diverse_perspectives_all" in st.session_state and st.session_state.diverse_perspectives_all:
                st.markdown("### 📊 生成された視点の比較")
                
                perspectives = st.session_state.diverse_perspectives_all
                
                tabs = st.tabs([f"{p['perspective']} ({p['probability']:.2f})" for p in perspectives])
                
                for idx, (tab, persp) in enumerate(zip(tabs, perspectives)):
                    with tab:
                        st.info(persp['description'])
                        
                        if 'idef0_nodes' in persp:
                            cat_tabs = st.tabs(list(persp['idef0_nodes'].keys()))
                            
                            for cat_tab, (cat_name, idef0_data) in zip(cat_tabs, persp['idef0_nodes'].items()):
                                with cat_tab:
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.markdown("**📥 Input:**")
                                        for inp in idef0_data.get('inputs', []):
                                            st.write(f"• {inp}")
                                    
                                    with col2:
                                        st.markdown("**🔧 Mechanism:**")
                                        for mech in idef0_data.get('mechanisms', []):
                                            st.write(f"• {mech}")
                                    
                                    with col3:
                                        st.markdown("**📤 Output:**")
                                        for out in idef0_data.get('outputs', []):
                                            st.write(f"• {out}")
                        
                        if st.button(f"この視点を採用", key=f"adopt_all_{idx}", type="primary", use_container_width=True):
                            if 'idef0_nodes' in persp:
                                for cat_name, idef0_data in persp['idef0_nodes'].items():
                                    SessionManager.set_idef0_node(cat_name, idef0_data)
                            st.success(f"✅ 『{persp['perspective']}』を採用しました！ノードが更新されました。")
                            st.info("💡 ステップ4でノード影響評価に進んでください。")
        
        elif generation_mode == "Zigzagging粒度調整":
            st.caption("🔍 既存のIDEF0ノードを段階的に細分化")
            
            all_idef0 = SessionManager.get_all_idef0_nodes()
            
            if not all_idef0:
                st.warning("⚠️ まず「AI主導対話」または「多様性生成」でノードを生成してください")
                st.info("""
                **Zigzagging粒度調整の使い方:**
                1. 最初に「AI主導対話」または「多様性生成」でIDEF0ノードを生成
                2. 分析結果を見て「粒度が粗い」と感じたカテゴリを選択
                3. Zigzagging手法で段階的に細分化
                4. 細分化後のノードで再評価・再分析
                """)
                return
            
            if "selected_refinement_node" in st.session_state and st.session_state.selected_refinement_node:
                selected_node = st.session_state.selected_refinement_node
                
                found_category = None
                for cat_name, idef0_data in all_idef0.items():
                    if selected_node in idef0_data.get("inputs", []) or \
                       selected_node in idef0_data.get("mechanisms", []) or \
                       selected_node in idef0_data.get("outputs", []):
                        found_category = cat_name
                        break
                
                if found_category:
                    st.success(f"💡 タブ5から「{selected_node}」の細分化が提案されました")
                    st.info(f"📍 該当カテゴリ: **{found_category}**")
                    st.markdown("---")
            
            st.markdown("""
            **反復的な知識精緻化プロセス**
            
            分析結果（ヒートマップ、PageRank、DSM最適化など）を見て粒度が粗いと感じた場合、
            このモードで段階的に細分化できます。
            
            - **Output**: より細かい性能指標・品質要素に分解
            - **Mechanism**: より細かい作業手順・ステップに分解
            - **Input**: より細かい構成要素に分解
            """)
            
            default_index = 0
            if "selected_refinement_node" in st.session_state and st.session_state.selected_refinement_node:
                selected_node = st.session_state.selected_refinement_node
                for i, cat_name in enumerate(all_idef0.keys()):
                    idef0_data = all_idef0[cat_name]
                    if selected_node in idef0_data.get("inputs", []) or \
                       selected_node in idef0_data.get("mechanisms", []) or \
                       selected_node in idef0_data.get("outputs", []):
                        default_index = i
                        break
            
            selected_category = st.selectbox(
                "細分化するカテゴリを選択",
                options=list(all_idef0.keys()),
                index=default_index,
                key="zigzag_category_select"
            )
            
            current_idef0 = all_idef0.get(selected_category, {})
            
            st.markdown(f"### 📋 現在のIDEF0ノード: **{selected_category}**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📥 Input（材料・情報）:**")
                inputs = current_idef0.get("inputs", [])
                if inputs:
                    for inp in inputs:
                        st.write(f"• {inp}")
                else:
                    st.caption("（なし）")
                st.metric("要素数", len(inputs))
            
            with col2:
                st.markdown("**🔧 Mechanism（手段・手順）:**")
                mechanisms = current_idef0.get("mechanisms", [])
                if mechanisms:
                    for mech in mechanisms:
                        st.write(f"• {mech}")
                else:
                    st.caption("（なし）")
                st.metric("要素数", len(mechanisms))
            
            with col3:
                st.markdown("**📤 Output（性能・成果物）:**")
                outputs = current_idef0.get("outputs", [])
                if outputs:
                    for out in outputs:
                        st.write(f"• {out}")
                else:
                    st.caption("（なし）")
                st.metric("要素数", len(outputs))
            
            st.divider()
            
            refinement_depth = st.slider(
                "細分化の深さ",
                min_value=1,
                max_value=3,
                value=1,
                help="1: 軽度（各要素を2-3個に分解） / 2: 中程度（3-5個に分解） / 3: 詳細（5-7個に分解）"
            )
            
            depth_labels = {
                1: "🌱 軽度：各要素を2-3個の下位要素に分解",
                2: "🌿 中程度：各要素を3-5個の下位要素に詳細分解",
                3: "🌳 詳細：各要素を5-7個の下位要素に徹底的に分解"
            }
            st.info(depth_labels[refinement_depth])
            
            if st.button("🔄 Zigzaggingで細分化", type="primary", use_container_width=True):
                try:
                    llm_client = LLMClient()
                    
                    with st.spinner(f"🤖 AIが「{selected_category}」を細分化中..."):
                        refined_idef0 = llm_client.refine_idef0_with_zigzagging(
                            process_name=SessionManager.get_process_name(),
                            category=selected_category,
                            current_idef0=current_idef0,
                            refinement_depth=refinement_depth
                        )
                    
                    st.session_state.refined_idef0_preview = refined_idef0
                    st.success("✅ 細分化が完了しました！プレビューを確認してください。")
                    
                except OpenAIError as e:
                    st.error(f"OpenAI APIエラー: {str(e)}")
                    with st.expander("詳細を表示"):
                        st.exception(e)
                except Exception as e:
                    st.error(f"エラー: {str(e)}")
                    with st.expander("詳細を表示"):
                        st.exception(e)
            
            if "refined_idef0_preview" in st.session_state and st.session_state.refined_idef0_preview:
                refined = st.session_state.refined_idef0_preview
                
                st.divider()
                st.markdown(f"### 📊 細分化後のIDEF0ノード: **{refined.get('category', selected_category)}**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📥 Input（細分化後）:**")
                    refined_inputs = refined.get("inputs", [])
                    for inp in refined_inputs:
                        st.write(f"• {inp}")
                    
                    original_count = len(current_idef0.get("inputs", []))
                    new_count = len(refined_inputs)
                    delta = new_count - original_count
                    st.metric("要素数", new_count, delta=delta)
                
                with col2:
                    st.markdown("**🔧 Mechanism（細分化後）:**")
                    refined_mechs = refined.get("mechanisms", [])
                    for mech in refined_mechs:
                        st.write(f"• {mech}")
                    
                    original_count = len(current_idef0.get("mechanisms", []))
                    new_count = len(refined_mechs)
                    delta = new_count - original_count
                    st.metric("要素数", new_count, delta=delta)
                
                with col3:
                    st.markdown("**📤 Output（細分化後）:**")
                    refined_outputs = refined.get("outputs", [])
                    for out in refined_outputs:
                        st.write(f"• {out}")
                    
                    original_count = len(current_idef0.get("outputs", []))
                    new_count = len(refined_outputs)
                    delta = new_count - original_count
                    st.metric("要素数", new_count, delta=delta)
                
                st.markdown("---")
                
                col_apply, col_cancel = st.columns([1, 1])
                
                with col_apply:
                    if st.button("✅ この細分化を適用", type="primary", use_container_width=True):
                        SessionManager.set_idef0_node(refined['category'], refined)
                        del st.session_state.refined_idef0_preview
                        st.success(f"✅ 「{refined['category']}」を細分化しました！ノードが更新されました。")
                        st.info("💡 ステップ4でノード影響評価を再実行してください。")
                
                with col_cancel:
                    if st.button("❌ キャンセル", use_container_width=True):
                        del st.session_state.refined_idef0_preview
                        st.info("キャンセルしました。")

    nodes = SessionManager.get_nodes()
    if nodes:
        st.divider()
        st.header("定義されたノード一覧")
        st.info(
            f"抽出された{len(nodes)}個のノードを確認・編集できます。"
            "行の追加・削除・編集が可能です。"
        )

        df_nodes = pd.DataFrame({"ノード名": nodes})

        edited_nodes_df = st.data_editor(
            df_nodes,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "ノード名": st.column_config.TextColumn(
                    "ノード名",
                    help="プロセスの構成要素（工程、道具、材料など）",
                    max_chars=100,
                    required=True,
                )
            },
        )

        updated_nodes = edited_nodes_df["ノード名"].dropna().tolist()
        updated_nodes = [node.strip() for node in updated_nodes if node.strip()]

        if updated_nodes != nodes:
            SessionManager.set_nodes(updated_nodes)

        if updated_nodes:
            st.caption(f"現在のノード数: {len(updated_nodes)}")
            st.success("✅ ノード定義が完了しました")
            st.info("次のタブ「ノード影響評価」に進んでください")




def tab4_node_evaluation():
    """タブ4: ノード影響評価（機能カテゴリベース行列評価）"""
    
    st.header("⚖️ ステップ4: ノード間影響評価（機能カテゴリベース行列評価）")
    
    nodes = SessionManager.get_nodes()
    idef0_nodes = SessionManager.get_all_idef0_nodes()
    process_name = SessionManager.get_process_name()
    categories = SessionManager.get_functional_categories()
    
    if not nodes or len(nodes) < 2:
        st.warning("⚠️ 先にタブ3でノードを2つ以上定義してください")
        return
    
    if not idef0_nodes:
        st.warning("⚠️ 先にタブ3でIDEF0ノードを定義してください")
        return
    
    st.markdown("""
    ## 🎓 評価ロジック
    
    **時系列順カテゴリを活用した3フェーズ段階的評価**
    
    ### フェーズ1: 同一カテゴリ内評価（距離0）
    - **目的**: 各カテゴリ内部のn×n行列を評価
    - **特徴**: ナレッジなし（初回評価）、対角線=0
    - **評価対象**: 内部依存関係のみ
    
    ### フェーズ2: 隣接カテゴリ間評価（距離1）
    - **目的**: カテゴリA→Bのn×m行列を評価
    - **特徴**: フェーズ1の非ゼロ評価をナレッジとして活用
    - **評価対象**: 前工程の成果物が次工程に与える影響
    
    ### フェーズ3: 遠距離評価（距離2+、オプション）
    - **目的**: カテゴリA→Cのn×m行列を評価
    - **特徴**: A→B→Cの中間パスナレッジを活用
    - **評価対象**: 推移的影響の論理的評価
    
    **評価スケール**: ±0, ±1, ±3, ±9（明確な判断のため中間値削除）
    
    **トークン削減効果**: 行列形式により60-80%削減
    """)
    
    if "matrix_evaluator" not in st.session_state:
        st.session_state.matrix_evaluator = None
    if "evaluation_plans" not in st.session_state:
        st.session_state.evaluation_plans = []
    if "current_phase" not in st.session_state:
        st.session_state.current_phase = 0
    if "completed_plans" not in st.session_state:
        st.session_state.completed_plans = set()
    
    st.markdown("---")
    st.subheader("📋 ステップ1: 評価計画の作成")
    
    st.markdown(f"""
    **現在のノード数**: {len(nodes)}個
    **カテゴリ数**: {len(categories)}個
    **カテゴリ**: {', '.join(categories)}
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        max_distance = st.selectbox(
            "評価する最大カテゴリ間距離",
            options=[0, 1, 2],
            index=1,
            help="0=同一カテゴリのみ、1=隣接まで（推奨）、2=遠距離を含む"
        )
    with col2:
        enable_distant = st.checkbox(
            "遠距離評価を有効化（距離2+）",
            value=False,
            help="フェーズ3を実行（LLM呼び出しが大幅に増加）",
            disabled=(max_distance < 2)
        )
    
    if st.button("🔄 評価計画を作成", type="primary", key="create_plan_btn"):
        try:
            evaluator = MatrixEvaluator(categories, idef0_nodes, nodes)
            plans = evaluator.plan_evaluation_phases(
                max_distance=max_distance,
                enable_distant=enable_distant
            )
            
            st.session_state.matrix_evaluator = evaluator
            st.session_state.evaluation_plans = plans
            st.session_state.current_phase = 0
            st.session_state.completed_plans = set()
            
            SessionManager.get_project_data()["evaluations"] = []
            
            summary = evaluator.get_phase_summary(plans)
            
            st.success(f"✅ 評価計画を作成しました（全{summary['total_plans']}件）")
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                phase1 = summary["phase_1_same"]
                st.metric(
                    "フェーズ1（同一カテゴリ）",
                    f"{phase1['count']}件",
                    delta=f"{phase1['total_pairs']}ペア"
                )
            with col_s2:
                phase2 = summary["phase_2_adjacent"]
                st.metric(
                    "フェーズ2（隣接カテゴリ）",
                    f"{phase2['count']}件",
                    delta=f"{phase2['total_pairs']}ペア"
                )
            with col_s3:
                phase3 = summary["phase_3_distant"]
                st.metric(
                    "フェーズ3（遠距離）",
                    f"{phase3['count']}件",
                    delta=f"{phase3['total_pairs']}ペア"
                )
            
            st.info("💡 次は「ステップ2: 段階的評価実行」に進んでください。")
            
        except Exception as e:
            st.error(f"❌ 評価計画作成エラー: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    if not st.session_state.evaluation_plans:
        st.info("ℹ️ 評価計画を作成してください。")
        return
    
    st.markdown("---")
    st.subheader("🚀 ステップ2: 段階的評価実行")
    
    plans = st.session_state.evaluation_plans
    evaluator = st.session_state.matrix_evaluator
    completed = st.session_state.completed_plans
    
    remaining_plans = [p for i, p in enumerate(plans) if i not in completed]
    
    if not remaining_plans:
        st.success("✅ 全ての評価が完了しました！")
        st.info("👉 下の「ステップ3: 評価結果確認」で詳細を確認できます。")
    else:
        st.markdown(f"""
        **進捗**: {len(completed)} / {len(plans)} 完了
        
        次に評価する行列を選択してください。
        """)
        
        phase_groups = {}
        for i, plan in enumerate(plans):
            if i in completed:
                continue
            phase_idx = plan["phase_index"]
            if phase_idx not in phase_groups:
                phase_groups[phase_idx] = []
            phase_groups[phase_idx].append((i, plan))
        
        first_uncompleted_phase = min(phase_groups.keys()) if phase_groups else 1
        
        for phase_idx in sorted(phase_groups.keys()):
            phase_plans = phase_groups[phase_idx]
            phase_name = {
                1: "フェーズ1: 同一カテゴリ内",
                2: "フェーズ2: 隣接カテゴリ間",
                3: "フェーズ3: 遠距離"
            }[phase_idx]
            
            st.markdown(f"### 📊 {phase_name} ({len(phase_plans)}件)")
            
            st.info(f"💡 ナレッジベース: {len(evaluator.knowledge_base)}件の非ゼロ評価を参照可能")
            
            if st.button(f"🚀 {phase_name}を全て評価", type="primary", key=f"batch_eval_phase_{phase_idx}"):
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    total = len(phase_plans)
                    success_count = 0
                    
                    for idx, (plan_idx, plan) in enumerate(phase_plans):
                        status_text.text(f"評価中: {idx + 1}/{total} - {plan['from_category']} → {plan['to_category']}")
                        
                        try:
                            _execute_matrix_evaluation(
                                plan_idx,
                                plan,
                                evaluator,
                                idef0_nodes,
                                process_name
                            )
                            success_count += 1
                        except Exception as e:
                            st.error(f"❌ エラー: {plan['from_category']} → {plan['to_category']}: {str(e)}")
                        
                        progress_bar.progress((idx + 1) / total)
                    
                    status_text.text("")
                    progress_bar.empty()
                    
                    st.success(f"✅ {phase_name}の評価が完了しました！（{success_count}/{total}件成功）")
                    st.rerun()
            
            with st.expander(f"📋 個別評価 ({len(phase_plans)}件)", expanded=(phase_idx == first_uncompleted_phase)):
                for plan_idx, plan in phase_plans:
                    col_info, col_action = st.columns([3, 1])
                    
                    with col_info:
                        n, m = plan["matrix_size"]
                        total_pairs = n * (n - 1) if plan["distance"] == 0 else n * m
                        
                        st.markdown(f"""
                        **{plan['from_category']} → {plan['to_category']}**  
                        行列サイズ: {n}×{m} ({total_pairs}ペア)  
                        カテゴリ間距離: {plan['distance']}
                        """)
                    
                    with col_action:
                        if st.button("評価", key=f"eval_plan_{plan_idx}"):
                            _execute_matrix_evaluation(
                                plan_idx,
                                plan,
                                evaluator,
                                idef0_nodes,
                                process_name
                            )
                            st.rerun()
    
    st.markdown("---")
    st.subheader("✅ ステップ3: 評価結果確認")
    
    evaluations = SessionManager.get_evaluations()
    
    if not evaluations:
        st.warning("⚠️ 評価結果がありません。")
        return
    
    st.success(f"🎉 全{len(evaluations)}件の評価が完了しました！")
    
    non_zero_evals = [e for e in evaluations if e.get("score", 0) != 0]
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("非ゼロ評価ペア", f"{len(non_zero_evals)} / {len(evaluations)}")
    with col_m2:
        sparsity = 100 * (1 - len(non_zero_evals) / len(evaluations)) if evaluations else 0
        st.metric("疎行列率", f"{sparsity:.1f}%")
    
    if non_zero_evals:
        with st.expander("🔥 高スコアペア（|score| ≥ 5）", expanded=True):
            high_score_evals = [e for e in non_zero_evals if abs(e.get("score", 0)) >= 5]
            
            if high_score_evals:
                high_score_evals_sorted = sorted(
                    high_score_evals,
                    key=lambda x: abs(x.get("score", 0)),
                    reverse=True
                )
                
                for eval_item in high_score_evals_sorted[:20]:
                    score = eval_item.get("score", 0)
                    score_color = "green" if score > 0 else "red"
                    
                    st.markdown(
                        f"**{eval_item['from_node']}** → **{eval_item['to_node']}**: "
                        f":{score_color}[{score:+d}]"
                    )
            else:
                st.info("スコア絶対値5以上のペアはありません。")
    
    st.markdown("---")
    st.markdown("### 次のステップ")
    st.info("👉 **タブ5** で隣接行列とヒートマップを確認できます。")
    
    st.markdown("---")
    st.subheader("🗑️ リセット")
    
    if st.button("🔄 評価計画をリセット", key="reset_plan_btn"):
        st.session_state.matrix_evaluator = None
        st.session_state.evaluation_plans = []
        st.session_state.current_phase = 0
        st.session_state.completed_plans = set()
        SessionManager.get_project_data()["evaluations"] = []
        st.info("🔄 評価計画をリセットしました。「ステップ1」から再実行してください。")
        st.rerun()


def _execute_matrix_evaluation(
    plan_idx: int,
    plan: dict,
    evaluator: MatrixEvaluator,
    idef0_nodes: dict,
    process_name: str
):
    """
    行列評価を実行
    
    Args:
        plan_idx: 評価計画のインデックス
        plan: 評価計画
        evaluator: MatrixEvaluatorインスタンス
        idef0_nodes: IDEF0ノードデータ
        process_name: プロセス名
    """
    try:
        from_category = plan["from_category"]
        to_category = plan["to_category"]
        from_nodes = plan["from_nodes"]
        to_nodes = plan["to_nodes"]
        distance = plan["distance"]
        
        idef0_from = idef0_nodes.get(from_category, {})
        idef0_to = idef0_nodes.get(to_category, {})
        
        knowledge = evaluator.extract_knowledge_for_plan(plan, top_k=10)
        
        with st.spinner(f"🤖 LLMが{plan['matrix_size'][0]}×{plan['matrix_size'][1]}行列を評価中..."):
            st.caption(f"参考評価: {len(knowledge)}件")
            
            if knowledge:
                with st.expander("参考にした評価", expanded=False):
                    for k in knowledge:
                        sign = "+" if k["score"] > 0 else ""
                        st.caption(f"{k['from_node']} → {k['to_node']}: {sign}{k['score']}")
            
            llm_client = LLMClient()
            
            matrix = llm_client.evaluate_matrix_with_knowledge(
                from_category=from_category,
                to_category=to_category,
                from_nodes=from_nodes,
                to_nodes=to_nodes,
                idef0_from=idef0_from,
                idef0_to=idef0_to,
                process_name=process_name,
                knowledge=knowledge,
                distance=distance,
                categories=evaluator.categories
            )
        
        # 行列形式で一括保存（メモリ効率向上）
        SessionManager.save_evaluation_matrix(
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            matrix=matrix,
            from_category=from_category,
            to_category=to_category
        )
        
        # evaluatorのナレッジベース更新（非ゼロのみ）
        for i, from_node in enumerate(from_nodes):
            for j, to_node in enumerate(to_nodes):
                score = matrix[i][j]
                if score != 0:  # 非ゼロのみナレッジベースに追加
                    evaluator.add_evaluation_result(from_node, to_node, score)
        
        st.session_state.completed_plans.add(plan_idx)
        
        non_zero_count = sum(1 for row in matrix for val in row if val != 0)
        total_count = len(from_nodes) * len(to_nodes)
        
        st.success(
            f"✅ 評価完了！非ゼロ: {non_zero_count}/{total_count}ペア "
            f"({100 * non_zero_count / total_count:.1f}%)"
        )
        
    except Exception as e:
        st.error(f"❌ 評価エラー: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
def tab5_matrix_analysis():
    """タブ5: 行列分析とヒートマップ可視化"""
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    st.header("📈 ステップ5: 隣接行列とヒートマップ")
    
    nodes = SessionManager.get_nodes()
    evaluations = SessionManager.get_evaluations()
    
    if not nodes or len(nodes) < 2:
        st.warning("⚠️ 先にタブ3でノードを2つ以上定義してください")
        return
    
    if not evaluations:
        st.warning("⚠️ 先にタブ4でノード間の評価を完了してください")
        st.info("""
        **ワークフロー:**
        1. タブ4で評価ペアを生成
        2. 各ペアをLLMで評価
        3. このタブで行列を生成・可視化
        """)
        return
    
    st.markdown("""
    評価結果を隣接行列に変換し、ヒートマップで可視化します。
    
    - **行**: 評価元ノード（From）
    - **列**: 評価先ノード（To）
    - **セル値**: 影響スコア（-9～+9）
    """)
    
    st.subheader("📊 隣接行列の生成")
    
    if st.button("🔄 行列を生成して可視化", type="primary", use_container_width=True):
        try:
            df_evals = pd.DataFrame(evaluations)
            
            pivot_matrix = df_evals.pivot_table(
                index='from_node',
                columns='to_node',
                values='score',
                fill_value=0
            )
            
            adj_matrix_df = pivot_matrix.reindex(
                index=nodes,
                columns=nodes,
                fill_value=0
            )
            
            adj_matrix_np = adj_matrix_df.values
            
            SessionManager.set_adjacency_matrix(adj_matrix_np)
            
            st.session_state.adj_matrix_df = adj_matrix_df
            
            st.success(f"✅ {len(nodes)}×{len(nodes)}の隣接行列を生成しました")
            st.info("💡 下のヒートマップで可視化されています。ステップ6でネットワーク可視化に進んでください。")
            
        except Exception as e:
            st.error(f"❌ 行列生成エラー: {str(e)}")
            with st.expander("詳細を表示"):
                st.exception(e)
    
    if "adj_matrix_df" in st.session_state and st.session_state.adj_matrix_df is not None:
        adj_matrix_df = st.session_state.adj_matrix_df
        
        st.divider()
        st.subheader("🎨 ヒートマップ可視化")
        
        # 日本語フォント設定（文字化け防止）
        import japanize_matplotlib
        japanize_matplotlib.japanize()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            adj_matrix_df,
            annot=True,
            fmt='.0f',
            cmap='coolwarm',
            center=0,
            vmin=-9,
            vmax=9,
            linewidths=0.5,
            cbar_kws={'label': 'Influence Score'},
            ax=ax
        )
        
        ax.set_title('Node Influence Heatmap', fontsize=16, pad=20)
        ax.set_xlabel('To Node', fontsize=12)
        ax.set_ylabel('From Node', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()
        
        # adjacency_matrixをnumpy配列としても保存（ステップ6で使用）
        st.session_state.adjacency_matrix = adj_matrix_df.values
        
        st.divider()
        st.subheader("📋 隣接行列データ")
        
        st.dataframe(
            adj_matrix_df,
            use_container_width=True,
            height=400
        )
        
        non_zero_count = np.count_nonzero(adj_matrix_df.values)
        total_count = adj_matrix_df.shape[0] * adj_matrix_df.shape[1]
        density = non_zero_count / total_count if total_count > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("非ゼロ要素数", non_zero_count)
        with col2:
            st.metric("総要素数", total_count)
        with col3:
            st.metric("行列密度", f"{density:.2%}")
        
        st.divider()
        st.subheader("🔍 粒度調整の提案")
        
        st.markdown("""
        **反復的知識精緻化プロセス**
        
        以下のノードは多くのノードと関係しているため、粒度が粗い可能性があります。
        Zigzagging手法で細分化することで、知識の解像度を向上できます。
        """)
        
        in_degrees = adj_matrix_df.sum(axis=0)
        out_degrees = adj_matrix_df.sum(axis=1)
        total_degrees = in_degrees + out_degrees
        
        degree_df = pd.DataFrame({
            'ノード名': nodes,
            '入次数（受ける影響の合計）': in_degrees.values,
            '出次数（与える影響の合計）': out_degrees.values,
            '総次数': total_degrees.values
        })
        
        degree_df = degree_df.sort_values('総次数', ascending=False, key=lambda x: abs(x))
        
        top_5 = degree_df.head(5)
        
        st.markdown("### 📊 粒度が粗い可能性のあるノード（トップ5）")
        
        for idx, row in top_5.iterrows():
            node_name = row['ノード名']
            total_deg = row['総次数']
            in_deg = row['入次数（受ける影響の合計）']
            out_deg = row['出次数（与える影響の合計）']
            
            with st.container(border=True):
                col_info, col_btn = st.columns([3, 1])
                
                with col_info:
                    st.markdown(f"**{node_name}**")
                    st.caption(f"総次数: {total_deg:.1f} (入: {in_deg:.1f}, 出: {out_deg:.1f})")
                    
                    if abs(total_deg) > 10:
                        st.warning("⚠️ 非常に多くのノードと関係 → 粒度が粗い可能性が高い")
                    elif abs(total_deg) > 5:
                        st.info("💡 複数のノードと関係 → 細分化を検討")
                
                with col_btn:
                    if st.button("🔄 細分化", key=f"refine_{node_name}", use_container_width=True):
                        st.session_state.selected_refinement_node = node_name
                        st.info(f"💡 「ステップ3: ノード定義」タブに移動して、「{node_name}」を含むカテゴリを細分化してください")
                        st.info("タブ3で「Zigzagging粒度調整」モードを選択してください")
    
    else:
        st.info("👆 「行列を生成して可視化」ボタンをクリックして開始してください")


def tab6_network_visualization():
    """タブ6: ネットワーク可視化（3D/2D）"""
    st.header("📊 ステップ6: ネットワーク可視化")
    
    nodes = SessionManager.get_nodes()
    
    if not nodes or len(nodes) < 2:
        st.warning("⚠️ 先にタブ3でノードを2つ以上定義してください")
        return
    
    st.markdown("""
    3D/2D空間でノード間の関係性を可視化します。
    """)
    
    # 隣接行列の確認（ステップ5で生成されたデータを使用）
    if "adjacency_matrix" not in st.session_state or st.session_state.adjacency_matrix is None:
        if "adj_matrix_df" in st.session_state and st.session_state.adj_matrix_df is not None:
            # DataFrameから変換
            st.session_state.adjacency_matrix = st.session_state.adj_matrix_df.values
            st.info("✅ ステップ5で生成された隣接行列を使用しています。")
        else:
            # デモ用のランダムデータ
            n = len(nodes)
            demo_matrix = np.random.randint(-5, 6, size=(n, n))
            np.fill_diagonal(demo_matrix, 0)
            st.session_state.adjacency_matrix = demo_matrix
            st.warning("⚠️ デモ用のランダムデータを表示しています。タブ4で評価を実行し、タブ5で隣接行列を生成してください。")
    else:
        st.success("✅ ステップ5で生成された隣接行列を使用しています。")
    
    viz_tab1, viz_tab2 = st.tabs(["🎮 3D可視化", "📊 2D可視化"])
    
    with viz_tab1:
        st.info("💡 3D空間でノード間の関係性を可視化します（要: 隣接行列データ）")
        
        if st.session_state.adjacency_matrix is not None:
            from utils.networkmaps_bridge import convert_pim_to_networkmaps
            from components.networkmaps_viewer import networkmaps_3d_viewer
            
            col_viewer, col_controls = st.columns([3, 1])
            
            with col_controls:
                st.subheader("表示設定")
                
                scale = st.slider(
                    "空間のスケール",
                    min_value=5.0,
                    max_value=20.0,
                    value=10.0,
                    step=1.0,
                    help="ノード間の距離を調整します"
                )
                
                camera_mode = st.radio(
                    "カメラモード",
                    options=["3d", "2d"],
                    format_func=lambda x: "3D視点" if x == "3d" else "2D俯瞰",
                    help="視点を切り替えます"
                )
                
                st.divider()
                st.caption("💡 操作方法")
                st.markdown("""
                **マウス操作:**
                - 🖱️ 左ドラッグ: 回転
                - 🖱️ ホイール: ズーム
                - 🖱️ 右ドラッグ: パン
                - 🖱️ クリック: ノード選択
                """)
        
            with col_viewer:
                try:
                    diagram_data = convert_pim_to_networkmaps(
                        nodes=nodes,
                        adjacency_matrix=st.session_state.adjacency_matrix,
                        categories=SessionManager.get_functional_categories(),
                        idef0_data=SessionManager.get_all_idef0_nodes(),
                        evaluations=st.session_state.get('evaluations', []),
                        scale=scale
                    )
                    
                    selected_node = networkmaps_3d_viewer(
                        diagram_data=diagram_data,
                        height=700,
                        enable_interaction=True,
                        camera_mode=camera_mode,
                        key="pim_network_3d_viewer"
                    )
                    
                    if selected_node:
                        st.success(f"**選択ノード:** {selected_node['node_name']}")
                        
                        st.markdown("### 🔍 詳細情報")
                        with st.container():
                            node_idx = nodes.index(selected_node['node_name'])
                            
                            st.markdown("**このノードからの影響:**")
                            outgoing = []
                            for j, target in enumerate(nodes):
                                score = st.session_state.adjacency_matrix[node_idx, j]
                                if score != 0:
                                    outgoing.append(f"→ {target}: **{score:+.1f}**")
                            
                            if outgoing:
                                for item in outgoing:
                                    st.markdown(item)
                            else:
                                st.caption("影響なし")
                            
                            st.divider()
                            
                            st.markdown("**このノードへの影響:**")
                            incoming = []
                            for i, source in enumerate(nodes):
                                score = st.session_state.adjacency_matrix[i, node_idx]
                                if score != 0:
                                    incoming.append(f"{source} →: **{score:+.1f}**")
                            
                            if incoming:
                                for item in incoming:
                                    st.markdown(item)
                            else:
                                st.caption("影響なし")
                
                except Exception as e:
                    st.error(f"3D可視化エラー: {str(e)}")
                    st.caption("**エラー詳細:**")
                    st.code(str(e), language="python")
    
    with viz_tab2:
        st.info("💡 2Dグラフでノード間の関係性を可視化します（Cytoscape.js）")
        
        if st.session_state.adjacency_matrix is not None:
            from utils.cytoscape_bridge import convert_pim_to_cytoscape
            from components.cytoscape_viewer import cytoscape_2d_viewer
            
            col_viewer2d, col_controls2d = st.columns([3, 1])
            
            with col_controls2d:
                st.subheader("表示設定")
                
                threshold = st.slider(
                    "スコア閾値",
                    min_value=0.0,
                    max_value=9.0,
                    value=2.0,
                    step=0.5,
                    help="この値以上のスコアを持つエッジのみ表示"
                )
                
                layout = st.selectbox(
                    "レイアウト",
                    options=["hierarchical", "cose", "breadthfirst", "circle", "grid"],
                    format_func=lambda x: {
                        "hierarchical": "階層的（3D構造準拠）",
                        "cose": "力学モデル",
                        "breadthfirst": "階層的（自動）",
                        "circle": "円形",
                        "grid": "グリッド"
                    }[x],
                    help="グラフのレイアウトアルゴリズム"
                )
                
                st.divider()
                st.caption("💡 操作方法")
                st.markdown("""
                **マウス操作:**
                - 🖱️ ドラッグ: パン
                - 🖱️ ホイール: ズーム
                - 🖱️ クリック: ノード選択
                
                **色の意味:**
                - 🟢 Output（成果物）
                - 🔵 Mechanism（手段）
                - 🟠 Input（材料・情報）
                """)
            
            with col_viewer2d:
                try:
                    cyto_data = convert_pim_to_cytoscape(
                        nodes=nodes,
                        adjacency_matrix=st.session_state.adjacency_matrix,
                        categories=SessionManager.get_functional_categories(),
                        idef0_data=SessionManager.get_all_idef0_nodes(),
                        threshold=threshold,
                        use_hierarchical_layout=(layout == "hierarchical")
                    )
                    
                    selected_node_2d = cytoscape_2d_viewer(
                        graph_data=cyto_data,
                        layout=layout,
                        height=700,
                        threshold=threshold,
                        key="pim_cytoscape_2d"
                    )
                    
                    if selected_node_2d:
                        st.success(f"**選択ノード:** {selected_node_2d['node_name']}")
                
                except Exception as e:
                    st.error(f"2D可視化エラー: {str(e)}")
                    st.caption("**エラー詳細:**")
                    st.code(str(e), language="python")


def tab7_network_analysis():
    """タブ7: ネットワーク分析（ステップ7）"""
    st.header("🔬 ステップ7: ネットワーク分析")
    
    adj_matrix_df = st.session_state.get("adj_matrix_df")
    nodes = SessionManager.get_nodes()
    
    if adj_matrix_df is None or nodes is None or len(nodes) < 2:
        st.warning("⚠️ 先にタブ5で隣接行列を生成してください")
        return
    
    st.markdown("""
    ネットワーク分析により、重要なノードを特定します。
    - **PageRank**: 各ノードの影響力スコア
    - **入次数中心性**: 他ノードから影響を受ける度合い
    - **出次数中心性**: 他ノードに影響を与える度合い
    - **媒介中心性**: ボトルネックとなるノードを検出
    """)
    
    if st.button("🔬 ネットワークを生成して分析", type="primary", use_container_width=True):
        with st.spinner("グラフを生成・分析中..."):
            try:
                import networkx as nx
                
                # グラフ生成
                adj_matrix = adj_matrix_df.values
                G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
                
                # ノード名でリラベル
                node_mapping = {i: nodes[i] for i in range(len(nodes))}
                G = nx.relabel_nodes(G, node_mapping)
                
                # 分析結果を保存
                st.session_state.network_graph = G
                
                st.success("✅ グラフの生成と分析が完了しました")
                
            except Exception as e:
                st.error(f"❌ エラー: {str(e)}")
                st.code(str(e), language="python")
                return
    
    # グラフが生成されている場合は結果を表示
    if "network_graph" in st.session_state and st.session_state.network_graph is not None:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = st.session_state.network_graph
        
        # 7.1 グラフ可視化
        st.markdown("---")
        st.subheader("7.1. グラフ可視化")
        
        col_viz, col_layout = st.columns([3, 1])
        
        with col_layout:
            layout_type = st.selectbox(
                "レイアウト",
                options=["spring", "circular", "kamada_kawai", "shell"],
                format_func=lambda x: {
                    "spring": "Spring（力学モデル）",
                    "circular": "Circular（円形）",
                    "kamada_kawai": "Kamada-Kawai（力学最適化）",
                    "shell": "Shell（同心円）"
                }[x]
            )
            
            node_size = st.slider("ノードサイズ", 100, 3000, 1500)
            font_size = st.slider("フォントサイズ", 6, 16, 10)
        
        with col_viz:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # 日本語フォント設定
            plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # レイアウト計算
            if layout_type == "spring":
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout_type == "circular":
                pos = nx.circular_layout(G)
            elif layout_type == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.shell_layout(G)
            
            # エッジの太さと色をスコアから計算
            edges = G.edges()
            edge_weights = [G[u][v]['weight'] for u, v in edges]
            edge_colors = ['red' if w < 0 else 'blue' for w in edge_weights]
            edge_widths = [abs(w) * 0.3 for w in edge_weights]
            
            # 描画
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                  node_size=node_size, alpha=0.9, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=font_size, 
                                   font_family='sans-serif', ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                                  width=edge_widths, alpha=0.6, 
                                  arrows=True, arrowsize=20, ax=ax)
            
            ax.set_title("ノード関係グラフ", fontsize=14, pad=20)
            ax.axis('off')
            
            st.pyplot(fig)
            plt.close()
        
        # 7.2 ネットワーク分析
        st.markdown("---")
        st.subheader("7.2. ネットワーク分析結果")
        
        # PageRank計算
        try:
            pagerank = nx.pagerank(G, weight='weight')
        except:
            pagerank = nx.pagerank(G)
        
        # 次数中心性
        in_degree = dict(G.in_degree(weight='weight'))
        out_degree = dict(G.out_degree(weight='weight'))
        
        # 媒介中心性
        try:
            betweenness = nx.betweenness_centrality(G, weight='weight')
        except:
            betweenness = nx.betweenness_centrality(G)
        
        # DataFrameにまとめる
        analysis_data = []
        for node in nodes:
            analysis_data.append({
                'ノード名': node,
                'PageRank': pagerank.get(node, 0),
                '入次数': in_degree.get(node, 0),
                '出次数': out_degree.get(node, 0),
                '媒介中心性': betweenness.get(node, 0)
            })
        
        df_analysis = pd.DataFrame(analysis_data)
        
        # ソートオプション
        sort_by = st.selectbox(
            "並び替え基準",
            options=['PageRank', '入次数', '出次数', '媒介中心性'],
            index=0
        )
        
        df_sorted = df_analysis.sort_values(by=sort_by, ascending=False)
        
        st.dataframe(
            df_sorted,
            use_container_width=True,
            hide_index=True
        )
        
        # 統計サマリー
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("総ノード数", len(nodes))
        with col_s2:
            st.metric("総エッジ数", G.number_of_edges())
        with col_s3:
            avg_pagerank = sum(pagerank.values()) / len(pagerank)
            st.metric("平均PageRank", f"{avg_pagerank:.4f}")
        with col_s4:
            avg_betweenness = sum(betweenness.values()) / len(betweenness)
            st.metric("平均媒介中心性", f"{avg_betweenness:.4f}")
        
        # 7.3 粒度調整提案
        st.markdown("---")
        st.subheader("7.3. 粒度調整提案")
        
        st.markdown("""
        ネットワーク分析の結果から、細分化を検討すべきノードを提案します：
        - **PageRank上位**: 影響力が大きいノード → 詳細な分析価値が高い
        - **媒介中心性上位**: ボトルネックとなるノード → 細分化で最適化の余地
        """)
        
        # PageRank上位5ノード
        top_pagerank = df_analysis.nlargest(5, 'PageRank')
        
        st.markdown("#### 📈 PageRank上位ノード（重要ノード）")
        
        for idx, row in top_pagerank.iterrows():
            node_name = row['ノード名']
            pr_score = row['PageRank']
            
            with st.container(border=True):
                col_info, col_btn = st.columns([3, 1])
                
                with col_info:
                    st.markdown(f"**{node_name}**")
                    st.caption(f"PageRank: {pr_score:.4f}")
                    
                    if pr_score > avg_pagerank * 2:
                        st.warning("⚠️ 平均の2倍以上の影響力 → 重要ノード、細分化の価値が高い")
                    elif pr_score > avg_pagerank * 1.5:
                        st.info("💡 平均以上の影響力 → 細分化を検討")
                
                with col_btn:
                    if st.button("🔄 細分化", key=f"refine_pr_{node_name}", use_container_width=True):
                        st.session_state.selected_refinement_node = node_name
                        st.info(f"💡 「ステップ3: ノード定義」タブに移動して、「{node_name}」を含むカテゴリを細分化してください")
                        st.info("タブ3で「Zigzagging粒度調整」モードを選択してください")
        
        # 媒介中心性上位5ノード
        st.markdown("#### 🔗 媒介中心性上位ノード（ボトルネック）")
        
        top_betweenness = df_analysis.nlargest(5, '媒介中心性')
        
        for idx, row in top_betweenness.iterrows():
            node_name = row['ノード名']
            btw_score = row['媒介中心性']
            
            with st.container(border=True):
                col_info, col_btn = st.columns([3, 1])
                
                with col_info:
                    st.markdown(f"**{node_name}**")
                    st.caption(f"媒介中心性: {btw_score:.4f}")
                    
                    if btw_score > avg_betweenness * 2:
                        st.warning("⚠️ 平均の2倍以上 → ボトルネック、最適化の余地あり")
                    elif btw_score > avg_betweenness * 1.5:
                        st.info("💡 平均以上 → 細分化を検討")
                
                with col_btn:
                    if st.button("🔄 細分化", key=f"refine_btw_{node_name}", use_container_width=True):
                        st.session_state.selected_refinement_node = node_name
                        st.info(f"💡 「ステップ3: ノード定義」タブに移動して、「{node_name}」を含むカテゴリを細分化してください")
                        st.info("タブ3で「Zigzagging粒度調整」モードを選択してください")
    
    else:
        st.info("👆 「ネットワークを生成して分析」ボタンをクリックして開始してください")


def tab8_dsm_optimization():
    """タブ8: DSM最適化（NSGA-II）"""
    st.header("🎮 ステップ8: DSM最適化（NSGA-II）")
    
    adj_matrix_df = st.session_state.get("adj_matrix_df")
    nodes = SessionManager.get_nodes()
    all_idef0 = SessionManager.get_all_idef0_nodes()
    
    if adj_matrix_df is None or nodes is None or len(nodes) < 2:
        st.warning("⚠️ 先にタブ5で隣接行列を生成してください")
        return
    
    st.markdown("""
    設計構造マトリクス（DSM）の多目的最適化を行います。
    
    **STEP-1**: 設計パラメータ選択（コスト vs 自由度）  
    **STEP-2**: 依存関係方向決定（調整困難度 vs 競合困難度 vs ループ困難度）
    """)
    
    # 8.1 DSM設定
    st.markdown("---")
    st.subheader("8.1. DSM設定")
    
    from utils.idef0_classifier import classify_node_type, NodeType
    
    # FR/DP分類
    fr_nodes = []
    dp_nodes = []
    for node_name in nodes:
        node_type, _ = classify_node_type(node_name, all_idef0)
        if node_type == NodeType.OUTPUT:
            fr_nodes.append(node_name)
        else:
            dp_nodes.append(node_name)
    
    col_fr, col_dp = st.columns(2)
    with col_fr:
        st.metric("FR（機能要求）", len(fr_nodes), help="Outputノード")
    with col_dp:
        st.metric("DP（設計パラメータ）", len(dp_nodes), help="Mechanism + Inputノード")
    
    st.info("""
    💡 **FR/DP分類について**
    - **FR（機能要求）**: プロセスの成果物（Output）
    - **DP（設計パラメータ）**: 成果物を実現する手段と材料（Mechanism + Input）
    """)
    
    param_mode = st.radio(
        "パラメータ設定方法",
        options=["llm_auto", "fixed_default", "manual_custom"],
        format_func=lambda x: {
            "llm_auto": "🤖 LLMによる自動評価（推奨）",
            "fixed_default": "📊 固定デフォルト値",
            "manual_custom": "⚙️ 手動カスタム設定（上級者向け）"
        }[x],
        index=0,
        help="LLM自動評価: プロセスの文脈を考慮したパラメータ評価 | 固定デフォルト: Cost=1, Range=1, Importance=1"
    )
    
    # LLM評価モード
    if param_mode == "llm_auto":
        st.markdown("""
        **LLMが評価するパラメータ:**
        - **Cost（コスト）**: 1-5スケール（DPのみ）
        - **Range（変動範囲）**: 0.1-2.0スケール（DPのみ）
        - **Importance（重要度）**: 1-5スケール（FRのみ）
        - **Structure（構造グループ）**: 論理的なグループ名
        """)
        
        if st.button("🤖 パラメータをLLMで評価", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                from core.llm_client import LLMClient
                
                llm_client = LLMClient()
                
                # ノード分類を作成
                node_classifications = {}
                for node_name in nodes:
                    node_type, _ = classify_node_type(node_name, all_idef0)
                    if node_type == NodeType.OUTPUT:
                        node_classifications[node_name] = "FR"
                    else:
                        node_classifications[node_name] = "DP"
                
                # 進捗コールバック関数
                def update_progress(ratio):
                    progress_bar.progress(ratio)
                    status_text.text(f"LLMが評価中... {int(ratio*100)}%")
                
                # LLM評価（バッチ処理）
                result = llm_client.evaluate_dsm_parameters(
                    process_name=SessionManager.get_process_name(),
                    process_description=SessionManager.get_process_description(),
                    nodes=nodes,
                    idef0_nodes=all_idef0,
                    node_classifications=node_classifications,
                    batch_size=10,
                    progress_callback=update_progress
                )
                
                # セッションに保存
                st.session_state.dsm_llm_params = result
                
                progress_bar.empty()
                status_text.empty()
                st.success("✅ LLMによるパラメータ評価が完了しました")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ エラー: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="python")
        
        # 評価結果の表示
        if "dsm_llm_params" in st.session_state and st.session_state.dsm_llm_params:
            result = st.session_state.dsm_llm_params
            params_data = result.get("parameters", {})
            
            with st.expander("📊 評価結果", expanded=True):
                # DataFrame作成
                df_data = []
                for node_name in nodes:
                    node_params = params_data.get(node_name, {})
                    node_type, _ = classify_node_type(node_name, all_idef0)
                    
                    row = {
                        "ノード名": node_name,
                        "タイプ": "FR" if node_type == NodeType.OUTPUT else "DP"
                    }
                    
                    if node_type == NodeType.OUTPUT:
                        row["コスト"] = "-"
                        row["変動範囲"] = "-"
                        row["重要度"] = f"{node_params.get('importance', '-')}"
                    else:
                        row["コスト"] = f"{node_params.get('cost', '-')}"
                        row["変動範囲"] = f"{node_params.get('range', '-')}"
                        row["重要度"] = "-"
                    
                    row["構造グループ"] = node_params.get("structure", "-")
                    df_data.append(row)
                
                df_params = pd.DataFrame(df_data)
                st.dataframe(df_params, use_container_width=True, hide_index=True)
                
                st.markdown("**評価の根拠:**")
                st.write(result.get("reasoning", "根拠が提供されませんでした"))
    
    elif param_mode == "fixed_default":
        st.info("""
        📊 **固定デフォルト値**
        - Cost = 1
        - Range = 1
        - Importance = 1
        - Structure = カテゴリ名
        """)
    
    else:  # manual_custom
        st.warning("⚠️ カスタムパラメータは上級者向けです。デフォルト値またはLLM評価の使用を推奨します。")
    
    # 8.2 STEP-1実行
    st.markdown("---")
    st.subheader("8.2. STEP-1: 設計パラメータ選択")
    
    st.markdown("""
    どの設計パラメータを削除すべきかを決定します。
    - **目的1**: コスト最小化（同一構造内の最大コストの合計）
    - **目的2**: 設計自由度最大化（各FRの調整能力比の総和）
    """)
    
    # 軽量モード
    lightweight_mode = st.checkbox(
        "⚡ 軽量モード（推奨）",
        value=True,
        help="個体数と世代数を削減し、サーバークラッシュを防ぎます"
    )
    
    if lightweight_mode:
        default_pop, default_gen = 100, 50
    else:
        default_pop, default_gen = 200, 100
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        step1_pop = st.slider("個体数", 50, 500, default_pop, 50, key="step1_pop")
    with col_p2:
        step1_gen = st.slider("世代数", 20, 200, default_gen, 10, key="step1_gen")
    
    # データ構築（ボタンの外で準備）
    llm_params = st.session_state.get("dsm_llm_params") if param_mode == "llm_auto" else None
    
    if st.button("🚀 STEP-1を実行", type="primary", use_container_width=True):
        from utils.dsm_optimizer import PIMDSMData, PIMStep1NSGA2
        import time
        
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            start_time = time.time()
            
            # データ構築
            dsm_data = PIMDSMData(
                adj_matrix_df=adj_matrix_df,
                nodes=nodes,
                idef0_nodes=all_idef0,
                param_mode=param_mode,
                llm_params=llm_params,
                custom_params=None
            )
            
            # 進捗コールバック
            def progress_callback(gen: int, pareto_size: int):
                progress_pct = gen / step1_gen
                progress_placeholder.progress(
                    progress_pct,
                    text=f"世代 {gen}/{step1_gen} (パレート解: {pareto_size}個)"
                )
            
            # STEP-1実行（同期）
            step1 = PIMStep1NSGA2(dsm_data)
            pareto_front = step1.run(
                n_pop=step1_pop,
                n_gen=step1_gen,
                checkpoint_id=None,
                save_every=10,
                progress_callback=progress_callback
            )
            
            elapsed = time.time() - start_time
            
            # 結果をリスト化
            step1_results = []
            for ind in pareto_front:
                cost, freedom_inv = ind.fitness.values
                removed_indices = [i for i, val in enumerate(ind) if val == 1]
                removed_nodes = [dsm_data.reordered_nodes[i] for i in removed_indices]
                step1_results.append({
                    'individual': list(ind),
                    'cost': cost,
                    'freedom_inv': freedom_inv,
                    'freedom': 1/freedom_inv if freedom_inv != float('inf') else 0,
                    'removed_count': len(removed_nodes),
                    'removed_nodes': removed_nodes
                })
            
            # セッションに保存
            st.session_state.dsm_data = dsm_data
            st.session_state.step1_results = step1_results
            
            progress_placeholder.empty()
            status_placeholder.success(f"✅ STEP-1完了: {len(pareto_front)}個のパレート解を発見（{elapsed:.1f}秒）")
            
        except Exception as e:
            progress_placeholder.empty()
            status_placeholder.error(f"❌ エラー: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
    
    # STEP-1結果の可視化
    if "step1_results" in st.session_state and st.session_state.step1_results:
        results = st.session_state.step1_results
        
        st.markdown("#### パレートフロント（2D）")
        
        # 日本語フォント設定
        plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 散布図
        fig, ax = plt.subplots(figsize=(10, 6))
        costs = [r['cost'] for r in results]
        freedoms = [r['freedom'] for r in results]
        
        scatter = ax.scatter(costs, freedoms, c=range(len(results)), cmap='viridis', s=100, alpha=0.7)
        ax.set_xlabel('コスト（最小化）', fontsize=12)
        ax.set_ylabel('設計自由度（最大化）', fontsize=12)
        ax.set_title('STEP-1 パレートフロント', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='解番号')
        
        st.pyplot(fig)
        plt.close()
        
        # 解選択
        st.markdown("#### 解の選択")
        
        # DataFrame表示
        df_results = pd.DataFrame([{
            '解番号': i,
            'コスト': f"{r['cost']:.2f}",
            '設計自由度': f"{r['freedom']:.4f}",
            '削除DP数': r['removed_count']
        } for i, r in enumerate(results)])
        
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        
        selected_idx = st.selectbox(
            "STEP-2に使用する解を選択",
            options=list(range(len(results))),
            format_func=lambda i: f"解{i}: コスト={results[i]['cost']:.2f}, 自由度={results[i]['freedom']:.4f}"
        )
        
        if selected_idx is not None:
            selected = results[selected_idx]
            
            with st.expander(f"📊 解{selected_idx}の詳細", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("コスト", f"{selected['cost']:.2f}")
                with col2:
                    st.metric("設計自由度", f"{selected['freedom']:.4f}")
                with col3:
                    st.metric("削除DP数", selected['removed_count'])
                
                if selected['removed_nodes']:
                    st.markdown("**削除される設計パラメータ:**")
                    for node in selected['removed_nodes']:
                        st.caption(f"- {node}")
                else:
                    st.info("すべての設計パラメータが保持されます")
            
            # 選択を保存
            st.session_state.step1_selected_idx = selected_idx
    
    # 8.3 STEP-2実行
    if "step1_selected_idx" in st.session_state:
        st.markdown("---")
        st.subheader("8.3. STEP-2: 依存関係方向決定")
        
        st.markdown("""
        残った設計パラメータ間の依存関係の方向を最適化します。
        - **目的1**: 調整困難度最小化（αパターン + γパターン）
        - **目的2**: 競合困難度最小化（列への複数影響の相乗効果）
        - **目的3**: ループ困難度最小化（閉路の累積影響）
        """)
        
        # 軽量モード（STEP-2）
        lightweight_mode_s2 = st.checkbox(
            "⚡ 軽量モード（推奨）",
            value=True,
            help="個体数と世代数を削減し、サーバークラッシュを防ぎます",
            key="lightweight_s2"
        )
        
        if lightweight_mode_s2:
            default_pop_s2, default_gen_s2 = 100, 30
        else:
            default_pop_s2, default_gen_s2 = 200, 50
        
        col_p3, col_p4 = st.columns(2)
        with col_p3:
            step2_pop = st.slider("個体数", 50, 500, default_pop_s2, 50, key="step2_pop")
        with col_p4:
            step2_gen = st.slider("世代数", 20, 200, default_gen_s2, 10, key="step2_gen")
        
        if st.button("🚀 STEP-2を実行", type="primary", use_container_width=True):
            from utils.dsm_optimizer import PIMStep2NSGA2
            import time
            
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # 初期メッセージ
            status_placeholder.info(f"🚀 NSGA-II最適化を開始しました（{step2_pop}個体 × {step2_gen}世代）...")
            
            with st.spinner("最適化実行中... 進捗は下のプログレスバーで確認できます"):
                try:
                    start_time = time.time()
                    gen_times = []
                    
                    dsm_data = st.session_state.dsm_data
                    selected = st.session_state.step1_results[st.session_state.step1_selected_idx]
                    removed_indices = [i for i, val in enumerate(selected['individual']) if val == 1]
                    
                    # 進捗コールバック
                    def progress_callback(gen: int, pareto_size: int):
                        progress_pct = gen / step2_gen
                        
                        # 推定残り時間計算
                        if gen > 0:
                            elapsed = time.time() - start_time
                            avg_time_per_gen = elapsed / gen
                            remaining_gens = step2_gen - gen
                            eta_seconds = avg_time_per_gen * remaining_gens
                            eta_min = int(eta_seconds // 60)
                            eta_sec = int(eta_seconds % 60)
                            eta_text = f" | 推定残り時間: {eta_min}分{eta_sec}秒"
                        else:
                            eta_text = ""
                        
                        progress_placeholder.progress(
                            progress_pct,
                            text=f"世代 {gen}/{step2_gen} (パレート解: {pareto_size}個){eta_text}"
                        )
                    
                    # STEP-2実行（同期）
                    step2 = PIMStep2NSGA2(dsm_data, removed_indices)
                    pareto_front = step2.run(
                        n_pop=step2_pop,
                        n_gen=step2_gen,
                        checkpoint_id=None,
                        save_every=1,
                        progress_callback=progress_callback
                    )
                    
                    elapsed = time.time() - start_time
                    
                    # 結果をリスト化
                    step2_results = []
                    for ind in pareto_front:
                        adj, conf, loop = ind.fitness.values
                        step2_results.append({
                            'matrix': ind[0].copy(),
                            'adjustment': adj,
                            'conflict': conf,
                            'loop': loop
                        })
                    
                    st.session_state.step2_results = step2_results
                    st.session_state.step2_package = step2.pkg
                    
                    progress_placeholder.empty()
                    elapsed_min = int(elapsed // 60)
                    elapsed_sec = int(elapsed % 60)
                    status_placeholder.success(f"✅ STEP-2完了: {len(pareto_front)}個のパレート解を発見（{elapsed_min}分{elapsed_sec}秒）")
                    
                except Exception as e:
                    progress_placeholder.empty()
                    status_placeholder.error(f"❌ エラー: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
        
        # STEP-2結果の可視化
        if "step2_results" in st.session_state and st.session_state.step2_results:
            results2 = st.session_state.step2_results
            
            st.markdown("#### パレートフロント（3D）")
            
            # 日本語フォント設定
            plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 3D散布図
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            adjs = [r['adjustment'] for r in results2]
            confs = [r['conflict'] for r in results2]
            loops = [r['loop'] for r in results2]
            
            scatter = ax.scatter(adjs, confs, loops, c=range(len(results2)), cmap='plasma', s=100, alpha=0.7)
            ax.set_xlabel('調整困難度', fontsize=10)
            ax.set_ylabel('競合困難度', fontsize=10)
            ax.set_zlabel('ループ困難度', fontsize=10)
            ax.set_title('STEP-2 パレートフロント', fontsize=14)
            plt.colorbar(scatter, ax=ax, label='解番号', shrink=0.5)
            
            st.pyplot(fig)
            plt.close()
            
            # 解選択
            st.markdown("#### 解の選択")
            
            df_results2 = pd.DataFrame([{
                '解番号': i,
                '調整困難度': f"{r['adjustment']:.2f}",
                '競合困難度': f"{r['conflict']:.2f}",
                'ループ困難度': f"{r['loop']:.2f}"
            } for i, r in enumerate(results2)])
            
            st.dataframe(df_results2, use_container_width=True, hide_index=True)
            
            selected_idx2 = st.selectbox(
                "最終解を選択",
                options=list(range(len(results2))),
                format_func=lambda i: f"解{i}: 調整={results2[i]['adjustment']:.2f}, 競合={results2[i]['conflict']:.2f}, ループ={results2[i]['loop']:.2f}"
            )
            
            if selected_idx2 is not None:
                selected2 = results2[selected_idx2]
                pkg = st.session_state.step2_package
                
                with st.expander(f"📊 解{selected_idx2}の詳細", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("調整困難度", f"{selected2['adjustment']:.2f}")
                    with col2:
                        st.metric("競合困難度", f"{selected2['conflict']:.2f}")
                    with col3:
                        st.metric("ループ困難度", f"{selected2['loop']:.2f}")
                    
                    st.markdown("**最適化されたDSM:**")
                    
                    # ヒートマップ
                    optimized_matrix = selected2['matrix']
                    node_names = [pkg['node_name'][0][i] for i in range(pkg['matrix_size'])]
                    
                    df_optimized = pd.DataFrame(
                        optimized_matrix,
                        index=node_names,
                        columns=node_names
                    )
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # 日本語フォント設定
                    plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    sns.heatmap(
                        df_optimized,
                        annot=True,
                        fmt='.0f',
                        cmap='coolwarm',
                        center=0,
                        vmin=-9,
                        vmax=9,
                        linewidths=0.5,
                        cbar_kws={'label': '影響スコア'},
                        ax=ax
                    )
                    ax.set_title('最適化されたDSM', fontsize=14, pad=20)
                    
                    st.pyplot(fig)
                    plt.close()
                
                # 選択を保存
                st.session_state.step2_selected_idx = selected_idx2
                st.session_state.optimized_dsm = selected2['matrix']
    
    # 8.4 STEP-3: パーティショニングとモジュール化
    if "step2_selected_idx" in st.session_state and "optimized_dsm" in st.session_state:
        st.markdown("---")
        st.subheader("8.4. STEP-3: パーティショニングとモジュール化")
        
        with st.expander("💡 この分析について", expanded=True):
            st.markdown("""
            **何がわかるか:**
            - どの要素をグループ化すべきか（モジュール検出）
            - どの順番で設計すべきか（デザインシーケンス）
            - どこで手戻りが発生するか（フィードバックループ）
            
            **どう使えばいいか:**
            - チーム編成の参考にする（モジュール単位で分担）
            - 作業順序を決定する（デザインシーケンスに従う）
            - 手戻りを事前に認識する（イテレーション箇所の特定）
            
            **結果の見方:**
            - モジュール: 密に結合したノード群
            - デザインシーケンス: 依存関係に基づく最適な設計順序
            - フィードバック比率: 手戻りの度合い（低いほど良い）
            """)
        
        st.info("⏱️ 推定計算時間: <1分")
        
        with st.expander("⚙️ 詳細設定", expanded=False):
            n_modules = st.slider(
                "モジュール数",
                min_value=2,
                max_value=min(10, len(st.session_state.optimized_dsm) // 2),
                value=None,
                help="Noneの場合は自動決定（√(N/2)個）"
            )
        
        if st.button("🚀 パーティショニングを実行", type="primary", use_container_width=True):
            with st.spinner("分析中..."):
                try:
                    from utils.dsm_partitioning import DSMPartitioner
                    import time
                    
                    start_time = time.time()
                    
                    optimized_matrix = st.session_state.optimized_dsm
                    pkg = st.session_state.step2_package
                    node_names = [pkg['node_name'][0][i] for i in range(pkg['matrix_size'])]
                    
                    partitioner = DSMPartitioner(optimized_matrix, node_names)
                    
                    analysis_result = partitioner.full_analysis(n_clusters=n_modules)
                    
                    st.session_state.partitioning_result = analysis_result
                    
                    elapsed = time.time() - start_time
                    
                    st.success(f"✅ パーティショニング完了（{elapsed:.2f}秒）")
                    
                except Exception as e:
                    st.error(f"❌ エラー: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
                    return
        
        if "partitioning_result" in st.session_state:
            result = st.session_state.partitioning_result
            
            st.markdown("### 分析結果")
            
            # 8.4.1 モジュール情報
            with st.expander("📦 モジュール情報", expanded=True):
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("モジュール数", result['modules']['n_modules'])
                with col_m2:
                    st.metric("モジュラリティスコア", f"{result['modularity_score']:.3f}",
                             help="高いほど良いモジュール分割（-1～1）")
                with col_m3:
                    st.metric("フィードバック比率", f"{result['feedback_loops']['feedback_ratio']:.1%}",
                             help="低いほど手戻りが少ない")
                
                st.markdown("**各モジュールのメンバー:**")
                for module_id, members in result['module_members'].items():
                    with st.expander(f"モジュール{module_id}（{len(members)}ノード）"):
                        for member in members:
                            st.caption(f"- {member}")
            
            # 8.4.2 デザインシーケンス
            with st.expander("📋 デザインシーケンス（設計順序）", expanded=True):
                st.markdown("**推奨される設計順序:**")
                sequence_nodes = result['design_sequence']['reordered_nodes']
                for i, node in enumerate(sequence_nodes, 1):
                    st.caption(f"{i}. {node}")
            
            # 8.4.3 フィードバックループ
            with st.expander("🔁 フィードバックループ（手戻り箇所）", expanded=True):
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    st.metric("フィードフォワード", result['feedback_loops']['feedforward_count'])
                with col_f2:
                    st.metric("フィードバック（手戻り）", result['feedback_loops']['feedback_count'])
                with col_f3:
                    st.metric("対角要素", result['feedback_loops']['diagonal_count'])
                
                if result['feedback_loops']['feedback_elements']:
                    st.markdown("**手戻りが発生する箇所:**")
                    feedback_df = pd.DataFrame([
                        {
                            "From": elem['from'],
                            "To": elem['to'],
                            "影響スコア": elem['value']
                        }
                        for elem in result['feedback_loops']['feedback_elements'][:20]
                    ])
                    st.dataframe(feedback_df, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ 手戻りがありません（理想的な設計順序）")
            
            # 8.4.4 パーティショニング済みDSMヒートマップ
            with st.expander("📊 パーティショニング済みDSMヒートマップ", expanded=True):
                reordered_matrix = result['design_sequence']['reordered_matrix']
                reordered_nodes = result['design_sequence']['reordered_nodes']
                
                df_partitioned = pd.DataFrame(
                    reordered_matrix,
                    index=reordered_nodes,
                    columns=reordered_nodes
                )
                
                fig, ax = plt.subplots(figsize=(14, 12))
                
                plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                
                sns.heatmap(
                    df_partitioned,
                    annot=True,
                    fmt='.0f',
                    cmap='coolwarm',
                    center=0,
                    vmin=-9,
                    vmax=9,
                    linewidths=0.5,
                    cbar_kws={'label': '影響スコア'},
                    ax=ax
                )
                ax.set_title('パーティショニング済みDSM（デザインシーケンス順）', fontsize=14, pad=20)
                ax.set_xlabel('To Node (影響を受ける)', fontsize=12)
                ax.set_ylabel('From Node (影響を与える)', fontsize=12)
                
                st.pyplot(fig)
                plt.close()
    
    else:
        st.info("👆 まずSTEP-1を実行してください")


def tab9_advanced_analytics():
    """タブ9: 高度な分析（ステップ9）"""
    st.header("🧬 ステップ9: 高度な分析")
    
    adj_matrix_df = st.session_state.get("adj_matrix_df")
    nodes = SessionManager.get_nodes()
    all_idef0 = SessionManager.get_all_idef0_nodes()
    
    if adj_matrix_df is None or nodes is None or len(nodes) < 2:
        st.warning("⚠️ 先にタブ5で隣接行列を生成してください")
        return
    
    st.markdown("""
    データ分析の専門知識がなくても使える高度な分析ツールです。
    各手法で「何がわかるか」「どう使えばいいか」を平易に説明します。
    
    **7つの分析手法:**
    1. 協力貢献度分析（Shapley Value）
    2. 情報フロー分析（Transfer Entropy）
    3. 統計的検定（Bootstrap法）
    4. 不確実性定量化（Bayesian Inference）
    5. 因果推論（Pearl's Causal Inference）
    6. 潜在構造発見（Graph Embedding）
    7. 感度分析（Fisher Information）
    """)
    
    st.info("💡 各分析には計算時間の見積もりが表示されます。興味のある分析から順に実行してください。")
    
    # 9.1 Shapley Value
    st.markdown("---")
    st.subheader("9.1. 協力貢献度分析（Shapley Value）⭐ 推奨")
    
    with st.expander("💡 この分析について", expanded=False):
        st.markdown("""
        **何がわかるか:**
        
        各ノードの「真の貢献度」を公平に評価します。
        「このノードを削除したら全体性能がどれだけ下がるか」を数値化します。
        
        **どう使えばいいか:**
        - 投資優先順位の決定（貢献度が高い工程を優先改善）
        - 見えにくい「縁の下の力持ち」の発見
        - リソース配分の根拠作成
        
        **結果の見方:**
        - Shapley値が高い = 全体への貢献が大きい
        - 上位10ノードを重点管理対象とする
        - 負の値 = 削除すると全体が改善する可能性（要再検討）
        """)
    
    st.info(f"⏱️ 推定計算時間: 2-5分（{len(nodes)}ノード、サンプル数1000）")
    
    col_settings, col_execute = st.columns([2, 1])
    
    with col_settings:
        n_samples = st.slider(
            "サンプル数",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="多いほど精度向上、計算時間増加"
        )
        
        value_function = st.selectbox(
            "価値関数",
            options=["pagerank_sum", "efficiency", "connectivity"],
            format_func=lambda x: {
                "pagerank_sum": "PageRank合計（推奨）",
                "efficiency": "ネットワーク効率性",
                "connectivity": "接続性"
            }[x],
            help="ネットワークの価値をどう評価するか"
        )
    
    with col_execute:
        st.write("")
        st.write("")
        execute_shapley = st.button("🚀 分析実行", key="shapley_btn", type="primary", use_container_width=True)
    
    if execute_shapley:
        try:
            with st.spinner("Shapley Value計算中..."):
                from utils.shapley_analysis import ShapleyAnalyzer
                from utils.analytics_progress import AnalyticsProgressTracker, create_simple_callback
                
                # 進捗トラッカーを初期化
                tracker = AnalyticsProgressTracker("Shapley Value分析", total_steps=n_samples)
                
                # ノードカテゴリマッピング
                categories_list = SessionManager.get_functional_categories()
                all_idef0 = SessionManager.get_all_idef0_nodes()
                node_categories = {}
                for category in categories_list:
                    if category in all_idef0:
                        idef0_dict = all_idef0[category]
                        for node_type in ['outputs', 'mechanisms', 'inputs']:
                            if node_type in idef0_dict:
                                for node_name in idef0_dict[node_type]:
                                    node_categories[node_name] = category
                
                # Shapley分析実行
                analyzer = ShapleyAnalyzer(
                    adjacency_matrix=st.session_state.adjacency_matrix,
                    node_names=nodes,
                    node_categories=node_categories,
                    value_function=value_function
                )
                
                # シンプルコールバックを生成
                progress_callback = create_simple_callback(tracker)
                
                result = analyzer.compute_shapley_values(
                    n_samples=n_samples,
                    progress_callback=progress_callback
                )
                
                # 結果を保存
                if "advanced_analytics_results" not in st.session_state:
                    st.session_state.advanced_analytics_results = {}
                
                st.session_state.advanced_analytics_results["shapley"] = {
                    "result": result,
                    "parameters": {
                        "n_samples": n_samples,
                        "value_function": value_function
                    },
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # 完了処理
                tracker.complete(result.computation_time)
        except Exception as e:
            if 'tracker' in locals():
                tracker.error(str(e))
            st.error(f"❌ Shapley Value分析エラー: {str(e)}")
            with st.expander("🔍 エラー詳細"):
                import traceback
                st.code(traceback.format_exc(), language="python")
    
    # 結果表示
    if "advanced_analytics_results" in st.session_state and "shapley" in st.session_state.advanced_analytics_results:
        result_data = st.session_state.advanced_analytics_results["shapley"]
        result = result_data["result"]
        
        st.markdown("---")
        st.subheader("📊 分析結果")
        
        # 解釈文
        with st.expander("💡 結果の解釈", expanded=True):
            st.markdown(result.interpretation)
        
        # メトリクス
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("総ノード数", len(result.shapley_values))
        with col_m2:
            st.metric("全体価値", f"{result.total_value:.4f}")
        with col_m3:
            st.metric("計算時間", f"{result.computation_time:.1f}秒")
        with col_m4:
            top_value = result.top_contributors[0][1] if result.top_contributors else 0
            st.metric("最大貢献度", f"{top_value:.4f}")
        
        # 上位貢献者表
        st.markdown("### 🏆 貢献度ランキング（上位20）")
        top_20 = result.top_contributors[:20]
        df_top = pd.DataFrame([
            {
                "順位": i+1,
                "ノード名": name,
                "Shapley値": value,
                "貢献率%": (value / result.total_value * 100) if result.total_value > 0 else 0
            }
            for i, (name, value) in enumerate(top_20)
        ])
        st.dataframe(df_top, use_container_width=True, hide_index=True)
        
        # 可視化
        import matplotlib.pyplot as plt
        # 日本語フォント設定
        plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("### 📊 貢献度分布（上位15）")
            top_15 = result.top_contributors[:15]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            names = [name for name, _ in top_15]
            values = [value for _, value in top_15]
            
            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
            ax.barh(range(len(names)), values, color=colors, alpha=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel('Shapley Value')
            ax.set_title('Top 15 Contributors')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        with col_viz2:
            st.markdown("### 📈 累積貢献度")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            x = [n for n, _ in result.cumulative_contribution]
            y = [pct for _, pct in result.cumulative_contribution]
            
            ax.plot(x, y, marker='o', linewidth=2, markersize=4, color='#3498db')
            ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80%ライン')
            ax.set_xlabel('Top N Nodes')
            ax.set_ylabel('Cumulative Contribution (%)')
            ax.set_title('Cumulative Contribution Curve')
            ax.legend()
            ax.grid(alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        # カテゴリ別貢献度
        if result.category_contributions:
            st.markdown("### 📦 カテゴリ別平均貢献度")
            df_cat = pd.DataFrame([
                {"カテゴリ": cat, "平均Shapley値": value}
                for cat, value in sorted(result.category_contributions.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(df_cat, use_container_width=True, hide_index=True)
        
        # 7. 連携安定性分析
        st.markdown("### 🔗 連携安定性分析")
        st.markdown("""
        **目的:** Shapley値上位ノード同士を連携させることで、相乗効果を最大化
        
        上位25%のノード間の接続強度を分析し、密結合ペアを特定。
        これらの連携を強化することで、全体性能の向上が期待できます。
        """)
        
        if st.button("🔗 連携安定性を分析", key="coalition_stability_btn"):
            with st.spinner("連携安定性を計算中..."):
                from utils.shapley_analysis import compute_shapley_coalition_stability
                
                stability_result = compute_shapley_coalition_stability(
                    shapley_values=result.shapley_values,
                    adjacency_matrix=st.session_state.adjacency_matrix,
                    node_names=nodes
                )
                
                # セッションステートに保存
                st.session_state.advanced_analytics_results["shapley"]["stability"] = stability_result
                
                st.success(f"✅ 連携安定性分析完了（上位{len(stability_result['top_contributors'])}ノード分析）")
        
        # 結果表示
        if "stability" in st.session_state.advanced_analytics_results["shapley"]:
            stability_result = st.session_state.advanced_analytics_results["shapley"]["stability"]
            
            # 推奨メッセージ
            st.info(stability_result["recommendation"])
            
            col_stab1, col_stab2 = st.columns([1, 1])
            
            with col_stab1:
                st.markdown("#### 🏆 上位貢献者（Top 25%）")
                top_nodes = stability_result["top_contributors"]
                
                # データフレーム
                df_top_nodes = pd.DataFrame([
                    {
                        "順位": i+1,
                        "ノード名": node,
                        "Shapley値": result.shapley_values[node]
                    }
                    for i, node in enumerate(top_nodes)
                ])
                st.dataframe(df_top_nodes, use_container_width=True, hide_index=True)
            
            with col_stab2:
                st.markdown("#### 🤝 密結合ペア（Top 10）")
                dense_connections = stability_result["dense_connections"]
                
                if dense_connections:
                    df_dense = pd.DataFrame([
                        {
                            "順位": i+1,
                            "ノード1": node1,
                            "ノード2": node2,
                            "接続強度": strength
                        }
                        for i, (node1, node2, strength) in enumerate(dense_connections)
                    ])
                    st.dataframe(df_dense, use_container_width=True, hide_index=True)
                else:
                    st.warning("上位ノード間に接続がありません（疎なネットワーク）")
            
            # ネットワーク図可視化
            if dense_connections:
                st.markdown("#### 🌐 連携ネットワーク図")
                
                import networkx as nx
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # グラフ構築
                G = nx.Graph()
                
                # 上位ノードを追加
                top_nodes = stability_result["top_contributors"]
                for node in top_nodes:
                    G.add_node(node, shapley=result.shapley_values[node])
                
                # 密結合エッジを追加
                for node1, node2, strength in dense_connections:
                    G.add_edge(node1, node2, weight=strength)
                
                # レイアウト計算（spring layout）
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
                
                # ノードサイズ（Shapley値に比例）
                node_sizes = [result.shapley_values[node] * 3000 for node in G.nodes()]
                
                # ノード色（Shapley値でグラデーション）
                shapley_vals = [result.shapley_values[node] for node in G.nodes()]
                
                # エッジ幅（接続強度に比例）
                edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
                
                # 描画
                nx.draw_networkx_nodes(
                    G, pos, 
                    node_size=node_sizes,
                    node_color=shapley_vals,
                    cmap=plt.cm.YlGnBu,
                    alpha=0.8,
                    ax=ax
                )
                
                nx.draw_networkx_edges(
                    G, pos,
                    width=edge_widths,
                    alpha=0.6,
                    edge_color='gray',
                    ax=ax
                )
                
                nx.draw_networkx_labels(
                    G, pos,
                    font_size=9,
                    font_weight='bold',
                    ax=ax
                )
                
                ax.set_title('Coalition Stability Network (Top Contributors)', fontsize=14, fontweight='bold')
                ax.axis('off')
                
                # カラーバー
                sm = plt.cm.ScalarMappable(
                    cmap=plt.cm.YlGnBu,
                    norm=plt.Normalize(vmin=min(shapley_vals), vmax=max(shapley_vals))
                )
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Shapley Value', rotation=270, labelpad=20)
                
                st.pyplot(fig)
                plt.close()
    
    # 9.2 Transfer Entropy
    st.markdown("---")
    st.subheader("9.2. 情報フロー分析（Transfer Entropy）⭐ 推奨")
    
    with st.expander("💡 この分析について", expanded=False):
        st.markdown("""
        **何がわかるか:**
        
        「誰が誰に何bit情報を伝えているか」を定量化します。
        単なる相関ではなく、因果的な情報の流れを検出します。
        
        **どう使えばいいか:**
        - 真のボトルネックの特定（情報が集中・遮断される箇所）
        - 間接的な影響経路の発見
        - コミュニケーション設計の改善
        
        **結果の見方:**
        - Transfer Entropy が高い = 強い因果的影響
        - 0に近い = 見かけの相関のみ（実際には影響していない）
        """)
    
    st.info(f"⏱️ 推定計算時間: 1-3分（{len(nodes)}ノード）")
    
    col_settings_te, col_execute_te = st.columns([2, 1])
    
    with col_settings_te:
        n_walks = st.slider(
            "ランダムウォーク回数",
            min_value=500,
            max_value=5000,
            value=1000,
            step=100,
            help="多いほど精度向上、計算時間増加"
        )
        
        walk_length = st.slider(
            "ウォーク長",
            min_value=20,
            max_value=100,
            value=50,
            step=10,
            help="時系列の長さ"
        )
        
        n_bins = st.slider(
            "離散化ビン数",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            help="低い=粗い分類、高い=細かい分類"
        )
    
    with col_execute_te:
        st.write("")
        st.write("")
        execute_te = st.button("🚀 分析実行", key="te_btn", type="primary", use_container_width=True)
    
    if execute_te:
        try:
            with st.spinner("Transfer Entropy計算中..."):
                from utils.information_theory_analysis import TransferEntropyAnalyzer
                from utils.analytics_progress import AnalyticsProgressTracker
                
                # 進捗トラッカーを初期化
                tracker_te = AnalyticsProgressTracker("Transfer Entropy分析", total_steps=100)
                
                # progress_callbackを定義（message, pct形式）
                def progress_callback_te(message, pct):
                    tracker_te.update(int(pct * 100), message)
                
                analyzer_te = TransferEntropyAnalyzer(
                    adjacency_matrix=st.session_state.adjacency_matrix,
                    node_names=nodes,
                    n_walks=n_walks,
                    walk_length=walk_length,
                    n_bins=n_bins
                )
                
                result_te = analyzer_te.compute_transfer_entropy(
                    progress_callback=progress_callback_te
                )
                
                if "advanced_analytics_results" not in st.session_state:
                    st.session_state.advanced_analytics_results = {}
                
                st.session_state.advanced_analytics_results["transfer_entropy"] = {
                    "result": result_te,
                    "parameters": {
                        "n_walks": n_walks,
                        "walk_length": walk_length,
                        "n_bins": n_bins
                    },
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # 完了処理
                tracker_te.complete(result_te.computation_time)
        except Exception as e:
            if 'tracker_te' in locals():
                tracker_te.error(str(e))
            st.error(f"❌ Transfer Entropy分析エラー: {str(e)}")
            with st.expander("🔍 エラー詳細"):
                import traceback
                st.code(traceback.format_exc(), language="python")
    
    if "advanced_analytics_results" in st.session_state and "transfer_entropy" in st.session_state.advanced_analytics_results:
        result_data_te = st.session_state.advanced_analytics_results["transfer_entropy"]
        result_te = result_data_te["result"]
        
        st.markdown("---")
        st.subheader("📡 分析結果")
        
        with st.expander("💡 結果の解釈", expanded=True):
            st.markdown(result_te.interpretation)
        
        col_m1_te, col_m2_te, col_m3_te, col_m4_te = st.columns(4)
        with col_m1_te:
            st.metric("総ノード数", len(nodes))
        with col_m2_te:
            avg_te = result_te.te_matrix[result_te.te_matrix > 0].mean() if (result_te.te_matrix > 0).any() else 0
            st.metric("平均TE", f"{avg_te:.3f} bits")
        with col_m3_te:
            st.metric("計算時間", f"{result_te.computation_time:.1f}秒")
        with col_m4_te:
            st.metric("有意フロー数", len(result_te.significant_flows))
        
        st.markdown("### 🔝 有意な情報フロー（上位20）")
        top_20_te = result_te.significant_flows[:20]
        df_top_te = pd.DataFrame([
            {
                "順位": i+1,
                "From": source,
                "To": target,
                "TE (bits)": te_value
            }
            for i, (source, target, te_value) in enumerate(top_20_te)
        ])
        st.dataframe(df_top_te, use_container_width=True, hide_index=True)
        
        import matplotlib.pyplot as plt
        # 日本語フォント設定
        plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        col_viz1_te, col_viz2_te = st.columns(2)
        
        with col_viz1_te:
            st.markdown("### 📊 TE行列ヒートマップ")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(result_te.te_matrix, ax=ax, cmap='Blues',
                       xticklabels=nodes, yticklabels=nodes,
                       cbar_kws={'label': 'Transfer Entropy (bits)'})
            ax.set_title('Transfer Entropy Matrix')
            ax.set_xlabel('To Node')
            ax.set_ylabel('From Node')
            
            st.pyplot(fig)
            plt.close()
        
        with col_viz2_te:
            st.markdown("### 📈 情報流入/流出量")
            
            inflow_vals = [result_te.info_inflow.get(node, 0) for node in nodes[:15]]
            outflow_vals = [result_te.info_outflow.get(node, 0) for node in nodes[:15]]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            x = np.arange(len(nodes[:15]))
            width = 0.35
            
            ax.barh(x - width/2, inflow_vals, width, label='流入量', color='#3498db')
            ax.barh(x + width/2, outflow_vals, width, label='流出量', color='#e74c3c')
            
            ax.set_yticks(x)
            ax.set_yticklabels(nodes[:15])
            ax.set_xlabel('Information Flow (bits)')
            ax.set_title('Top 15 Nodes: Inflow/Outflow')
            ax.legend()
            ax.invert_yaxis()
            
            st.pyplot(fig)
            plt.close()
        
        st.markdown("### 🔍 元の隣接行列との比較")
        st.markdown("元の評価スコアと Transfer Entropy の差異を分析")
        
        comparison_filtered = result_te.comparison_with_original[
            result_te.comparison_with_original["判定"] != "✅ 一致"
        ].head(20)
        
        if len(comparison_filtered) > 0:
            st.dataframe(comparison_filtered, use_container_width=True, hide_index=True)
        else:
            st.info("元の評価スコアとTransfer Entropyは概ね一致しています。")
        
        if result_te.bottleneck_nodes:
            st.markdown("### 🚧 情報ボトルネックノード")
            st.markdown("多くの情報が集中・経由する重要な中継点:")
            for node in result_te.bottleneck_nodes:
                st.markdown(f"- **{node}**")
    
    # 9.3 Bootstrap統計検定
    st.markdown("---")
    st.subheader("9.3. 統計的検定（Bootstrap法）")
    
    with st.expander("💡 この分析について", expanded=False):
        st.markdown("""
        **何がわかるか:**
        
        「この結果は偶然ではない」という統計的根拠を提供します。
        全ての指標に信頼区間と有意性検定を適用します。
        
        **どう使えばいいか:**
        - 分析結果の信頼性評価
        - 経営層への説明資料（統計的根拠付き）
        - 小規模データでも頑健な分析
        
        **結果の見方:**
        - p値 < 0.05 = 統計的に有意（95%信頼）
        - 信頼区間が0をまたがない = 有意な差がある
        """)
    
    st.info(f"⏱️ 推定計算時間: 2-4分（リサンプル1000回）")
    
    col_settings_bs, col_execute_bs = st.columns([2, 1])
    
    with col_settings_bs:
        n_bootstrap = st.slider(
            "リサンプル回数",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="多いほど精度向上、計算時間増加"
        )
        
        alpha = st.slider(
            "有意水準",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            help="0.05 = 95%信頼区間"
        )
    
    with col_execute_bs:
        st.write("")
        st.write("")
        execute_bs = st.button("🚀 検定実行", key="bootstrap_btn", type="primary", use_container_width=True)
    
    if execute_bs:
        try:
            with st.spinner("Bootstrap統計検定中..."):
                from utils.statistical_testing import BootstrapTester
                from utils.analytics_progress import AnalyticsProgressTracker
                
                # 進捗トラッカーを初期化
                tracker_bs = AnalyticsProgressTracker("Bootstrap統計検定", total_steps=100)
                
                # progress_callbackを定義（message, pct形式）
                def progress_callback_bs(message, pct):
                    tracker_bs.update(int(pct * 100), message)
                
                categories_list = SessionManager.get_functional_categories()
                all_idef0 = SessionManager.get_all_idef0_nodes()
                node_groups_bs = {}
                for category in categories_list:
                    if category in all_idef0:
                        idef0_dict = all_idef0[category]
                        for node_type in ['outputs', 'mechanisms', 'inputs']:
                            if node_type in idef0_dict:
                                for node_name in idef0_dict[node_type]:
                                    node_groups_bs[node_name] = category
                
                tester = BootstrapTester(
                    adjacency_matrix=st.session_state.adjacency_matrix,
                    node_names=nodes,
                    node_groups=node_groups_bs,
                    n_bootstrap=n_bootstrap,
                    alpha=alpha
                )
                
                result_bs = tester.run_comprehensive_bootstrap_analysis(
                    metric_name="PageRank",
                    progress_callback=progress_callback_bs
                )
                
                if "advanced_analytics_results" not in st.session_state:
                    st.session_state.advanced_analytics_results = {}
                
                st.session_state.advanced_analytics_results["bootstrap"] = {
                    "result": result_bs,
                    "parameters": {
                        "n_bootstrap": n_bootstrap,
                        "alpha": alpha
                    },
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # 完了処理
                tracker_bs.complete(result_bs.computation_time)
        except Exception as e:
            if 'tracker_bs' in locals():
                tracker_bs.error(str(e))
            st.error(f"❌ Bootstrap統計検定エラー: {str(e)}")
            with st.expander("🔍 エラー詳細"):
                import traceback
                st.code(traceback.format_exc(), language="python")
    
    if "advanced_analytics_results" in st.session_state and "bootstrap" in st.session_state.advanced_analytics_results:
        result_data_bs = st.session_state.advanced_analytics_results["bootstrap"]
        result_bs = result_data_bs["result"]
        
        st.markdown("---")
        st.subheader("📋 検定結果")
        
        with st.expander("💡 結果の解釈", expanded=True):
            st.markdown(result_bs.interpretation)
        
        col_m1_bs, col_m2_bs, col_m3_bs, col_m4_bs = st.columns(4)
        with col_m1_bs:
            st.metric("総ノード数", len(result_bs.node_ci))
        with col_m2_bs:
            st.metric("安定", len(result_bs.stable_findings))
        with col_m3_bs:
            st.metric("不安定", len(result_bs.unstable_findings))
        with col_m4_bs:
            st.metric("リサンプル数", result_bs.n_bootstrap)
        
        st.markdown(f"### 📊 {result_bs.metric_name}の信頼区間（上位15）")
        
        top_15_ci = sorted(result_bs.node_ci.items(), key=lambda x: x[1][0], reverse=True)[:15]
        
        df_ci = pd.DataFrame([
            {
                "順位": i+1,
                "ノード名": node,
                "値": ci[0],
                "下限": ci[1],
                "上限": ci[2],
                "相対誤差%": ((ci[2] - ci[1]) / (2 * abs(ci[0])) * 100) if abs(ci[0]) > 1e-6 else 0
            }
            for i, (node, ci) in enumerate(top_15_ci)
        ])
        st.dataframe(df_ci, use_container_width=True, hide_index=True)
        
        import matplotlib.pyplot as plt
        # 日本語フォント設定
        plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        col_viz1_bs, col_viz2_bs = st.columns(2)
        
        with col_viz1_bs:
            st.markdown("📊 エラーバー付き棒グラフ")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            names = [node for node, _ in top_15_ci]
            values = [ci[0] for _, ci in top_15_ci]
            lower_errors = [ci[0] - ci[1] for _, ci in top_15_ci]
            upper_errors = [ci[2] - ci[0] for _, ci in top_15_ci]
            
            ax.barh(range(len(names)), values, 
                   xerr=[lower_errors, upper_errors],
                   capsize=5, alpha=0.8, color='#3498db')
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel(f'{result_bs.metric_name} ({(1-result_bs.alpha)*100:.0f}% CI)')
            ax.set_title(f'Top 15 {result_bs.metric_name} with Confidence Intervals')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        with col_viz2_bs:
            st.markdown("📉 安定性スコア")
            
            from utils.statistical_testing import compute_stability_score
            stability_df = compute_stability_score(result_bs.node_ci)
            
            top_20_stability = stability_df.head(20)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            rel_errors = top_20_stability["相対誤差"].values
            node_names_stab = top_20_stability["ノード名"].values
            judgments = top_20_stability["判定"].values
            
            colors = ['green' if '安定' in j else 'orange' if 'やや' in j else 'red' for j in judgments]
            
            ax.barh(range(len(node_names_stab)), rel_errors, color=colors, alpha=0.8)
            ax.axvline(0.2, color='green', linestyle='--', linewidth=2, label='安定(<20%)')
            ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='やや不安定(<50%)')
            ax.set_yticks(range(len(node_names_stab)))
            ax.set_yticklabels(node_names_stab)
            ax.set_xlabel('Relative Error')
            ax.set_title('Stability Assessment (lower=more stable)')
            ax.legend()
            ax.invert_yaxis()
            
            st.pyplot(fig)
            plt.close()
        
        if len(result_bs.group_comparison) > 0:
            st.markdown("### 🔍 グループ間比較（Permutation検定）")
            st.dataframe(result_bs.group_comparison, use_container_width=True, hide_index=True)
            
            significant = result_bs.group_comparison[result_bs.group_comparison["有意性"] == "✅ 有意"]
            if len(significant) > 0:
                st.success(f"✅ {len(significant)}組のペアで統計的に有意な差が検出されました（p<{result_bs.alpha}）")
            else:
                st.info("グループ間に統計的に有意な差は検出されませんでした。")
    
    # 9.4 Bayesian Inference
    st.markdown("---")
    st.subheader("9.4. 不確実性定量化（Bayesian Inference）")
    
    with st.expander("💡 この分析について", expanded=False):
        st.markdown("""
        **何がわかるか:**
        
        LLM評価の「信頼性」を数値化します。
        「このスコアは 3.5±0.8」のように不確実性を明示します。
        
        **どう使えばいいか:**
        - 再評価が必要なノードの特定（信頼区間が広い箇所）
        - 意思決定のリスク評価
        - 不確実性を考慮したシナリオ分析
        
        **結果の見方:**
        - 信頼区間が狭い = 評価が安定している
        - 信頼区間が広い = 再評価推奨
        
        **技術背景:**
        Bootstrap-based Bayesian Approximation（簡易版）を使用。
        共役事前分布により解析的に事後分布を計算（MCMC不要）。
        """)
    
    st.info(f"⏱️ 推定計算時間: 1-2分（Bootstrap {len(nodes)}ノード）")
    
    col_settings_bi, col_execute_bi = st.columns([2, 1])
    
    with col_settings_bi:
        st.markdown("**パラメータ設定**")
        
        col_param1_bi, col_param2_bi = st.columns(2)
        
        with col_param1_bi:
            n_bootstrap_bi = st.slider(
                "Bootstrapサンプル数",
                min_value=500,
                max_value=2000,
                value=1000,
                step=100,
                help="多いほど精度向上（計算時間増加）"
            )
        
        with col_param2_bi:
            credible_level_str = st.selectbox(
                "信用区間レベル",
                ["90%", "95%", "99%"],
                index=1,
                help="95%推奨（真の値が区間内にある確率）"
            )
            credible_level_bi = float(credible_level_str.replace("%", "")) / 100.0
    
    with col_execute_bi:
        st.markdown("**実行**")
        if st.button("🚀 Bayesian推論を実行", key="bayesian_btn", use_container_width=True):
            from utils.bayesian_analysis import BayesianAnalyzer
            from utils.analytics_progress import AnalyticsProgressTracker
            
            tracker = AnalyticsProgressTracker("Bayesian Inference分析", total_steps=100)
            
            try:
                analyzer = BayesianAnalyzer(
                    adjacency_matrix=adjacency_matrix,
                    node_names=nodes,
                    n_bootstrap=n_bootstrap_bi,
                    credible_level=credible_level_bi,
                    prior_type='weak_informative'
                )
                
                result_bi = analyzer.compute_bayesian_inference(
                    progress_callback=tracker.update
                )
                
                tracker.complete(result_bi.computation_time)
                
                if "advanced_analytics_results" not in st.session_state:
                    st.session_state.advanced_analytics_results = {}
                
                st.session_state.advanced_analytics_results["bayesian_inference"] = {
                    "result": result_bi,
                    "parameters": {
                        "n_bootstrap": n_bootstrap_bi,
                        "credible_level": credible_level_bi,
                        "prior_type": "weak_informative"
                    },
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.success(f"✅ Bayesian Inference分析完了！（{result_bi.computation_time:.1f}秒）")
                st.rerun()
            
            except Exception as e:
                tracker.error(str(e))
                st.error(f"❌ エラー: {str(e)}")
    
    if "advanced_analytics_results" in st.session_state and "bayesian_inference" in st.session_state.advanced_analytics_results:
        result_bi = st.session_state.advanced_analytics_results["bayesian_inference"]["result"]
        
        st.markdown("### 💡 結果の解釈")
        st.markdown(result_bi.interpretation)
        
        st.markdown("---")
        st.markdown("### 📊 分析メトリクス")
        
        col_metric1_bi, col_metric2_bi, col_metric3_bi, col_metric4_bi = st.columns(4)
        
        with col_metric1_bi:
            st.metric("総エッジ数", result_bi.n_edges)
        
        with col_metric2_bi:
            avg_uncertainty = np.mean(list(result_bi.uncertainty_scores.values())) if result_bi.uncertainty_scores else 0
            st.metric("平均不確実性", f"{avg_uncertainty:.3f}")
        
        with col_metric3_bi:
            n_high_uncertainty = sum(1 for score in result_bi.uncertainty_scores.values() if score > 0.5)
            st.metric("高不確実性エッジ", n_high_uncertainty)
        
        with col_metric4_bi:
            st.metric("計算時間", f"{result_bi.computation_time:.1f}秒")
        
        st.markdown("---")
        st.markdown("### 📈 可視化")
        
        col_viz1_bi, col_viz2_bi = st.columns(2)
        
        with col_viz1_bi:
            st.markdown("📊 不確実性ランキング（上位20エッジ）")
            
            import matplotlib.pyplot as plt
            
            plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            top_20_uncertainty = result_bi.high_uncertainty_edges[:20]
            
            if top_20_uncertainty:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                edge_labels = [f"{s}→{t}" for s, t, _ in top_20_uncertainty]
                uncertainty_values = [score for _, _, score in top_20_uncertainty]
                
                y_pos = np.arange(len(edge_labels))
                
                colors = ['red' if u > 0.7 else 'orange' if u > 0.5 else 'yellow' for u in uncertainty_values]
                
                ax.barh(y_pos, uncertainty_values, color=colors, alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(edge_labels)
                ax.set_xlabel('Uncertainty Score')
                ax.set_title('Top 20 High-Uncertainty Edges')
                ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='High (>0.5)')
                ax.axvline(0.7, color='red', linestyle='--', linewidth=2, label='Very High (>0.7)')
                ax.legend()
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
        
        with col_viz2_bi:
            st.markdown("📋 信用区間テーブル（上位20エッジ）")
            
            credible_pct = int(result_bi.credible_level * 100)
            
            ci_data = []
            for source, target, _ in result_bi.high_uncertainty_edges[:20]:
                edge = (source, target)
                if edge in result_bi.credible_intervals:
                    mean_val, lower, upper = result_bi.credible_intervals[edge]
                    uncertainty = result_bi.uncertainty_scores.get(edge, 0)
                    
                    if uncertainty > 0.7:
                        status = "❌ 非常に不安定"
                    elif uncertainty > 0.5:
                        status = "⚠️ 不安定"
                    elif uncertainty > 0.3:
                        status = "⚡ やや不安定"
                    else:
                        status = "✅ 安定"
                    
                    ci_data.append({
                        "From": source,
                        "To": target,
                        "事後平均": f"{mean_val:.2f}",
                        f"下限{credible_pct}%": f"{lower:.2f}",
                        f"上限{credible_pct}%": f"{upper:.2f}",
                        "不確実性": f"{uncertainty:.3f}",
                        "判定": status
                    })
            
            ci_df = pd.DataFrame(ci_data)
            st.dataframe(ci_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 📊 事後分布の可視化（上位10エッジ）")
        
        if len(result_bi.high_uncertainty_edges) > 0:
            top_10_edges = result_bi.high_uncertainty_edges[:10]
            
            n_rows = (len(top_10_edges) + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, (source, target, _) in enumerate(top_10_edges):
                row = idx // 2
                col = idx % 2
                ax = axes[row, col]
                
                edge = (source, target)
                if edge in result_bi.credible_intervals:
                    mean_val, lower, upper = result_bi.credible_intervals[edge]
                    std_val = result_bi.posterior_std.get(edge, 1.0)
                    
                    x = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 200)
                    y = stats.norm.pdf(x, mean_val, std_val)
                    
                    ax.plot(x, y, 'b-', linewidth=2, label='Posterior')
                    ax.axvline(mean_val, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.2f}')
                    ax.axvline(lower, color='orange', linestyle='--', linewidth=1.5, label=f'CI: [{lower:.2f}, {upper:.2f}]')
                    ax.axvline(upper, color='orange', linestyle='--', linewidth=1.5)
                    ax.fill_between(x, y, where=(x >= lower) & (x <= upper), alpha=0.3, color='orange')
                    
                    ax.set_title(f"{source} → {target}")
                    ax.set_xlabel("Score")
                    ax.set_ylabel("Density")
                    ax.legend(fontsize=8)
                    ax.grid(alpha=0.3)
            
            for idx in range(len(top_10_edges), n_rows * 2):
                row = idx // 2
                col = idx % 2
                fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # 9.5 Causal Inference
    st.markdown("---")
    st.subheader("9.5. 因果推論（Pearl's Causal Inference）")
    
    with st.expander("💡 この分析について", expanded=False):
        st.markdown("""
        **何がわかるか:**
        
        「もしこのノードを改善したら、全体がどう変わるか」を予測します。
        相関ではなく因果関係を推定します。
        
        **どう使えばいいか:**
        - プロセス改善のシミュレーション
        - 投資効果の事前予測
        - 反事実分析（「もしあの時...」）
        - 因果経路の可視化
        - 交絡因子の検出
        
        **結果の見方:**
        - 介入効果: do(X=改善) → Y が 15%向上
        - 直接効果 vs 間接効果の比較
        - 因果経路の特定
        """)
    
    st.info(f"⏱️ 推定計算時間: 3-7分（{len(nodes)}ノード）")
    
    col_settings_ci, col_execute_ci = st.columns([2, 1])
    
    with col_settings_ci:
        intervention_node = st.selectbox(
            "介入対象ノード",
            options=nodes,
            help="このノードに介入（改善）した場合の効果を分析"
        )
        
        intervention_strength = st.slider(
            "介入の強さ",
            min_value=0.5,
            max_value=2.0,
            value=1.5,
            step=0.1,
            help="1.0=現状、1.5=50%改善、0.5=50%劣化"
        )
        
        max_path_length = st.slider(
            "最大経路長",
            min_value=2,
            max_value=6,
            value=4,
            help="分析する因果経路の最大長"
        )
    
    with col_execute_ci:
        st.write("")
        st.write("")
        execute_ci = st.button("🚀 分析実行", key="ci_btn", type="primary", use_container_width=True)
    
    if execute_ci:
        try:
            with st.spinner("因果推論分析中..."):
                from utils.causal_inference import CausalInferenceAnalyzer
                from utils.analytics_progress import AnalyticsProgressTracker
                
                tracker_ci = AnalyticsProgressTracker("因果推論分析", total_steps=100)
                
                def progress_callback_ci(message, pct):
                    tracker_ci.update(int(pct * 100), message)
                
                analyzer_ci = CausalInferenceAnalyzer(
                    adjacency_matrix=st.session_state.adjacency_matrix,
                    node_names=nodes,
                    max_path_length=max_path_length
                )
                
                result_ci = analyzer_ci.compute_causal_inference(
                    intervention_node=intervention_node,
                    intervention_strength=intervention_strength,
                    progress_callback=progress_callback_ci
                )
                
                if "advanced_analytics_results" not in st.session_state:
                    st.session_state.advanced_analytics_results = {}
                
                st.session_state.advanced_analytics_results["causal_inference"] = {
                    "result": result_ci,
                    "parameters": {
                        "intervention_node": intervention_node,
                        "intervention_strength": intervention_strength,
                        "max_path_length": max_path_length
                    },
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                tracker_ci.complete(result_ci.computation_time)
        except Exception as e:
            if 'tracker_ci' in locals():
                tracker_ci.error(str(e))
            st.error(f"❌ 因果推論分析エラー: {str(e)}")
            with st.expander("🔍 エラー詳細"):
                import traceback
                st.code(traceback.format_exc(), language="python")
    
    if "advanced_analytics_results" in st.session_state and "causal_inference" in st.session_state.advanced_analytics_results:
        result_data_ci = st.session_state.advanced_analytics_results["causal_inference"]
        result_ci = result_data_ci["result"]
        
        st.markdown("---")
        st.subheader("📊 分析結果")
        
        st.markdown(result_ci.interpretation)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("直接効果", len(result_ci.direct_effects))
        col2.metric("間接効果", len(result_ci.indirect_effects))
        col3.metric("交絡因子", len(result_ci.confounders))
        col4.metric("計算時間", f"{result_ci.computation_time:.1f}秒")
        
        st.markdown("### 🎯 介入効果")
        intervention_node_param = result_data_ci["parameters"]["intervention_node"]
        intervention_effects = result_ci.intervention_effects.get(intervention_node_param, {})
        
        if intervention_effects:
            import matplotlib.pyplot as plt
            # 日本語フォント設定
            plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            effects_df = pd.DataFrame([
                {"ノード": node, "因果効果": effect}
                for node, effect in sorted(intervention_effects.items(), 
                                          key=lambda x: abs(x[1]), reverse=True)[:15]
            ])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['red' if x < 0 else 'green' for x in effects_df["因果効果"]]
            ax.barh(effects_df["ノード"], effects_df["因果効果"], color=colors, alpha=0.7)
            ax.set_xlabel("因果効果")
            ax.set_title(f"do({intervention_node_param}) の波及効果")
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            st.pyplot(fig)
            plt.close()
        
        st.markdown("### 🛤️ 因果経路（上位5ペア）")
        
        top_pairs = sorted(result_ci.total_effects.items(), 
                          key=lambda x: abs(x[1]), reverse=True)[:5]
        
        for (source, target), effect in top_pairs:
            paths = result_ci.causal_paths.get((source, target), [])
            if paths:
                st.markdown(f"**{source} → {target}** (総効果: {effect:.4f})")
                for i, path in enumerate(paths[:3], 1):
                    path_str = " → ".join(path)
                    st.caption(f"経路{i}: {path_str}")
        
        if result_ci.confounders:
            st.markdown("### ⚠️ 交絡因子")
            confounders_df = pd.DataFrame([
                {
                    "From": source,
                    "To": target,
                    "交絡因子": ", ".join(conf_list)
                }
                for source, target, conf_list in result_ci.confounders[:10]
            ])
            st.dataframe(confounders_df, use_container_width=True, hide_index=True)
        
        st.markdown("### 🏆 最適な介入ターゲット（上位10）")
        if result_ci.top_intervention_targets:
            targets_df = pd.DataFrame([
                {"順位": i+1, "ノード": node, "総影響力": impact}
                for i, (node, impact) in enumerate(result_ci.top_intervention_targets[:10])
            ])
            st.dataframe(targets_df, use_container_width=True, hide_index=True)
    
    # 9.6 Graph Embedding
    st.markdown("---")
    st.subheader("9.6. 潜在構造発見（Graph Embedding + Community Detection）")
    
    with st.expander("💡 この分析について", expanded=False):
        st.markdown("""
        **何がわかるか:**
        
        表面的な接続を超えた「本質的な類似性」を発見します。
        機能的なグループを自動検出します。
        
        **どう使えばいいか:**
        - カテゴリを超えた自然なグループ分け
        - 類似ノードの統合・整理
        - 2D可視化で直感的理解
        
        **結果の見方:**
        - 近くに配置されたノード = 機能的に類似
        - 同じ色のコミュニティ = 協力関係が強い
        """)
    
    st.info(f"⏱️ 推定計算時間: 1-2分（{len(nodes)}ノード）")
    
    # パラメータ設定
    col_settings_ge, col_execute_ge = st.columns([2, 1])
    
    with col_settings_ge:
        embedding_dim = st.select_slider(
            "埋め込み次元数",
            options=[16, 32, 64, 128],
            value=64,
            help="ノードを表現するベクトルの次元数（大きいほど詳細だが計算時間増）"
        )
        
        col_walk_len, col_walk_num = st.columns(2)
        with col_walk_len:
            walk_length = st.selectbox(
                "ウォーク長",
                options=[10, 20, 30],
                index=1,
                help="ランダムウォークの最大長"
            )
        with col_walk_num:
            num_walks = st.selectbox(
                "ウォーク回数",
                options=[50, 100, 200, 500],
                index=1,
                help="各ノードから開始するウォーク数"
            )
        
        reduction_method = st.selectbox(
            "2D化手法",
            options=["mds", "spectral"],
            format_func=lambda x: "MDS（多次元尺度法）" if x == "mds" else "Spectral Embedding",
            help="高次元埋め込みを2Dに圧縮する手法"
        )
    
    with col_execute_ge:
        st.write("")  # スペース調整
        st.write("")
        execute_ge = st.button("🚀 分析実行", key="embedding_execute_btn", use_container_width=True)
    
    # 実行
    if execute_ge:
        try:
            from utils.graph_embedding import GraphEmbeddingAnalyzer
            from utils.analytics_progress import AnalyticsProgressTracker, create_simple_callback
            
            tracker_ge = AnalyticsProgressTracker("Graph Embedding分析", total_steps=100)
            
            # 進捗コールバック
            def progress_callback_ge(message: str, pct: float):
                tracker_ge.progress_text.text(message)
                tracker_ge.progress_bar.progress(pct)
            
            analyzer_ge = GraphEmbeddingAnalyzer(
                adjacency_matrix=st.session_state.adjacency_matrix,
                node_names=nodes,
                embedding_dim=embedding_dim,
                walk_length=walk_length,
                num_walks=num_walks,
                reduction_method=reduction_method
            )
            
            result_ge = analyzer_ge.compute_graph_embedding(progress_callback=progress_callback_ge)
            tracker_ge.complete(result_ge.computation_time)
            
            # セッションステートに保存
            if "advanced_analytics_results" not in st.session_state:
                st.session_state.advanced_analytics_results = {}
            
            st.session_state.advanced_analytics_results["graph_embedding"] = {
                "result": result_ge,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "parameters": {
                    "embedding_dim": embedding_dim,
                    "walk_length": walk_length,
                    "num_walks": num_walks,
                    "reduction_method": reduction_method
                }
            }
        except Exception as e:
            if 'tracker_ge' in locals():
                tracker_ge.error(str(e))
            st.error(f"❌ Graph Embedding分析エラー: {str(e)}")
            with st.expander("🔍 エラー詳細"):
                import traceback
                st.code(traceback.format_exc(), language="python")
    
    # 結果表示
    if "advanced_analytics_results" in st.session_state and \
       "graph_embedding" in st.session_state.advanced_analytics_results:
        
        ge_data = st.session_state.advanced_analytics_results["graph_embedding"]
        result_ge = ge_data["result"]
        
        st.markdown("---")
        st.markdown("### 📊 分析結果")
        
        # 1. 解釈文
        with st.expander("💡 結果の解釈", expanded=True):
            st.markdown(result_ge.interpretation)
        
        # 2. メトリクス
        st.markdown("#### 基本統計")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("コミュニティ数", result_ge.n_communities)
        col2.metric("Modularity", f"{result_ge.modularity:.3f}")
        col3.metric("埋め込み次元", result_ge.embedding_dim)
        col4.metric("計算時間", f"{result_ge.computation_time:.1f}秒")
        
        # 3. 2D散布図（コミュニティ別色分け）
        st.markdown("#### 2D可視化（コミュニティ別）")
        
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        # 日本語フォント設定
        plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # コミュニティごとに色を割り当て
        unique_communities = sorted(set(result_ge.communities.values()))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_communities)))
        community_colors = {comm_id: colors[i] for i, comm_id in enumerate(unique_communities)}
        
        # プロット
        for node in nodes:
            x, y = result_ge.node_positions_2d[node]
            comm_id = result_ge.communities[node]
            ax.scatter(x, y, c=[community_colors[comm_id]], s=200, alpha=0.7, edgecolors='black', linewidths=1.5)
            ax.annotate(node, (x, y), fontsize=9, ha='center', va='center')
        
        ax.set_xlabel("次元1", fontsize=12)
        ax.set_ylabel("次元2", fontsize=12)
        ax.set_title(f"Graph Embedding 2D可視化（{result_ge.n_communities}コミュニティ）", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
        
        # 4. コミュニティ詳細
        st.markdown("#### コミュニティ詳細")
        
        # コミュニティごとにメンバーを整理
        community_members = {}
        for node, comm_id in result_ge.communities.items():
            if comm_id not in community_members:
                community_members[comm_id] = []
            community_members[comm_id].append(node)
        
        # 各コミュニティの情報を表示
        comm_data = []
        for comm_id in sorted(community_members.keys()):
            members = community_members[comm_id]
            label = result_ge.community_labels.get(comm_id, f"コミュニティ{comm_id+1}")
            comm_data.append({
                "コミュニティID": comm_id + 1,
                "名前": label,
                "ノード数": len(members),
                "メンバー": ", ".join(members)
            })
        
        comm_df = pd.DataFrame(comm_data)
        st.dataframe(comm_df, use_container_width=True, hide_index=True)
        
        # 5. 類似ノードペア
        st.markdown("#### 類似ノードペア（上位20組）")
        
        similar_data = []
        for node1, node2, sim in result_ge.top_similar_pairs[:20]:
            similar_data.append({
                "ノード1": node1,
                "ノード2": node2,
                "類似度": f"{sim:.4f}"
            })
        
        similar_df = pd.DataFrame(similar_data)
        st.dataframe(similar_df, use_container_width=True, hide_index=True)
        
        # 注意事項
        st.info("""
        **💡 活用のヒント:**
        - 同じコミュニティ内のノードは機能的に密接に関係しています
        - 類似度が高いノードペアは、統合や整理の候補となります
        - 2D可視化で離れた位置にあるノードは、機能的に独立しています
        """)
    else:
        st.info("👆 上の「🚀 分析実行」ボタンをクリックして、Graph Embedding分析を開始してください。")
    
    # 9.7 Fisher Information
    st.markdown("---")
    st.subheader("9.7. 感度分析（Fisher Information Matrix）")
    
    with st.expander("💡 この分析について", expanded=False):
        st.markdown("""
        **何がわかるか:**
        
        「どのスコアが不正確だと全体が大きく歪むか」を特定します。
        推定精度の理論限界を計算します。
        
        **どう使えばいいか:**
        - 再評価の優先順位決定（感度が高いノードを優先）
        - 最適実験計画（どこを精密に測定すべきか）
        - パラメータ推定の信頼性評価
        
        **結果の見方:**
        - Fisher情報量が高い = そのノードが全体に大きく影響
        - Cramér-Rao下限 = 推定精度の理論限界
        """)
    
    st.info(f"⏱️ 推定計算時間: <1分（{len(nodes)}ノード）")
    
    # パラメータ設定
    col_settings_fi, col_execute_fi = st.columns([2, 1])
    
    with col_settings_fi:
        noise_variance_fi = st.slider(
            "ノイズ分散（σ²）",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="観測ノイズの分散を仮定します（大きいほど不確実性が高い）"
        )
        
        top_k_fi = st.slider(
            "表示する上位エッジ数",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="感度が高いエッジを何組表示するか"
        )
    
    with col_execute_fi:
        st.write("")  # スペース調整
        st.write("")
        execute_fi = st.button("🚀 分析実行", key="fisher_execute_btn", use_container_width=True)
    
    # 実行
    if execute_fi:
        try:
            from utils.fisher_information import FisherInformationAnalyzer
            from utils.analytics_progress import AnalyticsProgressTracker
            
            tracker_fi = AnalyticsProgressTracker("Fisher Information分析", total_steps=100)
            
            # 進捗コールバック
            def progress_callback_fi(message: str, pct: float):
                tracker_fi.progress_text.text(message)
                tracker_fi.progress_bar.progress(pct)
            
            analyzer_fi = FisherInformationAnalyzer(
                adjacency_matrix=st.session_state.adjacency_matrix,
                node_names=nodes,
                noise_variance=noise_variance_fi
            )
            
            result_fi = analyzer_fi.compute_fisher_information(progress_callback=progress_callback_fi)
            tracker_fi.complete(result_fi.computation_time)
            
            # セッションステートに保存
            if "advanced_analytics_results" not in st.session_state:
                st.session_state.advanced_analytics_results = {}
            
            st.session_state.advanced_analytics_results["fisher_information"] = {
                "result": result_fi,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "parameters": {
                    "noise_variance": noise_variance_fi,
                    "top_k": top_k_fi
                }
            }
        except Exception as e:
            if 'tracker_fi' in locals():
                tracker_fi.error(str(e))
            st.error(f"❌ Fisher Information分析エラー: {str(e)}")
            with st.expander("🔍 エラー詳細"):
                import traceback
                st.code(traceback.format_exc(), language="python")
    
    # 結果表示
    if "advanced_analytics_results" in st.session_state and \
       "fisher_information" in st.session_state.advanced_analytics_results:
        
        fi_data = st.session_state.advanced_analytics_results["fisher_information"]
        result_fi = fi_data["result"]
        
        st.markdown("---")
        st.markdown("### 📊 分析結果")
        
        # 1. 解釈文
        with st.expander("💡 結果の解釈", expanded=True):
            st.markdown(result_fi.interpretation)
        
        # 2. メトリクス
        st.markdown("#### 基本統計")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("総エッジ数", result_fi.n_edges)
        col2.metric("条件数", f"{result_fi.condition_number:.2f}")
        col3.metric("実効ランク", result_fi.effective_rank)
        col4.metric("計算時間", f"{result_fi.computation_time:.1f}秒")
        
        # 3. 感度スコアランキング
        st.markdown("#### 感度スコアランキング（上位20）")
        
        import matplotlib.pyplot as plt
        # 日本語フォント設定
        plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        top_edges = result_fi.top_sensitive_edges[:20]
        edge_labels = [f"{s}→{t}" for s, t, _ in top_edges]
        sensitivities = [score for _, _, score in top_edges]
        
        y_pos = np.arange(len(edge_labels))
        ax.barh(y_pos, sensitivities, color='steelblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(edge_labels, fontsize=9)
        ax.set_xlabel("感度スコア", fontsize=12)
        ax.set_title("Fisher情報量（感度）ランキング", fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        st.pyplot(fig)
        plt.close()
        
        # 4. Cramér-Rao下限テーブル
        st.markdown("#### Cramér-Rao下限（推定精度限界、上位20）")
        
        # CR下限を降順ソート（大きい = 推定が困難）
        cr_sorted = sorted(
            result_fi.cramer_rao_bounds.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        cr_data = []
        for (source, target), bound in cr_sorted:
            cr_data.append({
                "From": source,
                "To": target,
                "CR下限": f"{bound:.6f}",
                "推定難易度": "高" if bound > np.mean(list(result_fi.cramer_rao_bounds.values())) else "中"
            })
        
        cr_df = pd.DataFrame(cr_data)
        st.dataframe(cr_df, use_container_width=True, hide_index=True)
        
        # 5. 固有値分布
        st.markdown("#### 固有値分布（Scree Plot）")
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        
        eigenvalues = result_fi.eigenvalues
        ax2.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', color='darkblue', linewidth=2, markersize=6)
        ax2.set_xlabel("固有値のインデックス", fontsize=12)
        ax2.set_ylabel("固有値", fontsize=12)
        ax2.set_title("Fisher情報行列の固有値分布", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # 対数スケール
        
        st.pyplot(fig2)
        plt.close()
        
        # 注意事項
        st.info("""
        **💡 活用のヒント:**
        - 感度スコアが高いエッジは、再評価の優先順位が高いです
        - CR下限が大きいエッジは、推定が本質的に困難です（追加データが必要）
        - 条件数が大きい場合は、多重共線性がある可能性があります
        """)
    else:
        st.info("👆 上の「🚀 分析実行」ボタンをクリックして、Fisher Information分析を開始してください。")


def main() -> None:
    """メインアプリケーション"""
    
    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    SessionManager.initialize()
    
    st.title(f"{settings.APP_TITLE} - タブ形式UI")
    
    render_sidebar()
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📝 ステップ1: プロセス定義",
        "🎯 ステップ2: 機能カテゴリ",
        "🔧 ステップ3: ノード定義",
        "⚖️ ステップ4: ノード影響評価",
        "📈 ステップ5: 行列分析",
        "📊 ステップ6: ネットワーク可視化",
        "🔬 ステップ7: ネットワーク分析",
        "🎮 ステップ8: DSM最適化",
        "🧬 ステップ9: 高度な分析"
    ])
    
    with tab1:
        tab1_process_definition()
    
    with tab2:
        tab2_functional_categories()
    
    with tab3:
        tab3_node_definition()
    
    with tab4:
        tab4_node_evaluation()
    
    with tab5:
        tab5_matrix_analysis()
    
    with tab6:
        tab6_network_visualization()
    
    with tab7:
        tab7_network_analysis()
    
    with tab8:
        tab8_dsm_optimization()
    
    with tab9:
        tab9_advanced_analytics()


if __name__ == "__main__":
    main()
