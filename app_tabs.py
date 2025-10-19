"""
Process Insight Modeler (PIM) - タブ形式UI
生産プロセスの暗黙知を形式知に変換するアプリケーション
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAIError
from config.settings import settings
from core.session_manager import SessionManager
from core.llm_client import LLMClient
from core.data_models import (
    FunctionalCategory,
    CategoryGenerationOptions,
    CategorySet
)


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
    """タブ4: ノード影響評価（論理ルールベース + LLMバッチ評価）"""
    from utils.idef0_classifier import (
        generate_zigzagging_pairs,
        get_phase_statistics
    )
    from utils.evaluation_filter import (
        filter_pairs_by_logic,
        get_batch_summary,
        apply_default_scores
    )
    
    st.header("⚖️ ステップ4: ノード間影響評価（論理ルールベース + LLMバッチ評価）")
    
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
    **論理ルールベース評価システム**
    
    IDEF0構造とZigzagging手法に基づき、効率的かつ一貫性のある評価を実現します。
    
    ### 評価戦略
    
    1. **論理的フィルタリング**: カテゴリ間距離と評価フェーズに基づき、評価の必要性を自動判定
    2. **LLMバッチ評価**: 同一カテゴリ内のペアをまとめて評価（効率化 + 全体把握）
    3. **疎行列の厳守**: 直接的で強い影響のみを非ゼロとし、間接的影響は0
    
    評価スケール: **-9（強い負）** ～ **0（無関係）** ～ **+9（強い正）**
    """)
    
    # セッションステート初期化
    if "evaluation_pairs" not in st.session_state:
        st.session_state.evaluation_pairs = []
    if "filtered_results" not in st.session_state:
        st.session_state.filtered_results = None
    if "batch_evaluation_done" not in st.session_state:
        st.session_state.batch_evaluation_done = False
    
    # ステップ1: 評価ペア生成とフィルタリング
    if not st.session_state.evaluation_pairs:
        st.subheader("📋 ステップ1: 評価ペア生成と論理フィルタリング")
        
        st.markdown(f"""
        **現在のノード数**: {len(nodes)}個
        **カテゴリ数**: {len(categories)}個
        
        論理ルールに基づき、評価が必要なペアのみを抽出します。
        """)
        
        if st.button("🔄 評価ペア生成 + フィルタリング実行", type="primary", key="generate_pairs_btn"):
            try:
                # 全ペア生成
                all_pairs = generate_zigzagging_pairs(nodes, idef0_nodes)
                st.session_state.evaluation_pairs = all_pairs
                
                # 論理フィルタリング実行
                filtered = filter_pairs_by_logic(all_pairs, idef0_nodes, categories)
                st.session_state.filtered_results = filtered
                
                stats = filtered["statistics"]
                
                st.success(f"✅ 全{stats['total_pairs']}件のペアを生成し、論理フィルタリングを完了しました")
                
                st.markdown("### 📊 フィルタリング結果")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("必須評価（同一カテゴリ）", stats["must_evaluate"], 
                             help="同一カテゴリ内のペア - LLMバッチ評価必須")
                with col2:
                    st.metric("推奨評価（隣接カテゴリ）", stats["should_evaluate"],
                             help="隣接カテゴリ間のペア - 評価推奨")
                with col3:
                    st.metric("デフォルト0", stats["default_zero"],
                             help="論理的に影響なしと判定 - 自動的に0")
                
                reduction = stats.get("reduction_rate", 0)
                st.info(f"💡 評価作業量を **{reduction:.1f}%** 削減しました")
                st.info("💡 次は「ステップ2: LLMバッチ評価」に進んでください。")
                
            except Exception as e:
                st.error(f"❌ 評価ペア生成エラー: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        return
    
    # ステップ2: LLMバッチ評価実行
    if not st.session_state.batch_evaluation_done:
        filtered = st.session_state.filtered_results
        if not filtered:
            st.error("⚠️ フィルタリング結果が見つかりません。最初からやり直してください。")
            return
        
        st.markdown("---")
        st.subheader("🤖 ステップ2: LLMバッチ評価実行")
        
        stats = filtered["statistics"]
        must_eval = filtered["must_evaluate"]
        should_eval = filtered["should_evaluate"]
        default_zero = filtered["default_zero"]
        category_batches = filtered["category_batches"]
        
        batch_summary = get_batch_summary(category_batches)
        active_batches = [b for b in batch_summary if b["pair_count"] > 0]
        
        st.markdown(f"""
        **必須評価ペア**: {stats["must_evaluate"]}件
        **カテゴリバッチ数**: {len(active_batches)}個
        
        各カテゴリのIDEF0構造を把握しながら、同一カテゴリ内のペアをまとめて評価します。
        """)
        
        # カテゴリバッチサマリー表示
        if active_batches:
            with st.expander("📋 カテゴリ別ペア数", expanded=False):
                for batch in active_batches:
                    st.markdown(f"**{batch['category']}**: {batch['pair_count']}ペア")
        
        if st.button("🚀 LLMバッチ評価を開始", type="primary", key="start_batch_eval"):
            try:
                llm_client = LLMClient()
                all_results = []
                
                # デフォルト0のペアを自動追加
                default_results = apply_default_scores(default_zero)
                all_results.extend(default_results)
                
                # プログレスバー
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_batches = len(active_batches)
                
                for batch_idx, batch in enumerate(active_batches):
                    category = batch["category"]
                    pair_count = batch["pair_count"]
                    
                    status_text.text(f"カテゴリ '{category}' を評価中... ({batch_idx + 1}/{total_batches})")
                    
                    # このカテゴリのペアとIDEF0データを取得
                    category_pairs = category_batches[category]
                    idef0_data = idef0_nodes.get(category, {})
                    
                    # LLMバッチ評価実行
                    with st.spinner(f"🤖 LLMが{pair_count}ペアを評価中..."):
                        batch_results = llm_client.evaluate_category_batch(
                            category=category,
                            idef0_data=idef0_data,
                            pairs=category_pairs,
                            process_name=process_name
                        )
                    
                    all_results.extend(batch_results)
                    
                    # プログレス更新
                    progress = (batch_idx + 1) / total_batches
                    progress_bar.progress(progress)
                
                # 全評価結果をセッションに保存
                for result in all_results:
                    SessionManager.add_evaluation(
                        from_node=result["from_node"],
                        to_node=result["to_node"],
                        score=result["score"],
                        reason=result["reason"]
                    )
                
                st.session_state.batch_evaluation_done = True
                
                status_text.text("")
                progress_bar.empty()
                
                st.success(f"✅ 全{len(all_results)}件の評価が完了しました！")
                st.info("💡 下の「ステップ3: 評価結果確認」で詳細を確認し、ステップ5に進んでください。")
                
            except Exception as e:
                st.error(f"❌ バッチ評価エラー: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # ステップ2.5: Zigzagging推論（オプション機能）
        st.markdown("---")
        st.subheader("🔍 ステップ2.5: Zigzagging推論（オプション）")
        
        st.markdown("""
        **論理ルールベースで「デフォルト0」と判定されたペアの中から、Zigzagging思考プロセスで論理的な依存関係を探索します。**
        
        - 離れた工程間でも、**How関係（どのように貢献するか）**が明確なペアを発見
        - 「疎で階層的」な構造は維持（間接的な関係は除外）
        - 処理時間: 数分～10分程度（ペア数に依存）
        """)
        
        default_zero = filtered.get("default_zero", [])
        
        st.info(f"📊 デフォルト0と判定されたペア数: {len(default_zero)}件")
        
        enable_zigzagging = st.checkbox(
            "🔬 Zigzagging推論を有効化する（オプション機能）",
            value=False,
            help="離れた工程間の論理的な依存関係を探索します。処理時間がかかりますが、精度が向上します。"
        )
        
        if enable_zigzagging:
            if st.button("🚀 Zigzagging推論を実行", type="secondary", key="start_zigzag"):
                try:
                    llm_client = LLMClient()
                    
                    st.info(f"🔍 {len(default_zero)}件のペアをZigzagging推論で分析中...")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Zigzagging推論実行中...")
                    
                    with st.spinner("🤖 LLMがHow関係を推論中..."):
                        zigzag_results = llm_client.zigzagging_inference_for_distant_pairs(
                            distant_pairs=default_zero,
                            idef0_nodes=idef0_nodes,
                            process_name=process_name,
                            max_pairs_per_batch=30
                        )
                    
                    progress_bar.progress(1.0)
                    
                    # 見つかった関係を既存の評価に追加
                    if zigzag_results:
                        for result in zigzag_results:
                            SessionManager.add_evaluation(
                                from_node=result["from_node"],
                                to_node=result["to_node"],
                                score=result["score"],
                                reason=result.get("reason", "")  # 空文字列がデフォルト
                            )
                        
                        status_text.text("")
                        progress_bar.empty()
                        
                        st.success(f"✅ Zigzagging推論完了！{len(zigzag_results)}件の論理的な依存関係を発見しました")
                        
                        # 発見した関係を表示
                        with st.expander("🔎 発見された論理的依存関係", expanded=True):
                            for result in zigzag_results[:10]:  # 最初の10件
                                score = result["score"]
                                score_color = "green" if score > 0 else "red"
                                st.markdown(f"**{result['from_node']}** → **{result['to_node']}**: :{score_color}[{score:+d}]")
                                st.caption(result["reason"])
                                st.markdown("---")
                        
                        st.info("💡 新たに発見された依存関係が評価に反映されました。")
                    else:
                        status_text.text("")
                        progress_bar.empty()
                        st.info("ℹ️ 新たな論理的依存関係は発見されませんでした。現在の疎行列が維持されます。")
                
                except Exception as e:
                    st.error(f"❌ Zigzagging推論エラー: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        return
    
    # ステップ3: 評価結果確認
    st.markdown("---")
    st.subheader("✅ ステップ3: 評価結果確認")
    
    evaluations = SessionManager.get_evaluations()
    
    if not evaluations:
        st.warning("⚠️ 評価結果がありません。")
        return
    
    st.success(f"🎉 全{len(evaluations)}件の評価が完了しました！")
    
    # 非ゼロのペアのみ抽出
    import pandas as pd
    
    non_zero_evals = [e for e in evaluations if e.get("score", 0) != 0]
    
    st.metric("非ゼロ評価ペア", f"{len(non_zero_evals)} / {len(evaluations)}")
    st.caption(f"疎行列率: {100 * (1 - len(non_zero_evals) / len(evaluations)):.1f}% がゼロ")
    
    # 高スコアペアの表示
    if non_zero_evals:
        with st.expander("🔥 高スコアペア（|score| ≥ 5）", expanded=True):
            high_score_evals = [e for e in non_zero_evals if abs(e.get("score", 0)) >= 5]
            
            if high_score_evals:
                # スコアでソート
                high_score_evals_sorted = sorted(high_score_evals, key=lambda x: abs(x.get("score", 0)), reverse=True)
                
                for eval_item in high_score_evals_sorted[:20]:  # 上位20件
                    score = eval_item.get("score", 0)
                    score_color = "green" if score > 0 else "red"
                    
                    st.markdown(f"**{eval_item['from_node']}** → **{eval_item['to_node']}**: :{score_color}[{score:+d}]")
                    st.caption(eval_item.get("reason", ""))
                    st.markdown("---")
            else:
                st.info("スコア絶対値5以上のペアはありません。")
    
    st.markdown("---")
    st.markdown("### 次のステップ")
    st.info("👉 **タブ5** で隣接行列とヒートマップを確認できます。")
    
    st.markdown("---")
    st.subheader("🗑️ リセット")
    
    col_reset1, col_reset2 = st.columns(2)
    with col_reset1:
        if st.button("🔄 評価ペアをリセット", key="reset_pairs_btn"):
            st.session_state.evaluation_pairs = []
            st.session_state.filtered_results = None
            st.session_state.batch_evaluation_done = False
            st.info("🔄 評価ペアをリセットしました。「ステップ1」から再実行してください。")
    with col_reset2:
        if st.button("🗑️ 評価結果をクリア", key="clear_evals_btn"):
            if "evaluations" in st.session_state:
                st.session_state.evaluations = {}
            st.session_state.batch_evaluation_done = False
            st.info("🗑️ 評価結果をクリアしました。")


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
            with st.spinner("LLMがパラメータを評価中..."):
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
                    
                    # LLM評価
                    result = llm_client.evaluate_dsm_parameters(
                        process_name=SessionManager.get_process_name(),
                        process_description=SessionManager.get_process_description(),
                        nodes=nodes,
                        idef0_nodes=all_idef0,
                        node_classifications=node_classifications
                    )
                    
                    # セッションに保存
                    st.session_state.dsm_llm_params = result
                    
                    st.success("✅ LLMによるパラメータ評価が完了しました")
                    
                except Exception as e:
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
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        step1_pop = st.slider("個体数", 50, 500, 200, 50, key="step1_pop")
    with col_p2:
        step1_gen = st.slider("世代数", 20, 200, 50, 10, key="step1_gen")
    
    if st.button("🚀 STEP-1を実行", type="primary", use_container_width=True):
        with st.spinner(f"NSGA-II最適化中（{step1_gen}世代）..."):
            try:
                from utils.dsm_optimizer import PIMDSMData, PIMStep1NSGA2
                import time
                
                start_time = time.time()
                
                # データ構築
                llm_params = st.session_state.get("dsm_llm_params") if param_mode == "llm_auto" else None
                
                dsm_data = PIMDSMData(
                    adj_matrix_df=adj_matrix_df,
                    nodes=nodes,
                    idef0_nodes=all_idef0,
                    param_mode=param_mode,
                    llm_params=llm_params,
                    custom_params=None  # 将来的に手動カスタムで使用
                )
                
                # STEP-1実行
                step1 = PIMStep1NSGA2(dsm_data)
                pareto_front = step1.run(n_pop=step1_pop, n_gen=step1_gen)
                
                elapsed = time.time() - start_time
                
                # パレートフロントのデータを抽出
                step1_results = []
                for ind in pareto_front:
                    cost, freedom_inv = ind.fitness.values
                    removed_indices = [i for i, val in enumerate(ind) if val == 1]
                    removed_nodes = [dsm_data.reordered_nodes[i] for i in removed_indices]
                    step1_results.append({
                        'individual': ind,
                        'cost': cost,
                        'freedom_inv': freedom_inv,
                        'freedom': 1/freedom_inv if freedom_inv != float('inf') else 0,
                        'removed_count': len(removed_nodes),
                        'removed_nodes': removed_nodes
                    })
                
                # セッションに保存
                st.session_state.dsm_data = dsm_data
                st.session_state.step1_results = step1_results
                
                st.success(f"✅ STEP-1完了: {len(pareto_front)}個のパレート解を発見（{elapsed:.1f}秒）")
                
            except Exception as e:
                st.error(f"❌ エラー: {str(e)}")
                st.code(str(e), language="python")
                import traceback
                st.code(traceback.format_exc(), language="python")
                return
    
    # STEP-1結果の可視化
    if "step1_results" in st.session_state and st.session_state.step1_results:
        results = st.session_state.step1_results
        
        st.markdown("#### パレートフロント（2D）")
        
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
        
        col_p3, col_p4 = st.columns(2)
        with col_p3:
            step2_pop = st.slider("個体数", 50, 500, 200, 50, key="step2_pop")
        with col_p4:
            step2_gen = st.slider("世代数", 20, 200, 50, 10, key="step2_gen")
        
        if st.button("🚀 STEP-2を実行", type="primary", use_container_width=True):
            with st.spinner(f"NSGA-II最適化中（{step2_gen}世代）..."):
                try:
                    from utils.dsm_optimizer import PIMStep2NSGA2
                    import time
                    
                    start_time = time.time()
                    
                    dsm_data = st.session_state.dsm_data
                    selected = st.session_state.step1_results[st.session_state.step1_selected_idx]
                    removed_indices = [i for i, val in enumerate(selected['individual']) if val == 1]
                    
                    # STEP-2実行
                    step2 = PIMStep2NSGA2(dsm_data, removed_indices)
                    pareto_front = step2.run(n_pop=step2_pop, n_gen=step2_gen)
                    
                    elapsed = time.time() - start_time
                    
                    # パレートフロントのデータを抽出
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
                    
                    st.success(f"✅ STEP-2完了: {len(pareto_front)}個のパレート解を発見（{elapsed:.1f}秒）")
                    
                except Exception as e:
                    st.error(f"❌ エラー: {str(e)}")
                    st.code(str(e), language="python")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
                    return
        
        # STEP-2結果の可視化
        if "step2_results" in st.session_state and st.session_state.step2_results:
            results2 = st.session_state.step2_results
            
            st.markdown("#### パレートフロント（3D）")
            
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
    
    else:
        st.info("👆 まずSTEP-1を実行してください")


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
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📝 ステップ1: プロセス定義",
        "🎯 ステップ2: 機能カテゴリ",
        "🔧 ステップ3: ノード定義",
        "⚖️ ステップ4: ノード影響評価",
        "📈 ステップ5: 行列分析",
        "📊 ステップ6: ネットワーク可視化",
        "🔬 ステップ7: ネットワーク分析",
        "🎮 ステップ8: DSM最適化"
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


if __name__ == "__main__":
    main()
