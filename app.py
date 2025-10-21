"""
Process Insight Modeler (PIM)
生産プロセスの暗黙知を形式知に変換するアプリケーション
"""

import json
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


def main() -> None:
    """メインアプリケーション"""

    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    SessionManager.initialize()

    st.title(settings.APP_TITLE)

    with st.sidebar:
        st.header("1. プロセス定義")

        process_name = st.text_input(
            "生産プロセス名",
            value=SessionManager.get_process_name(),
            placeholder="例: 自動車エンジン組立工程",
            help="分析対象の生産プロセスの名称を入力してください",
        )

        process_description = st.text_area(
            "プロセスの概要",
            value=SessionManager.get_process_description(),
            height=200,
            placeholder="プロセスの詳細な説明を記述してください...",
            help="プロセスの詳細、主要な工程、使用する設備などを記述してください",
        )

        SessionManager.update_process_info(process_name, process_description)

        st.divider()
        st.header("2. 機能カテゴリの定義")

        with st.expander("🎯 プロセス機能の抽出設定", expanded=True):
            st.caption("「機能カテゴリ」= プロセスを構成する動的な変換機能（インプット→変換→アウトプット）")
            
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
                "カテゴリを自動抽出",
                type="primary",
                help="選択したオプションでAIがカテゴリを自動抽出します",
                use_container_width=True
            ):
                if not SessionManager.get_process_description():
                    st.error("プロセスの概要を入力してください")
                else:
                    try:
                        llm_client = LLMClient()

                        if use_verbalized_sampling:
                            st.info("🎲 5つの異なる分析哲学から生成しています...")

                            with st.spinner("Verbalized Samplingで多様な視点を生成中..."):
                                perspectives = llm_client.generate_diverse_category_sets(
                                    process_name=SessionManager.get_process_name(),
                                    process_description=SessionManager.get_process_description(),
                                    num_perspectives=5,
                                )
                            
                            if perspectives:
                                proposals = []
                                for i, persp in enumerate(perspectives, 1):
                                    from core.data_models import FunctionalCategory
                                    categories = [FunctionalCategory(**cat_data) for cat_data in persp['categories']]
                                    proposals.append({
                                        "name": f"{persp['perspective']} (確率: {persp['probability']:.2f})",
                                        "description": persp['description'],
                                        "probability": persp['probability'],
                                        "categories": categories,
                                    })
                                
                                st.session_state.category_proposals = proposals
                                st.success(f"🎲 {len(perspectives)}つの異なる視点を生成しました！下で比較してください。")
                            else:
                                st.error("カテゴリ生成に失敗しました。再試行してください。")

                        else:
                            with st.spinner("AIがカテゴリを抽出中です..."):
                                options = CategoryGenerationOptions(
                                    focus=analysis_focus,
                                    granularity=granularity
                                )
                                categories = llm_client.extract_categories_advanced(
                                    SessionManager.get_process_name(),
                                    SessionManager.get_process_description(),
                                    options
                                )
                                SessionManager.set_functional_categories(
                                    [cat.name for cat in categories]
                                )
                                if "categories_metadata" not in st.session_state:
                                    st.session_state.categories_metadata = {}
                                st.session_state.categories_metadata = {
                                    cat.name: cat.model_dump() for cat in categories
                                }
                                st.success(f"{len(categories)}個のカテゴリを抽出しました！")
                                st.rerun()

                    except ValueError as e:
                        st.error(f"入力エラー: {str(e)}")
                    except json.JSONDecodeError as e:
                        st.error(f"データ解析エラー: LLMからの応答を解析できませんでした")
                        with st.expander("詳細を表示"):
                            st.text(str(e))
                    except OpenAIError as e:
                        st.error(f"OpenAI APIエラー: {str(e)}")
                        st.info("APIキーが正しく設定されているか確認してください")
                    except Exception as e:
                        st.error(f"予期しないエラー: {str(e)}")
                        with st.expander("詳細を表示"):
                            st.exception(e)

        with col_btn2:
            if "category_proposals" in st.session_state and st.session_state.category_proposals:
                if st.button("案をクリア", use_container_width=True):
                    st.session_state.category_proposals = []
                    st.rerun()

        if "category_proposals" in st.session_state and st.session_state.category_proposals:
            st.divider()
            st.subheader("生成された案の比較")

            tabs = st.tabs([p["name"] for p in st.session_state.category_proposals])

            for idx, (tab, proposal) in enumerate(zip(tabs, st.session_state.category_proposals)):
                with tab:
                    if "description" in proposal:
                        st.info(proposal['description'])
                    
                    st.caption(f"カテゴリ数: {len(proposal['categories'])}個")

                    for cat in proposal['categories']:
                        with st.container():
                            col_name, col_imp = st.columns([3, 1])
                            with col_name:
                                st.markdown(f"**{cat.name}**")
                            with col_imp:
                                st.markdown(f"重要度: {'⭐' * cat.importance}")

                            st.caption(cat.description)
                            
                            if cat.inputs or cat.outputs:
                                col_in, col_out = st.columns(2)
                                with col_in:
                                    if cat.inputs:
                                        st.caption(f"📥 インプット: {', '.join(cat.inputs[:2])}")
                                with col_out:
                                    if cat.outputs:
                                        st.caption(f"📤 アウトプット: {', '.join(cat.outputs[:2])}")

                            if cat.examples:
                                st.caption(f"🔧 例: {', '.join(cat.examples[:3])}")

                            st.divider()

                    if st.button(
                        f"この案を採用",
                        key=f"adopt_{idx}",
                        type="primary",
                        use_container_width=True
                    ):
                        SessionManager.set_functional_categories(
                            [cat.name for cat in proposal['categories']]
                        )
                        if "categories_metadata" not in st.session_state:
                            st.session_state.categories_metadata = {}
                        st.session_state.categories_metadata = {
                            cat.name: cat.model_dump() for cat in proposal['categories']
                        }
                        st.session_state.category_proposals = []
                        st.success("カテゴリを設定しました！")
                        st.rerun()

    st.header("現在のプロジェクト情報")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("プロセス名")
        if SessionManager.get_process_name():
            st.info(SessionManager.get_process_name())
        else:
            st.warning("プロセス名が未入力です")

    with col2:
        st.subheader("プロセス概要")
        if SessionManager.get_process_description():
            st.text_area(
                "説明",
                value=SessionManager.get_process_description(),
                height=150,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            st.warning("プロセス概要が未入力です")

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
                    help="機能カテゴリの名称（品質、コスト、時間など）",
                    max_chars=50,
                    required=True,
                )
            },
        )

        updated_categories = edited_df["カテゴリ名"].dropna().tolist()
        updated_categories = [cat.strip() for cat in updated_categories if cat.strip()]

        if updated_categories != categories:
            SessionManager.set_functional_categories(updated_categories)

            if "categories_metadata" not in st.session_state:
                st.session_state.categories_metadata = {}

            old_set = set(categories)
            new_set = set(updated_categories)

            added = new_set - old_set
            removed = old_set - new_set

            if len(added) == 1 and len(removed) == 1 and len(updated_categories) == len(categories):
                old_name = list(removed)[0]
                new_name = list(added)[0]
                try:
                    old_idx = categories.index(old_name)
                    new_idx = updated_categories.index(new_name)
                    if old_idx == new_idx:
                        if old_name in st.session_state.categories_metadata:
                            metadata = st.session_state.categories_metadata.pop(old_name)
                            metadata["name"] = new_name
                            st.session_state.categories_metadata[new_name] = metadata
                        added.remove(new_name)
                        removed.remove(old_name)
                except ValueError:
                    pass

            for cat_name in added:
                if cat_name not in st.session_state.categories_metadata:
                    st.session_state.categories_metadata[cat_name] = {
                        "name": cat_name,
                        "description": "",
                        "transformation_type": "processing",
                        "inputs": [],
                        "outputs": [],
                        "process_phase": "main_process",
                        "importance": 3,
                        "examples": []
                    }

            for cat_name in removed:
                if cat_name in st.session_state.categories_metadata:
                    del st.session_state.categories_metadata[cat_name]

            if "categories_changed" not in st.session_state:
                st.session_state.categories_changed = True

        if updated_categories:
            st.caption(f"現在のカテゴリ数: {len(updated_categories)}")

        if st.session_state.get("categories_changed", False):
            st.warning(
                "⚠️ カテゴリが変更されました。Zigzagging対話を再開する場合は、"
                "下の「対話をリセット」ボタンをクリックしてください。"
            )

    project_data = SessionManager.get_project_data()

    with st.expander("📊 セッションステート情報（開発用）", expanded=False):
        st.json(project_data)

    st.divider()

    if categories:
        st.header("3. ノードの定義")
        
        generation_mode = st.radio(
            "生成モード",
            ["AI主導対話", "多様性生成（Verbalized Sampling）"],
            horizontal=True,
            help="AI主導対話：全カテゴリをソクラテス式対話で生成 / 多様性生成：複数の異なる視点から一度に生成"
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
                st.caption("🎲 Verbalized Sampling - 全カテゴリ一括生成")
                
                if st.button("🎲 多様な視点で生成", type="primary", use_container_width=True, help="5つの異なる思考モードから全カテゴリを生成"):
                    try:
                        llm_client = LLMClient()
                        
                        with st.spinner("🎲 5つの異なる思考モードから全カテゴリを生成中..."):
                            perspectives = llm_client.generate_diverse_idef0_nodes_all_categories(
                                process_name=SessionManager.get_process_name(),
                                process_description=SessionManager.get_process_description(),
                                categories=categories,
                                num_perspectives=5,
                            )
                        
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
                                st.success(f"『{persp['perspective']}』を採用しました！")
                                st.rerun()

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

    st.divider()
    
    # === ネットワーク可視化セクション ===
    st.header("ネットワーク可視化")
    
    viz_tab1, viz_tab2 = st.tabs(["🎮 3D可視化", "📊 2D可視化"])
    
    with viz_tab1:
        if nodes and len(nodes) >= 2:
            st.info("💡 3D空間でノード間の関係性を可視化します（要: 隣接行列データ）")
        
            # デモ用の隣接行列を生成（実際のデータがあれば置き換え）
            import numpy as np
            
            if "adjacency_matrix" not in st.session_state:
                # デモ用のランダム隣接行列を生成
                n = len(nodes)
                demo_matrix = np.random.randint(-5, 6, size=(n, n))
                np.fill_diagonal(demo_matrix, 0)
                st.session_state.adjacency_matrix = demo_matrix
                st.warning("⚠️ デモ用のランダムデータを表示しています。実際の評価データがあれば自動的に反映されます。")
            
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
        
        else:
            st.warning("ノードが2つ以上必要です。先にステップ3でノードを定義してください。")
    
    with viz_tab2:
        if nodes and len(nodes) >= 2:
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
        else:
            st.warning("ノードが2つ以上必要です。先にステップ3でノードを定義してください。")

    st.divider()

    st.info(
        """
        **使い方:**
        1. サイドバーで生産プロセス名と概要を入力してください
        2. 「カテゴリを自動抽出」ボタンで機能カテゴリを生成します
        3. AIとの対話を通じて、プロセスのノードを定義します
        4. 「対話からノードを抽出して保存」でノードを確定します
        5. 次のステップでノード間の評価に進みます
        """
    )


if __name__ == "__main__":
    main()
