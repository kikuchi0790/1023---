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

        with st.expander("🎯 カテゴリ抽出の詳細設定", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                analysis_focus = st.selectbox(
                    "分析の観点",
                    [
                        "balanced",
                        "quality",
                        "cost",
                        "time",
                        "safety",
                        "flexibility"
                    ],
                    format_func=lambda x: {
                        "balanced": "バランス型（推奨）",
                        "quality": "品質重視",
                        "cost": "コスト重視",
                        "time": "時間重視",
                        "safety": "安全性重視",
                        "flexibility": "柔軟性重視"
                    }[x],
                    help="どの観点を重視してカテゴリを抽出するか選択します"
                )

            with col2:
                granularity = st.selectbox(
                    "カテゴリの粒度",
                    ["standard", "coarse", "detailed"],
                    format_func=lambda x: {
                        "coarse": "粗い（4-5個）",
                        "standard": "標準（6-8個）",
                        "detailed": "詳細（10-12個）"
                    }[x],
                    help="生成するカテゴリの数と詳細レベル"
                )

            with col3:
                multi_generation = st.checkbox(
                    "複数案を生成",
                    value=False,
                    help="3つの異なる観点から案を生成し、比較できます"
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

                        if multi_generation:
                            st.info("3つの異なる観点から案を生成しています...")

                            if "category_proposals" not in st.session_state:
                                st.session_state.category_proposals = []

                            proposals = []
                            focuses = [analysis_focus, "quality", "cost"]

                            for i, focus in enumerate(focuses, 1):
                                with st.spinner(f"案{i}を生成中..."):
                                    options = CategoryGenerationOptions(
                                        focus=focus,
                                        granularity=granularity
                                    )
                                    categories = llm_client.extract_categories_advanced(
                                        SessionManager.get_process_name(),
                                        SessionManager.get_process_description(),
                                        options
                                    )
                                    proposals.append({
                                        "name": f"案{i}: {options.get_focus_description().split('：')[0]}",
                                        "categories": categories,
                                        "options": options
                                    })

                            st.session_state.category_proposals = proposals
                            st.success("3つの案を生成しました！下で比較してください。")

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
                    st.info(f"カテゴリ数: {len(proposal['categories'])}個")

                    for cat in proposal['categories']:
                        with st.container():
                            col_name, col_imp = st.columns([3, 1])
                            with col_name:
                                st.markdown(f"**{cat.name}**")
                            with col_imp:
                                st.markdown(f"重要度: {'⭐' * cat.importance}")

                            st.caption(cat.description)

                            if cat.examples:
                                st.caption(f"例: {', '.join(cat.examples[:3])}")

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

            with st.expander("📋 カテゴリの詳細情報", expanded=True):
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

                            if meta.get("examples"):
                                examples_str = "、".join(meta["examples"][:3])
                                st.caption(f"例: {examples_str}")

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
        st.header("3. ノードの定義 (Zigzagging)")

        messages = SessionManager.get_messages()

        if not messages:
            llm_client = LLMClient()
            initial_message = llm_client.generate_initial_message(
                SessionManager.get_process_name(), categories
            )
            SessionManager.add_message("assistant", initial_message)
            st.rerun()

        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input := st.chat_input("メッセージを入力してください..."):
            SessionManager.add_message("user", user_input)

            with st.chat_message("user"):
                st.markdown(user_input)

            try:
                with st.spinner("AIが応答を生成中..."):
                    llm_client = LLMClient()
                    assistant_response = llm_client.chat_zigzagging(
                        process_name=SessionManager.get_process_name(),
                        categories=categories,
                        chat_history=messages,
                        user_message=user_input,
                    )

                SessionManager.add_message("assistant", assistant_response)

                with st.chat_message("assistant"):
                    st.markdown(assistant_response)

                st.rerun()

            except OpenAIError as e:
                st.error(f"OpenAI APIエラー: {str(e)}")
            except Exception as e:
                st.error(f"エラー: {str(e)}")

        st.divider()

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button(
                "対話からノードを抽出して保存",
                type="primary",
                help="これまでの対話からノードを自動抽出します",
            ):
                if len(messages) < 2:
                    st.warning("ノードを抽出するには、もう少し対話を進めてください")
                else:
                    try:
                        with st.spinner("ノードを抽出中..."):
                            llm_client = LLMClient()
                            nodes = llm_client.extract_nodes_from_chat(messages)
                            SessionManager.set_nodes(nodes)
                            st.success(f"{len(nodes)}個のノードを抽出しました！")
                            st.rerun()

                    except json.JSONDecodeError:
                        st.error("ノードの抽出に失敗しました。対話を続けてください。")
                    except Exception as e:
                        st.error(f"エラー: {str(e)}")

        with col2:
            if st.button("対話をリセット", help="チャット履歴をクリアします"):
                SessionManager.clear_messages()
                if "categories_changed" in st.session_state:
                    st.session_state.categories_changed = False
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
