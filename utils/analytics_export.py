"""
Advanced Analytics Export Utilities
高度な分析結果のエクスポート機能

全ての高度な分析結果を統合してエクスポート（Excel/JSON形式）
"""

from typing import Any, Dict, List
from io import BytesIO
from datetime import datetime
import json

import pandas as pd
import numpy as np
import streamlit as st


class AdvancedAnalyticsExporter:
    """
    高度な分析結果のエクスポート機能
    
    対応分析:
    - Shapley Value
    - Transfer Entropy
    - Bootstrap統計検定
    - Causal Inference
    - Graph Embedding
    - Fisher Information
    - (将来) Bayesian Inference
    """
    
    def __init__(self, analytics_results: Dict[str, Any]):
        """
        Args:
            analytics_results: st.session_state.advanced_analytics_results
        """
        self.results = analytics_results
    
    def export_to_excel(self) -> BytesIO:
        """
        Excelファイルにエクスポート
        
        シート構成:
        1. サマリー: 全分析の概要
        2. Shapley_Values: Shapley値ランキング
        3. Shapley_Cumulative: 累積貢献度
        4. Shapley_Categories: カテゴリ別貢献度
        5. TE_Matrix: Transfer Entropy行列
        6. TE_Flows: 有意な情報フロー
        7. TE_Comparison: 元の隣接行列との比較
        8. Bootstrap_CI: 信頼区間
        9. Bootstrap_Groups: グループ間比較
        10. CI_InterventionEffects: 介入効果
        11. CI_TopTargets: 最適介入ターゲット
        12. CI_Confounders: 交絡因子
        13. GE_Communities: コミュニティメンバー
        14. GE_Positions2D: 2D座標
        15. GE_Similarity: 類似度上位ペア
        16. FI_SensitivityScores: 感度スコア
        17. FI_CramerRaoBounds: CR下限
        
        Returns:
            BytesIO: Excelファイルのバイナリストリーム
        """
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # 1. サマリーシート
            summary_data = self._create_summary()
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='サマリー', index=False)
            else:
                # 空の場合はダミーシートを作成
                dummy_df = pd.DataFrame({"メッセージ": ["分析結果がありません"]})
                dummy_df.to_excel(writer, sheet_name='情報', index=False)
            
            # 2. Shapley Value
            if "shapley" in self.results:
                self._export_shapley_to_excel(writer)
            
            # 3. Transfer Entropy
            if "transfer_entropy" in self.results:
                self._export_te_to_excel(writer)
            
            # 4. Bootstrap
            if "bootstrap" in self.results:
                self._export_bootstrap_to_excel(writer)
            
            # 5. Causal Inference
            if "causal_inference" in self.results:
                self._export_causal_to_excel(writer)
            
            # 6. Graph Embedding
            if "graph_embedding" in self.results:
                self._export_graph_embedding_to_excel(writer)
            
            # 7. Fisher Information
            if "fisher_information" in self.results:
                self._export_fisher_to_excel(writer)
            
            # 8. Bayesian Inference
            if "bayesian_inference" in self.results:
                self._export_bayesian_to_excel(writer)
        
        buffer.seek(0)
        return buffer
    
    def _create_summary(self) -> List[Dict]:
        """サマリー情報を作成"""
        summary = []
        
        for analysis_name, data in self.results.items():
            result = data.get("result")
            params = data.get("parameters", {})
            timestamp = data.get("timestamp", "")
            
            summary.append({
                "分析名": self._translate_analysis_name(analysis_name),
                "実行日時": timestamp,
                "計算時間(秒)": getattr(result, "computation_time", 0),
                "パラメータ": str(params)
            })
        
        return summary
    
    def _export_shapley_to_excel(self, writer):
        """Shapley Value結果をExcelに"""
        shapley_data = self.results["shapley"]
        result = shapley_data["result"]
        
        # シート1: Shapley値ランキング
        shapley_df = pd.DataFrame([
            {
                "順位": i+1,
                "ノード名": name,
                "Shapley値": value,
                "貢献率(%)": (value / result.total_value * 100) if result.total_value > 0 else 0
            }
            for i, (name, value) in enumerate(result.top_contributors)
        ])
        shapley_df.to_excel(writer, sheet_name='Shapley_Values', index=False)
        
        # シート2: 累積貢献度
        if result.cumulative_contribution:
            cumulative_df = pd.DataFrame([
                {
                    "上位Nノード": n,
                    "累積貢献率(%)": pct
                }
                for n, pct in result.cumulative_contribution
            ])
            cumulative_df.to_excel(writer, sheet_name='Shapley_Cumulative', index=False)
        
        # シート3: カテゴリ別貢献度
        if result.category_contributions:
            category_df = pd.DataFrame([
                {"カテゴリ": cat, "平均Shapley値": value}
                for cat, value in result.category_contributions.items()
            ])
            category_df.to_excel(writer, sheet_name='Shapley_Categories', index=False)
        
        # シート4-5: 連携安定性分析（存在する場合）
        if "stability" in shapley_data:
            stability = shapley_data["stability"]
            
            # シート4: 上位貢献者
            top_nodes_data = []
            for i, node in enumerate(stability["top_contributors"], 1):
                top_nodes_data.append({
                    "順位": i,
                    "ノード名": node,
                    "Shapley値": result.shapley_values.get(node, 0)
                })
            
            if top_nodes_data:
                top_nodes_df = pd.DataFrame(top_nodes_data)
                top_nodes_df.to_excel(writer, sheet_name='Shapley_TopContributors', index=False)
            
            # シート5: 密結合ペア
            dense_pairs_data = []
            for i, (node1, node2, strength) in enumerate(stability["dense_connections"], 1):
                dense_pairs_data.append({
                    "順位": i,
                    "ノード1": node1,
                    "ノード2": node2,
                    "接続強度": strength
                })
            
            if dense_pairs_data:
                dense_pairs_df = pd.DataFrame(dense_pairs_data)
                dense_pairs_df.to_excel(writer, sheet_name='Shapley_DensePairs', index=False)
    
    def _export_te_to_excel(self, writer):
        """Transfer Entropy結果をExcelに"""
        te_data = self.results["transfer_entropy"]
        result = te_data["result"]
        
        # シート1: TE行列
        te_matrix = result.te_matrix
        if hasattr(te_matrix, 'shape'):
            # NumPy配列の場合
            n = te_matrix.shape[0]
            node_names = list(range(n))  # デフォルトはインデックス
            
            # ノード名を取得できる場合
            if hasattr(result, 'node_names'):
                node_names = result.node_names
            
            te_matrix_df = pd.DataFrame(
                te_matrix,
                columns=node_names,
                index=node_names
            )
            te_matrix_df.to_excel(writer, sheet_name='TE_Matrix', index=True)
        
        # シート2: 有意なフロー
        if result.significant_flows:
            flows_df = pd.DataFrame([
                {
                    "順位": i+1,
                    "From": source,
                    "To": target,
                    "TE(bits)": te_value
                }
                for i, (source, target, te_value) in enumerate(result.significant_flows)
            ])
            flows_df.to_excel(writer, sheet_name='TE_Flows', index=False)
        
        # シート3: 比較表
        if hasattr(result, 'comparison_with_original') and len(result.comparison_with_original) > 0:
            result.comparison_with_original.to_excel(writer, sheet_name='TE_Comparison', index=False)
    
    def _export_bootstrap_to_excel(self, writer):
        """Bootstrap結果をExcelに"""
        bs_data = self.results["bootstrap"]
        result = bs_data["result"]
        
        # シート1: 信頼区間
        ci_df = pd.DataFrame([
            {
                "ノード名": node,
                "値": ci[0],
                "下限95%": ci[1],
                "上限95%": ci[2],
                "相対誤差(%)": ((ci[2] - ci[1]) / (2 * abs(ci[0])) * 100) if abs(ci[0]) > 1e-6 else 0
            }
            for node, ci in result.node_ci.items()
        ])
        ci_df.to_excel(writer, sheet_name='Bootstrap_CI', index=False)
        
        # シート2: グループ間比較
        if hasattr(result, 'group_comparison') and len(result.group_comparison) > 0:
            result.group_comparison.to_excel(writer, sheet_name='Bootstrap_Groups', index=False)
    
    def _export_causal_to_excel(self, writer):
        """Causal Inference結果をExcelに"""
        ci_data = self.results["causal_inference"]
        result = ci_data["result"]
        intervention_node = ci_data["parameters"].get("intervention_node")
        
        # シート1: 介入効果
        if intervention_node and intervention_node in result.intervention_effects:
            effects = result.intervention_effects[intervention_node]
            effects_df = pd.DataFrame([
                {
                    "ノード": node,
                    "因果効果": effect,
                    "方向": "改善" if effect > 0 else "悪化"
                }
                for node, effect in sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)
            ])
            effects_df.to_excel(writer, sheet_name='CI_InterventionEffects', index=False)
        
        # シート2: 最適介入ターゲット
        if result.top_intervention_targets:
            targets_df = pd.DataFrame([
                {
                    "順位": i+1,
                    "ノード": node,
                    "総影響力": impact
                }
                for i, (node, impact) in enumerate(result.top_intervention_targets)
            ])
            targets_df.to_excel(writer, sheet_name='CI_TopTargets', index=False)
        
        # シート3: 交絡因子
        if result.confounders:
            confounders_df = pd.DataFrame([
                {
                    "From": source,
                    "To": target,
                    "交絡因子": ", ".join(conf_list)
                }
                for source, target, conf_list in result.confounders
            ])
            confounders_df.to_excel(writer, sheet_name='CI_Confounders', index=False)
    
    def _export_graph_embedding_to_excel(self, writer):
        """Graph Embedding結果をExcelに"""
        ge_data = self.results["graph_embedding"]
        result = ge_data["result"]
        
        # シート1: コミュニティメンバー
        community_members = {}
        for node, comm_id in result.communities.items():
            if comm_id not in community_members:
                community_members[comm_id] = []
            community_members[comm_id].append(node)
        
        comm_data = []
        for comm_id in sorted(community_members.keys()):
            members = community_members[comm_id]
            label = result.community_labels.get(comm_id, f"コミュニティ{comm_id+1}")
            comm_data.append({
                "コミュニティID": comm_id + 1,
                "名前": label,
                "ノード数": len(members),
                "メンバー": ", ".join(members)
            })
        
        comm_df = pd.DataFrame(comm_data)
        comm_df.to_excel(writer, sheet_name='GE_Communities', index=False)
        
        # シート2: 2D座標
        positions_data = []
        for node, (x, y) in result.node_positions_2d.items():
            comm_id = result.communities[node]
            positions_data.append({
                "ノード": node,
                "X座標": x,
                "Y座標": y,
                "コミュニティID": comm_id + 1
            })
        
        positions_df = pd.DataFrame(positions_data)
        positions_df.to_excel(writer, sheet_name='GE_Positions2D', index=False)
        
        # シート3: 類似度上位ペア
        similar_data = []
        for i, (node1, node2, sim) in enumerate(result.top_similar_pairs[:50], 1):
            similar_data.append({
                "順位": i,
                "ノード1": node1,
                "ノード2": node2,
                "類似度": sim
            })
        
        similar_df = pd.DataFrame(similar_data)
        similar_df.to_excel(writer, sheet_name='GE_Similarity', index=False)
    
    def _export_fisher_to_excel(self, writer):
        """Fisher Information結果をExcelに"""
        fi_data = self.results["fisher_information"]
        result = fi_data["result"]
        
        # シート1: 感度スコアランキング
        sensitivity_data = []
        for i, (source, target, score) in enumerate(result.top_sensitive_edges, 1):
            sensitivity_data.append({
                "順位": i,
                "From": source,
                "To": target,
                "感度スコア": score
            })
        
        if sensitivity_data:
            sensitivity_df = pd.DataFrame(sensitivity_data)
            sensitivity_df.to_excel(writer, sheet_name='FI_SensitivityScores', index=False)
        
        # シート2: Cramér-Rao下限
        if result.cramer_rao_bounds:
            cr_sorted = sorted(
                result.cramer_rao_bounds.items(),
                key=lambda x: x[1],
                reverse=True
            )[:50]  # 上位50組
            
            cr_data = []
            for (source, target), bound in cr_sorted:
                cr_data.append({
                    "From": source,
                    "To": target,
                    "CR下限": bound
                })
            
            cr_df = pd.DataFrame(cr_data)
            cr_df.to_excel(writer, sheet_name='FI_CramerRaoBounds', index=False)
    
    def _export_bayesian_to_excel(self, writer):
        bi_data = self.results["bayesian_inference"]
        result = bi_data["result"]
        
        ci_data = []
        for source, target, uncertainty_score in result.high_uncertainty_edges:
            edge = (source, target)
            if edge in result.credible_intervals:
                mean_val, lower, upper = result.credible_intervals[edge]
                
                ci_data.append({
                    "From": source,
                    "To": target,
                    "事後平均": mean_val,
                    "下限": lower,
                    "上限": upper,
                    "不確実性スコア": uncertainty_score
                })
        
        if ci_data:
            ci_df = pd.DataFrame(ci_data)
            ci_df.to_excel(writer, sheet_name='BI_CredibleIntervals', index=False)
        
        high_uncertainty_data = []
        for i, (source, target, score) in enumerate(result.high_uncertainty_edges[:50], 1):
            high_uncertainty_data.append({
                "順位": i,
                "From": source,
                "To": target,
                "不確実性スコア": score
            })
        
        if high_uncertainty_data:
            hu_df = pd.DataFrame(high_uncertainty_data)
            hu_df.to_excel(writer, sheet_name='BI_HighUncertainty', index=False)
    
    def export_to_json(self) -> str:
        """
        JSON形式でエクスポート
        
        Returns:
            JSON文字列
        """
        export_data = {
            "export_version": "1.0.0",
            "export_timestamp": datetime.now().isoformat(),
            "analyses": {}
        }
        
        for analysis_name, data in self.results.items():
            result = data.get("result")
            
            # 各分析ごとに主要データを抽出
            if analysis_name == "shapley":
                export_data["analyses"]["shapley_value"] = {
                    "shapley_values": result.shapley_values,
                    "total_value": result.total_value,
                    "computation_time": result.computation_time,
                    "parameters": data.get("parameters", {})
                }
                
                # 連携安定性データ（存在する場合）
                if "stability" in data:
                    stability = data["stability"]
                    export_data["analyses"]["shapley_value"]["stability"] = {
                        "top_contributors": stability["top_contributors"],
                        "dense_connections": [(n1, n2, float(s)) for n1, n2, s in stability["dense_connections"]],
                        "recommendation": stability["recommendation"]
                    }
            
            elif analysis_name == "transfer_entropy":
                export_data["analyses"]["transfer_entropy"] = {
                    "te_matrix": result.te_matrix.tolist() if hasattr(result.te_matrix, 'tolist') else result.te_matrix,
                    "significant_flows": result.significant_flows,
                    "bottleneck_nodes": result.bottleneck_nodes,
                    "computation_time": result.computation_time,
                    "parameters": data.get("parameters", {})
                }
            
            elif analysis_name == "bootstrap":
                export_data["analyses"]["bootstrap"] = {
                    "node_ci": {k: list(v) for k, v in result.node_ci.items()},
                    "stable_findings": result.stable_findings,
                    "unstable_findings": result.unstable_findings,
                    "computation_time": result.computation_time,
                    "parameters": data.get("parameters", {})
                }
            
            elif analysis_name == "causal_inference":
                export_data["analyses"]["causal_inference"] = {
                    "direct_effects": {f"{k[0]}->{k[1]}": v for k, v in result.direct_effects.items()},
                    "indirect_effects": {f"{k[0]}->{k[1]}": v for k, v in result.indirect_effects.items()},
                    "total_effects": {f"{k[0]}->{k[1]}": v for k, v in result.total_effects.items()},
                    "top_intervention_targets": [(node, impact) for node, impact in result.top_intervention_targets],
                    "n_confounders": len(result.confounders),
                    "computation_time": result.computation_time,
                    "parameters": data.get("parameters", {})
                }
            
            elif analysis_name == "graph_embedding":
                export_data["analyses"]["graph_embedding"] = {
                    "communities": result.communities,
                    "modularity": result.modularity,
                    "n_communities": result.n_communities,
                    "top_similar_pairs": [(n1, n2, sim) for n1, n2, sim in result.top_similar_pairs[:20]],
                    "node_positions_2d": {node: list(pos) for node, pos in result.node_positions_2d.items()},
                    "computation_time": result.computation_time,
                    "parameters": data.get("parameters", {})
                }
            
            elif analysis_name == "fisher_information":
                export_data["analyses"]["fisher_information"] = {
                    "n_edges": result.n_edges,
                    "condition_number": result.condition_number,
                    "effective_rank": int(result.effective_rank),
                    "top_sensitive_edges": [(s, t, score) for s, t, score in result.top_sensitive_edges[:20]],
                    "eigenvalues": result.eigenvalues.tolist()[:10],  # 上位10固有値
                    "computation_time": result.computation_time,
                    "parameters": data.get("parameters", {})
                }
            
            elif analysis_name == "bayesian_inference":
                export_data["analyses"]["bayesian_inference"] = {
                    "n_edges": result.n_edges,
                    "credible_level": result.credible_level,
                    "n_bootstrap": result.n_bootstrap,
                    "high_uncertainty_edges": [(s, t, score) for s, t, score in result.high_uncertainty_edges[:20]],
                    "avg_uncertainty": float(np.mean(list(result.uncertainty_scores.values()))) if result.uncertainty_scores else 0.0,
                    "computation_time": result.computation_time,
                    "parameters": data.get("parameters", {})
                }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def _translate_analysis_name(self, name: str) -> str:
        """分析名の日本語化"""
        translations = {
            "shapley": "Shapley Value（協力貢献度分析）",
            "transfer_entropy": "Transfer Entropy（情報フロー分析）",
            "bootstrap": "Bootstrap統計検定",
            "causal_inference": "Causal Inference（因果推論）",
            "graph_embedding": "Graph Embedding（潜在構造発見）",
            "fisher_information": "Fisher Information（感度分析）",
            "bayesian_inference": "Bayesian Inference（不確実性定量化）"
        }
        return translations.get(name, name)


def add_analytics_export_to_sidebar():
    """
    サイドバーにエクスポートボタンを追加
    
    app_tabs.py の render_sidebar() から呼び出す
    """
    if "advanced_analytics_results" not in st.session_state or len(st.session_state.advanced_analytics_results) == 0:
        return
    
    st.sidebar.markdown("---")
    st.sidebar.header("📤 高度な分析結果エクスポート")
    
    n_analyses = len(st.session_state.advanced_analytics_results)
    st.sidebar.info(f"実行済み分析: {n_analyses}件")
    
    # Excelエクスポート
    if st.sidebar.button("📊 Excelでダウンロード", key="export_excel_adv", use_container_width=True):
        exporter = AdvancedAnalyticsExporter(st.session_state.advanced_analytics_results)
        excel_buffer = exporter.export_to_excel()
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_analytics_{timestamp}.xlsx"
        
        st.sidebar.download_button(
            label="⬇️ ダウンロード",
            data=excel_buffer,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="download_excel_adv"
        )
    
    # JSONエクスポート
    if st.sidebar.button("📄 JSONでダウンロード", key="export_json_adv", use_container_width=True):
        exporter = AdvancedAnalyticsExporter(st.session_state.advanced_analytics_results)
        json_data = exporter.export_to_json()
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_analytics_{timestamp}.json"
        
        st.sidebar.download_button(
            label="⬇️ ダウンロード",
            data=json_data,
            file_name=filename,
            mime="application/json",
            use_container_width=True,
            key="download_json_adv"
        )
