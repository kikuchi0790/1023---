"""
PIMアプリケーション全体の動作確認テスト
タブ1→8までのエンドツーエンドワークフロー検証
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from core.session_manager import SessionManager
from core.llm_client import LLMClient
from tests.test_sample_data import (
    SAMPLE_PROCESS_NAME,
    SAMPLE_PROCESS_DESCRIPTION,
    EXPECTED_MIN_NODES,
    EXPECTED_MAX_NODES
)


class WorkflowTester:
    def __init__(self):
        self.results = []
        self.errors = []
        
    def log(self, message: str, status: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{status}] {message}"
        print(log_entry)
        self.results.append(log_entry)
        
    def error(self, message: str):
        self.log(message, "ERROR")
        self.errors.append(message)
        
    def success(self, message: str):
        self.log(message, "SUCCESS")
        
    def test_tab1_process_definition(self):
        self.log("=== タブ1: プロセス定義 ===")
        
        try:
            SessionManager.update_process_info(
                SAMPLE_PROCESS_NAME,
                SAMPLE_PROCESS_DESCRIPTION
            )
            
            process_name = SessionManager.get_process_name()
            process_desc = SessionManager.get_process_description()
            
            assert process_name == SAMPLE_PROCESS_NAME, "プロセス名が一致しません"
            assert process_desc == SAMPLE_PROCESS_DESCRIPTION, "プロセス概要が一致しません"
            
            self.success("✅ タブ1: プロセス定義成功")
            return True
            
        except Exception as e:
            self.error(f"❌ タブ1エラー: {str(e)}")
            return False
    
    def test_tab2_functional_categories(self):
        self.log("=== タブ2: 機能カテゴリ抽出 ===")
        
        try:
            llm_client = LLMClient()
            
            self.log("LLMでカテゴリを生成中...")
            result = llm_client.generate_functional_categories(
                process_name=SAMPLE_PROCESS_NAME,
                process_description=SAMPLE_PROCESS_DESCRIPTION,
                num_categories=4
            )
            
            categories = [cat.name for cat in result.categories]
            SessionManager.set_functional_categories(categories)
            
            saved_categories = SessionManager.get_functional_categories()
            
            assert len(saved_categories) >= 3, f"カテゴリ数が少なすぎます: {len(saved_categories)}"
            assert len(saved_categories) <= 10, f"カテゴリ数が多すぎます: {len(saved_categories)}"
            
            self.log(f"生成されたカテゴリ: {saved_categories}")
            self.success("✅ タブ2: カテゴリ抽出成功")
            return True
            
        except Exception as e:
            self.error(f"❌ タブ2エラー: {str(e)}")
            import traceback
            self.error(traceback.format_exc())
            return False
    
    def test_tab3_node_definition(self):
        self.log("=== タブ3: ノード定義 ===")
        
        try:
            categories = SessionManager.get_functional_categories()
            llm_client = LLMClient()
            
            for category in categories:
                self.log(f"カテゴリ「{category}」のIDEF0ノードを生成中...")
                
                idef0_data = llm_client.generate_idef0_nodes_single_category(
                    process_name=SAMPLE_PROCESS_NAME,
                    process_description=SAMPLE_PROCESS_DESCRIPTION,
                    category_name=category,
                    all_categories=categories
                )
                
                SessionManager.set_idef0_node(category, idef0_data)
                self.log(f"  - Inputs: {len(idef0_data.get('inputs', []))}")
                self.log(f"  - Mechanisms: {len(idef0_data.get('mechanisms', []))}")
                self.log(f"  - Outputs: {len(idef0_data.get('outputs', []))}")
            
            nodes = SessionManager.get_nodes()
            
            assert len(nodes) >= EXPECTED_MIN_NODES, f"ノード数が少なすぎます: {len(nodes)}"
            assert len(nodes) <= EXPECTED_MAX_NODES, f"ノード数が多すぎます: {len(nodes)}"
            
            self.log(f"総ノード数: {len(nodes)}")
            self.success("✅ タブ3: ノード定義成功")
            return True
            
        except Exception as e:
            self.error(f"❌ タブ3エラー: {str(e)}")
            import traceback
            self.error(traceback.format_exc())
            return False
    
    def test_tab4_node_evaluation_simple(self):
        self.log("=== タブ4: ノード影響評価（簡易版） ===")
        
        try:
            from utils.idef0_classifier import generate_zigzagging_pairs
            import numpy as np
            
            nodes = SessionManager.get_nodes()
            all_idef0 = SessionManager.get_all_idef0_nodes()
            categories = SessionManager.get_functional_categories()
            
            self.log("評価ペアを生成中...")
            pairs = generate_zigzagging_pairs(nodes, all_idef0, categories)
            
            self.log(f"生成されたペア数: {len(pairs)}")
            
            n = len(nodes)
            matrix = np.zeros((n, n))
            
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            self.log("簡易評価（ランダム値）を設定中...")
            import random
            for pair in pairs[:50]:
                from_idx = node_to_idx[pair["from_node"]]
                to_idx = node_to_idx[pair["to_node"]]
                
                matrix[from_idx][to_idx] = random.choice([0, 0, 0, 0, 3, -2, 4, -3])
            
            st.session_state.adjacency_matrix = matrix
            
            import pandas as pd
            st.session_state.adj_matrix_df = pd.DataFrame(
                matrix,
                index=nodes,
                columns=nodes
            )
            
            non_zero_count = np.count_nonzero(matrix)
            sparsity = 1.0 - (non_zero_count / (n * n))
            
            self.log(f"非ゼロ要素数: {non_zero_count}")
            self.log(f"疎行列率: {sparsity:.1%}")
            
            self.success("✅ タブ4: 評価完了（簡易版）")
            return True
            
        except Exception as e:
            self.error(f"❌ タブ4エラー: {str(e)}")
            import traceback
            self.error(traceback.format_exc())
            return False
    
    def test_tab5_matrix_analysis(self):
        self.log("=== タブ5: 行列分析 ===")
        
        try:
            import numpy as np
            
            if st.session_state.get("adjacency_matrix") is None:
                raise ValueError("隣接行列が存在しません")
            
            matrix = st.session_state.adjacency_matrix
            nodes = SessionManager.get_nodes()
            
            n = matrix.shape[0]
            assert n == len(nodes), f"行列サイズ不一致: {n} != {len(nodes)}"
            
            non_zero_count = np.count_nonzero(matrix)
            sparsity = 1.0 - (non_zero_count / (n * n))
            
            self.log(f"行列サイズ: {n}x{n}")
            self.log(f"非ゼロ要素: {non_zero_count}")
            self.log(f"疎行列率: {sparsity:.1%}")
            
            in_degrees = np.count_nonzero(matrix, axis=0)
            out_degrees = np.count_nonzero(matrix, axis=1)
            
            high_in_degree_nodes = [(nodes[i], int(in_degrees[i])) 
                                    for i in range(n) if in_degrees[i] > 3]
            high_out_degree_nodes = [(nodes[i], int(out_degrees[i])) 
                                     for i in range(n) if out_degrees[i] > 3]
            
            if high_in_degree_nodes:
                self.log(f"高入次数ノード: {high_in_degree_nodes[:3]}")
            if high_out_degree_nodes:
                self.log(f"高出次数ノード: {high_out_degree_nodes[:3]}")
            
            self.success("✅ タブ5: 行列分析成功")
            return True
            
        except Exception as e:
            self.error(f"❌ タブ5エラー: {str(e)}")
            import traceback
            self.error(traceback.format_exc())
            return False
    
    def test_tab6_network_visualization(self):
        self.log("=== タブ6: ネットワーク可視化 ===")
        
        try:
            from utils.networkmaps_bridge import convert_pim_to_networkmaps
            from utils.cytoscape_bridge import convert_pim_to_cytoscape
            import numpy as np
            
            nodes = SessionManager.get_nodes()
            matrix = st.session_state.adjacency_matrix
            categories = SessionManager.get_functional_categories()
            idef0_data = SessionManager.get_all_idef0_nodes()
            
            self.log("NetworkMaps形式に変換中...")
            networkmaps_data = convert_pim_to_networkmaps(
                nodes=nodes,
                adjacency_matrix=matrix,
                categories=categories,
                idef0_data=idef0_data
            )
            
            assert "L2" in networkmaps_data, "L2データがありません"
            assert "devices" in networkmaps_data["L2"], "デバイスデータがありません"
            
            device_count = len(networkmaps_data["L2"]["devices"])
            self.log(f"NetworkMapsデバイス数: {device_count}")
            
            self.log("Cytoscape形式に変換中...")
            cytoscape_data = convert_pim_to_cytoscape(
                nodes=nodes,
                adjacency_matrix=matrix,
                categories=categories,
                idef0_data=idef0_data,
                threshold=2.0
            )
            
            assert "elements" in cytoscape_data, "elementsがありません"
            node_count = len([e for e in cytoscape_data["elements"] if "data" in e and "source" not in e["data"]])
            edge_count = len([e for e in cytoscape_data["elements"] if "data" in e and "source" in e["data"]])
            
            self.log(f"Cytoscapeノード数: {node_count}")
            self.log(f"Cytoscapeエッジ数: {edge_count}")
            
            self.success("✅ タブ6: 可視化データ変換成功")
            return True
            
        except Exception as e:
            self.error(f"❌ タブ6エラー: {str(e)}")
            import traceback
            self.error(traceback.format_exc())
            return False
    
    def test_tab7_network_analysis(self):
        self.log("=== タブ7: ネットワーク分析 ===")
        
        try:
            import networkx as nx
            import numpy as np
            
            nodes = SessionManager.get_nodes()
            matrix = st.session_state.adjacency_matrix
            
            self.log("NetworkXグラフを作成中...")
            G = nx.DiGraph()
            
            for i, node in enumerate(nodes):
                G.add_node(node)
            
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if matrix[i][j] != 0:
                        G.add_edge(nodes[i], nodes[j], weight=abs(matrix[i][j]))
            
            self.log(f"ノード数: {G.number_of_nodes()}")
            self.log(f"エッジ数: {G.number_of_edges()}")
            
            self.log("PageRankを計算中...")
            pagerank = nx.pagerank(G)
            top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]
            self.log(f"Top 3 PageRank: {[(n, f'{v:.4f}') for n, v in top_pagerank]}")
            
            self.log("中心性指標を計算中...")
            degree_centrality = nx.degree_centrality(G)
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            self.log(f"Top 3 Degree Centrality: {[(n, f'{v:.4f}') for n, v in top_degree]}")
            
            if nx.is_weakly_connected(G):
                betweenness = nx.betweenness_centrality(G)
                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
                self.log(f"Top 3 Betweenness: {[(n, f'{v:.4f}') for n, v in top_betweenness]}")
            else:
                self.log("グラフが非連結のため、Betweenness計算をスキップ")
            
            self.success("✅ タブ7: ネットワーク分析成功")
            return True
            
        except Exception as e:
            self.error(f"❌ タブ7エラー: {str(e)}")
            import traceback
            self.error(traceback.format_exc())
            return False
    
    def test_tab8_dsm_optimization_basic(self):
        self.log("=== タブ8: DSM最適化（基本チェック） ===")
        
        try:
            from utils.dsm_optimizer import PIMDSMData
            import pandas as pd
            
            nodes = SessionManager.get_nodes()
            adj_matrix_df = st.session_state.adj_matrix_df
            idef0_nodes = SessionManager.get_all_idef0_nodes()
            
            self.log("DSMデータを構築中...")
            dsm_data = PIMDSMData(
                adj_matrix_df=adj_matrix_df,
                nodes=nodes,
                idef0_nodes=idef0_nodes,
                param_mode="fixed_default"
            )
            
            self.log(f"FR数（Output）: {dsm_data.fn_num}")
            self.log(f"DP数（Mechanism+Input）: {dsm_data.dp_num}")
            self.log(f"リオーダー後ノード数: {len(dsm_data.reordered_nodes)}")
            
            assert dsm_data.fn_num > 0, "FR数が0です"
            assert dsm_data.dp_num > 0, "DP数が0です"
            assert dsm_data.om_size == len(nodes), "マトリクスサイズ不一致"
            
            self.success("✅ タブ8: DSMデータ構築成功")
            return True
            
        except Exception as e:
            self.error(f"❌ タブ8エラー: {str(e)}")
            import traceback
            self.error(traceback.format_exc())
            return False
    
    def test_data_export_import(self):
        self.log("=== データエクスポート/インポート ===")
        
        try:
            from utils.data_io import export_to_json, import_from_json
            
            self.log("JSONエクスポート中...")
            export_data = export_to_json()
            
            assert "version" in export_data, "バージョン情報がありません"
            assert "project_data" in export_data, "プロジェクトデータがありません"
            
            self.log(f"エクスポートバージョン: {export_data['version']}")
            self.log(f"エクスポート日時: {export_data['exported_at']}")
            
            original_process_name = SessionManager.get_process_name()
            
            st.session_state.clear()
            
            self.log("JSONインポート中...")
            success = import_from_json(export_data)
            
            assert success, "インポートに失敗しました"
            
            restored_process_name = SessionManager.get_process_name()
            
            assert restored_process_name == original_process_name, "プロセス名が復元されていません"
            
            self.success("✅ データエクスポート/インポート成功")
            return True
            
        except Exception as e:
            self.error(f"❌ エクスポート/インポートエラー: {str(e)}")
            import traceback
            self.error(traceback.format_exc())
            return False
    
    def run_all_tests(self):
        self.log("=" * 60)
        self.log("PIMアプリケーション 全体動作確認テスト")
        self.log("=" * 60)
        
        tests = [
            ("タブ1: プロセス定義", self.test_tab1_process_definition),
            ("タブ2: 機能カテゴリ", self.test_tab2_functional_categories),
            ("タブ3: ノード定義", self.test_tab3_node_definition),
            ("タブ4: ノード評価（簡易）", self.test_tab4_node_evaluation_simple),
            ("タブ5: 行列分析", self.test_tab5_matrix_analysis),
            ("タブ6: ネットワーク可視化", self.test_tab6_network_visualization),
            ("タブ7: ネットワーク分析", self.test_tab7_network_analysis),
            ("タブ8: DSM最適化（基本）", self.test_tab8_dsm_optimization_basic),
            ("データエクスポート/インポート", self.test_data_export_import),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            self.log("")
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                self.error(f"テスト実行エラー ({test_name}): {str(e)}")
                failed += 1
        
        self.log("")
        self.log("=" * 60)
        self.log(f"テスト結果: {passed}個成功, {failed}個失敗")
        self.log("=" * 60)
        
        if failed == 0:
            self.success("🎉 全てのテストが成功しました！")
        else:
            self.error(f"⚠️ {failed}個のテストが失敗しました")
            self.log("\n失敗したテスト:")
            for error in self.errors:
                self.log(f"  - {error}")
        
        return failed == 0
    
    def save_report(self, filename="test_report.txt"):
        with open(filename, "w", encoding="utf-8") as f:
            for line in self.results:
                f.write(line + "\n")
        self.log(f"\nテストレポートを保存しました: {filename}")


if __name__ == "__main__":
    tester = WorkflowTester()
    success = tester.run_all_tests()
    tester.save_report("tests/test_report.txt")
    
    sys.exit(0 if success else 1)
