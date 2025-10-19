"""
NetworkMaps統合: PIMデータをNetworkMaps形式に変換

このモジュールは以下を行います:
1. PIMのノードと隣接行列をNetworkMaps JSON形式に変換
2. 3D空間座標の計算（force-directed layout 3D版）
3. スコア値に基づくビジュアルマッピング（色、太さ）
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
import json


class NetworkMapsConverter:
    """PIMデータをNetworkMaps形式に変換するクラス"""
    
    def __init__(self, scale: float = 10.0):
        """
        Parameters
        ----------
        scale : float
            3D空間のスケール係数（デフォルト: 10.0）
        """
        self.scale = scale
        self.id_gen = 1000
    
    def convert(
        self,
        nodes: List[str],
        adjacency_matrix: np.ndarray,
        categories: List[str],
        idef0_data: Dict[str, Dict[str, Any]],
        evaluations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        メイン変換関数
        
        Parameters
        ----------
        nodes : List[str]
            ノード名のリスト
        adjacency_matrix : np.ndarray
            隣接行列（N×N）、値は-9～+9のスコア
        categories : List[str]
            機能カテゴリのリスト（時系列順）
        idef0_data : Dict[str, Dict[str, Any]]
            IDEF0データ（カテゴリごとのInput/Mechanism/Output）
        evaluations : List[Dict], optional
            評価データ（スコアと理由）
        
        Returns
        -------
        Dict[str, Any]
            NetworkMaps互換のJSONデータ
        """
        positions, node_layers = self._calculate_hierarchical_positions(
            nodes, categories, idef0_data
        )
        
        devices = {}
        device_id_map = {}
        
        # 機能カテゴリの仮想ノードを作成
        for i, category in enumerate(categories):
            func_node_name = f"【{category}】"
            device_id = str(self.id_gen + i)
            device_id_map[func_node_name] = device_id
            devices[device_id] = self._create_device(
                device_id=device_id,
                node_name=func_node_name,
                position=positions[func_node_name],
                layer="function"
            )
        
        # 実ノードを作成
        id_offset = len(categories)
        for i, node_name in enumerate(nodes):
            device_id = str(self.id_gen + id_offset + i)
            device_id_map[node_name] = device_id
            layer = node_layers.get(node_name, "unknown")
            devices[device_id] = self._create_device(
                device_id=device_id,
                node_name=node_name,
                position=positions[node_name],
                layer=layer
            )
        
        links = {}
        link_id_start = self.id_gen + len(categories) + len(nodes)
        link_counter = 0
        
        # 実ノード間のリンク
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                score = adjacency_matrix[i, j]
                if score != 0:
                    link_id = str(link_id_start + link_counter)
                    src_id = device_id_map[nodes[i]]
                    dst_id = device_id_map[nodes[j]]
                    
                    reason = self._get_evaluation_reason(
                        evaluations, nodes[i], nodes[j]
                    )
                    
                    links[link_id] = self._create_link(
                        link_id=link_id,
                        src_id=src_id,
                        dst_id=dst_id,
                        score=score,
                        reason=reason
                    )
                    link_counter += 1
        
        # 機能カテゴリから配下ノードへのリンク（構造を示すため）
        for category in categories:
            func_node_name = f"【{category}】"
            func_id = device_id_map[func_node_name]
            
            if category in idef0_data:
                idef0 = idef0_data[category]
                child_nodes = []
                
                if idef0.get("inputs"):
                    child_nodes.extend(idef0["inputs"])
                if idef0.get("mechanisms"):
                    child_nodes.extend(idef0["mechanisms"])
                if idef0.get("outputs"):
                    child_nodes.extend(idef0["outputs"])
                
                for child_node in child_nodes:
                    if child_node in device_id_map:
                        link_id = str(link_id_start + link_counter)
                        links[link_id] = self._create_link(
                            link_id=link_id,
                            src_id=func_id,
                            dst_id=device_id_map[child_node],
                            score=0,
                            reason="構造リンク",
                            is_structural=True
                        )
                        link_counter += 1
        
        return {
            "version": 3,
            "type": "network",
            "settings": {
                "id_gen": link_id_start + link_counter,
                "bg_color": 0xf0f0f0,
                "shapes": ["1"]
            },
            "L2": {
                "device": devices,
                "link": links,
                "base": self._create_category_planes(categories),
                "text": {},
                "symbol": {}
            }
        }
    
    def _calculate_hierarchical_positions(
        self,
        nodes: List[str],
        categories: List[str],
        idef0_data: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, str]]:
        """
        階層的3D空間座標を計算（プロセスフロー構造）
        
        Parameters
        ----------
        nodes : List[str]
            全ノードのリスト
        categories : List[str]
            機能カテゴリのリスト（時系列順）
        idef0_data : Dict[str, Dict[str, Any]]
            IDEF0データ
        
        Returns
        -------
        Tuple[Dict[str, Tuple], Dict[str, str]]
            (ノード名→座標の辞書, ノード名→層名の辞書)
        """
        positions = {}
        node_layers = {}
        
        CATEGORY_SPACING = self.scale * 1.0  # カテゴリ間隔を縮小
        NODE_SPACING = 1.5
        
        # Y軸の高さ定義
        LAYER_Y = {
            "function": 15,
            "output": 10,
            "mechanism": 5,
            "input": 0
        }
        
        # Z軸の奥行き定義（層ごとに分離）
        LAYER_Z = {
            "function": -6,    # 最も奥
            "output": -3,      # 奥から2番目
            "mechanism": 0,    # 中央
            "input": 3         # 手前
        }
        
        for cat_idx, category in enumerate(categories):
            base_x = cat_idx * CATEGORY_SPACING
            
            # 機能カテゴリノード（最上層）
            func_node_name = f"【{category}】"
            positions[func_node_name] = (base_x, LAYER_Y["function"], LAYER_Z["function"])
            
            if category not in idef0_data:
                continue
            
            idef0 = idef0_data[category]
            
            # Inputノード（最下層）
            if idef0.get("inputs"):
                inputs = idef0["inputs"]
                for i, input_node in enumerate(inputs):
                    offset_x = (i - len(inputs)/2) * NODE_SPACING
                    positions[input_node] = (
                        base_x + offset_x,
                        LAYER_Y["input"],
                        LAYER_Z["input"]
                    )
                    node_layers[input_node] = "input"
            
            # Mechanismノード（中层）
            if idef0.get("mechanisms"):
                mechanisms = idef0["mechanisms"]
                for i, mech_node in enumerate(mechanisms):
                    offset_x = (i - len(mechanisms)/2) * NODE_SPACING
                    positions[mech_node] = (
                        base_x + offset_x,
                        LAYER_Y["mechanism"],
                        LAYER_Z["mechanism"]
                    )
                    node_layers[mech_node] = "mechanism"
            
            # Outputノード（上層）
            if idef0.get("outputs"):
                outputs = idef0["outputs"]
                for i, output_node in enumerate(outputs):
                    offset_x = (i - len(outputs)/2) * NODE_SPACING
                    positions[output_node] = (
                        base_x + offset_x,
                        LAYER_Y["output"],
                        LAYER_Z["output"]
                    )
                    node_layers[output_node] = "output"
        
        # positions辞書にないノードはデフォルト位置
        for node in nodes:
            if node not in positions:
                positions[node] = (0, 2, 0)
                node_layers[node] = "unknown"
        
        return positions, node_layers
    
    def _create_device(
        self,
        device_id: str,
        node_name: str,
        position: Tuple[float, float, float],
        layer: str = "unknown"
    ) -> Dict[str, Any]:
        """
        NetworkMaps deviceオブジェクト生成
        
        Parameters
        ----------
        device_id : str
            デバイスID
        node_name : str
            ノード名
        position : Tuple[float, float, float]
            3D座標(x, y, z)
        layer : str
            レイヤー名（function/input/mechanism/output/unknown）
        
        Returns
        -------
        Dict[str, Any]
            NetworkMaps device オブジェクト
        """
        # 層ごとに色と形状を変更
        layer_config = {
            "function": {
                "type": "CUBE",
                "color1": 0x9966ff,  # 紫
                "color2": 0x7744dd,
                "scale": (2.0, 1.5, 2.0)
            },
            "output": {
                "type": "SPHERE",
                "color1": 0xff9933,  # オレンジ
                "color2": 0xdd7711,
                "scale": (0.8, 0.8, 0.8)
            },
            "mechanism": {
                "type": "CYLINDER",
                "color1": 0x44cc44,  # 緑
                "color2": 0x22aa22,
                "scale": (0.7, 1.2, 0.7)
            },
            "input": {
                "type": "CUBE",
                "color1": 0x4488ee,  # 青
                "color2": 0x3366cc,
                "scale": (0.8, 0.8, 0.8)
            },
            "unknown": {
                "type": "CUBE",
                "color1": 0xcccccc,  # グレー
                "color2": 0xaaaaaa,
                "scale": (0.6, 0.6, 0.6)
            }
        }
        
        config = layer_config.get(layer, layer_config["unknown"])
        sx, sy, sz = config["scale"]
        
        return {
            "type": config["type"],
            "name": node_name,
            "px": position[0],
            "py": position[1],
            "pz": position[2],
            "rx": 0,
            "ry": 0,
            "rz": 0,
            "sx": sx,
            "sy": sy,
            "sz": sz,
            "color1": config["color1"],
            "color2": config["color2"],
            "base": "0",
            "vrfs": {
                "default": {
                    "name": "default",
                    "interfaces": {}
                }
            },
            "infobox_type": "l",
            "data": [],
            "urls": {}
        }
    
    def _create_link(
        self,
        link_id: str,
        src_id: str,
        dst_id: str,
        score: float,
        reason: Optional[str] = None,
        is_structural: bool = False
    ) -> Dict[str, Any]:
        """
        NetworkMaps linkオブジェクト生成
        
        Parameters
        ----------
        link_id : str
            リンクID
        src_id : str
            送信元デバイスID
        dst_id : str
            送信先デバイスID
        score : float
            評価スコア（-9～+9）
        reason : str, optional
            評価理由
        is_structural : bool
            構造リンク（機能→配下ノード）かどうか
        
        Returns
        -------
        Dict[str, Any]
            NetworkMaps link オブジェクト
        """
        if is_structural:
            # 構造リンクは薄いグレー
            color = 0xcccccc
            thickness = 0.3
        else:
            color = self._score_to_color(score)
            thickness = min(abs(score) / 9.0, 1.0)
        
        link_data = {
            "src_device": src_id,
            "dst_device": dst_id,
            "src_vrf": "default",
            "dst_vrf": "default",
            "type": "l",
            "color": color,
            "shaft_color": color,
            "head_color": color,
            "shaft_type": "s",
            "head_type": "f",
            "tail_type": "n",
            "infobox_type": "l",
            "data": [],
            "urls": {}
        }
        
        if reason:
            link_data["data"] = [
                {"key": "評価理由", "value": reason},
                {"key": "スコア", "value": f"{score:+.1f}"}
            ]
        
        return link_data
    
    def _create_category_planes(self, categories: List[str]) -> Dict[str, Any]:
        """
        層別の背景planeを生成（全カテゴリ横断）
        
        Parameters
        ----------
        categories : List[str]
            機能カテゴリのリスト
        
        Returns
        -------
        Dict[str, Any]
            NetworkMaps base オブジェクト（Floor + 層別Plane群）
        """
        CATEGORY_SPACING = self.scale * 1.0
        total_width = len(categories) * CATEGORY_SPACING
        
        # 層別の定義
        LAYERS = [
            {"name": "Function", "y": 15, "z": -6, "color": 0xf0e6ff, "height": 2},  # 薄い紫
            {"name": "Output", "y": 10, "z": -3, "color": 0xfff0e6, "height": 2},    # 薄いオレンジ
            {"name": "Mechanism", "y": 5, "z": 0, "color": 0xe6ffe6, "height": 2},  # 薄い緑
            {"name": "Input", "y": 0, "z": 3, "color": 0xe6f0ff, "height": 2},       # 薄い青
        ]
        
        bases = {}
        
        # Floor（床）
        bases["0"] = {
            "type": "F",
            "name": "Floor",
            "px": (len(categories) - 1) * CATEGORY_SPACING / 2,
            "py": -0.5,
            "pz": 0,
            "rx": 0, "ry": 0, "rz": 0,
            "sx": total_width + 5,
            "sy": 0.5,
            "sz": 12,
            "color1": 0xffffff,
            "color2": 0xf5f5f5,
            "t1name": "b1_t1",
            "t2name": "b2_t1",
            "tsx": 1,
            "tsy": 1,
            "data": [],
            "urls": {}
        }
        
        # 層別の水平Plane（全カテゴリを横断）
        for idx, layer in enumerate(LAYERS):
            plane_id = str(idx + 1)
            bases[plane_id] = {
                "type": "P",
                "name": f"{layer['name']}層",
                "px": (len(categories) - 1) * CATEGORY_SPACING / 2,
                "py": layer["y"],
                "pz": layer["z"],
                "rx": 0,
                "ry": 0,
                "rz": 0,
                "sx": total_width + 3,
                "sy": layer["height"],
                "sz": 0.1,
                "color1": layer["color"],
                "color2": layer["color"],
                "t1name": "b1_t1",
                "t2name": "b2_t1",
                "tsx": 1,
                "tsy": 1,
                "data": [
                    {"key": "層", "value": layer["name"]}
                ],
                "urls": {},
                "opacity": 0.2
            }
        
        return bases
    
    @staticmethod
    def _score_to_color(score: float) -> int:
        """
        評価スコアを色コードに変換
        
        -9: 赤 (0xff0000)
         0: 黄 (0xffff00)
        +9: 緑 (0x00ff00)
        
        Parameters
        ----------
        score : float
            評価スコア（-9～+9）
        
        Returns
        -------
        int
            16進数カラーコード
        """
        normalized = max(-1.0, min(1.0, score / 9.0))
        
        if normalized < 0:
            ratio = (normalized + 1.0)
            r = 0xff
            g = int(0xff * ratio)
            b = 0x00
        else:
            ratio = normalized
            r = int(0xff * (1 - ratio))
            g = 0xff
            b = 0x00
        
        return (r << 16) | (g << 8) | b
    
    @staticmethod
    def _get_evaluation_reason(
        evaluations: Optional[List[Dict[str, Any]]],
        src_node: str,
        dst_node: str
    ) -> Optional[str]:
        """評価データから理由を取得"""
        if not evaluations:
            return None
        
        for eval_data in evaluations:
            if (eval_data.get("source") == src_node and 
                eval_data.get("target") == dst_node):
                return eval_data.get("reason")
        
        return None


def convert_pim_to_networkmaps(
    nodes: List[str],
    adjacency_matrix: np.ndarray,
    categories: List[str],
    idef0_data: Dict[str, Dict[str, Any]],
    evaluations: Optional[List[Dict[str, Any]]] = None,
    scale: float = 10.0
) -> Dict[str, Any]:
    """
    便利関数: PIMデータをNetworkMaps形式に変換
    
    Parameters
    ----------
    nodes : List[str]
        ノード名のリスト
    adjacency_matrix : np.ndarray
        隣接行列
    categories : List[str]
        機能カテゴリのリスト（時系列順）
    idef0_data : Dict[str, Dict[str, Any]]
        IDEF0データ（カテゴリごとのInput/Mechanism/Output）
    evaluations : List[Dict], optional
        評価データ
    scale : float
        3D空間のスケール
    
    Returns
    -------
    Dict[str, Any]
        NetworkMaps JSON
    """
    converter = NetworkMapsConverter(scale=scale)
    return converter.convert(nodes, adjacency_matrix, categories, idef0_data, evaluations)
