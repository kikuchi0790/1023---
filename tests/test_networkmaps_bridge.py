"""
NetworkMaps統合のユニットテスト
"""

import pytest
import numpy as np
from utils.networkmaps_bridge import (
    NetworkMapsConverter,
    convert_pim_to_networkmaps
)


class TestNetworkMapsConverter:
    """NetworkMapsConverterのテスト"""
    
    def test_score_to_color_negative(self):
        """負のスコアが赤系の色になることを確認"""
        color = NetworkMapsConverter._score_to_color(-9.0)
        assert color == 0xff0000
        
        color = NetworkMapsConverter._score_to_color(-4.5)
        r = (color >> 16) & 0xff
        g = (color >> 8) & 0xff
        b = color & 0xff
        assert r == 0xff
        assert 0 < g < 0xff
        assert b == 0x00
    
    def test_score_to_color_zero(self):
        """スコア0が黄色になることを確認"""
        color = NetworkMapsConverter._score_to_color(0.0)
        assert color == 0xffff00
    
    def test_score_to_color_positive(self):
        """正のスコアが緑系の色になることを確認"""
        color = NetworkMapsConverter._score_to_color(9.0)
        assert color == 0x00ff00
        
        color = NetworkMapsConverter._score_to_color(4.5)
        r = (color >> 16) & 0xff
        g = (color >> 8) & 0xff
        b = color & 0xff
        assert 0 < r < 0xff
        assert g == 0xff
        assert b == 0x00
    
    def test_calculate_3d_positions(self):
        """3D座標計算のテスト"""
        converter = NetworkMapsConverter(scale=10.0)
        
        adjacency = np.array([
            [0, 5, 0],
            [0, 0, 3],
            [2, 0, 0]
        ])
        
        positions = converter._calculate_3d_positions(adjacency)
        
        assert len(positions) == 3
        
        for pos in positions:
            assert len(pos) == 3
            x, y, z = pos
            assert y >= 0
    
    def test_create_device(self):
        """デバイス作成のテスト"""
        converter = NetworkMapsConverter()
        
        device = converter._create_device(
            device_id="1000",
            node_name="テストノード",
            position=(5.0, 2.0, -3.0)
        )
        
        assert device["name"] == "テストノード"
        assert device["px"] == 5.0
        assert device["py"] == 2.0
        assert device["pz"] == -3.0
        assert device["type"] == "CUBE"
    
    def test_create_link(self):
        """リンク作成のテスト"""
        converter = NetworkMapsConverter()
        
        link = converter._create_link(
            link_id="2000",
            src_id="1000",
            dst_id="1001",
            score=7.5,
            reason="強い正の影響があります"
        )
        
        assert link["src_device"] == "1000"
        assert link["dst_device"] == "1001"
        assert len(link["data"]) == 2
        assert link["data"][0]["value"] == "強い正の影響があります"
    
    def test_convert_full(self):
        """フル変換のテスト"""
        nodes = ["ノードA", "ノードB", "ノードC"]
        adjacency = np.array([
            [0, 5, 0],
            [0, 0, -3],
            [0, 0, 0]
        ])
        
        result = convert_pim_to_networkmaps(nodes, adjacency)
        
        assert result["version"] == 3
        assert result["type"] == "network"
        
        assert len(result["L2"]["device"]) == 3
        
        assert len(result["L2"]["link"]) == 2
        
        assert "0" in result["L2"]["base"]
        assert result["L2"]["base"]["0"]["type"] == "F"


class TestConvenienceFunction:
    """便利関数のテスト"""
    
    def test_convert_pim_to_networkmaps(self):
        """convert_pim_to_networkmaps関数のテスト"""
        nodes = ["A", "B"]
        adjacency = np.array([[0, 5], [3, 0]])
        
        result = convert_pim_to_networkmaps(nodes, adjacency, scale=15.0)
        
        assert result is not None
        assert "L2" in result
        assert len(result["L2"]["device"]) == 2
