"""
Cytoscape 2DビューアのStreamlitコンポーネント定義
"""

import os
import streamlit.components.v1 as components
from typing import Dict, Any, Optional

_RELEASE = os.getenv("NETWORKMAPS_RELEASE", "false").lower() == "true"

if not _RELEASE:
    _component_func = components.declare_component(
        "cytoscape_2d_viewer",
        url="http://localhost:3003",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "cytoscape_2d_viewer",
        path=build_dir
    )


def cytoscape_2d_viewer(
    graph_data: Dict[str, Any],
    layout: str = "cose",
    height: int = 600,
    threshold: float = 2.0,
    network_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Cytoscape 2Dビューアを表示
    
    Parameters
    ----------
    graph_data : Dict[str, Any]
        Cytoscape形式のJSONデータ
        {
            "nodes": [{"data": {"id": "n1", "name": "ノード1", ...}}, ...],
            "edges": [{"data": {"id": "e1", "source": "n1", "target": "n2", ...}}, ...]
        }
    layout : str
        レイアウトアルゴリズム ("cose", "breadthfirst", "circle", "grid")
    height : int
        コンポーネントの高さ (px)
    threshold : float
        エッジ表示の閾値
    network_metrics : Optional[Dict[str, Dict[str, float]]]
        ネットワーク分析メトリクス（ノード名 → {pagerank, betweenness, ...}）
    key : str, optional
        Streamlitコンポーネントキー
    
    Returns
    -------
    Dict[str, Any] or None
        選択されたノード情報
    """
    return _component_func(
        graph_data=graph_data,
        layout=layout,
        height=height,
        threshold=threshold,
        network_metrics=network_metrics,
        key=key,
        default=None
    )
