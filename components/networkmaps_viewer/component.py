"""
NetworkMaps 3DビューアのStreamlitコンポーネント定義
"""

import os
import streamlit.components.v1 as components
from typing import Dict, Any, Optional

_RELEASE = os.getenv("NETWORKMAPS_RELEASE", "false").lower() == "true"

if not _RELEASE:
    _component_func = components.declare_component(
        "networkmaps_3d_viewer",
        url="http://localhost:3002",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "networkmaps_3d_viewer",
        path=build_dir
    )


def networkmaps_3d_viewer(
    diagram_data: Dict[str, Any],
    height: int = 600,
    enable_interaction: bool = True,
    camera_mode: str = "3d",
    key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    NetworkMaps 3Dビューアを表示
    
    Parameters
    ----------
    diagram_data : Dict[str, Any]
        NetworkMaps形式のJSONデータ
    height : int
        コンポーネントの高さ (px)
    enable_interaction : bool
        インタラクション有効化
    camera_mode : str
        カメラモード ("3d" or "2d")
    key : str, optional
        Streamlitコンポーネントキー
    
    Returns
    -------
    Dict[str, Any] or None
        選択されたノード情報
    """
    return _component_func(
        diagram_data=diagram_data,
        height=height,
        enable_interaction=enable_interaction,
        camera_mode=camera_mode,
        key=key,
        default=None
    )
