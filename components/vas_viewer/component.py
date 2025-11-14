"""
VAS Viewer Streamlit Component
Python API for the VAS 3D viewer component
"""

import os
import streamlit.components.v1 as components
from typing import Dict, Any, Optional

_RELEASE = os.getenv("NETWORKMAPS_RELEASE", "").lower() == "true"

if not _RELEASE:
    _component_func = components.declare_component(
        "vas_viewer",
        url="http://localhost:3002",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "vas_viewer", 
        path=build_dir
    )


def vas_viewer(
    vas_data: Dict[str, Any],
    height: int = 700,
    camera_mode: str = "3d",
    enable_search: bool = True,
    enable_filters: bool = True,
    show_level_control: bool = True,
    score_threshold: float = 0.5,
    key: Optional[str] = None
) -> Optional[Dict]:
    """
    Display a VAS System 3D Viewer component
    
    Parameters
    ----------
    vas_data : Dict[str, Any]
        VAS data structure containing nodes and links
        Expected structure:
        {
            "nodes": List[{
                "id": str,
                "labels": List[str],
                "properties": {
                    "name": str,
                    "level": int,
                    "category": str,
                    ...
                }
            }],
            "links": List[{
                "id": str,
                "type": str,
                "startNode": str,
                "endNode": str,
                "properties": {
                    "score": float,
                    "direction": "positive" | "negative",
                    ...
                }
            }]
        }
    
    height : int, optional
        Height of the component in pixels (default: 700)
    
    camera_mode : str, optional
        Camera mode: "3d" or "2d" (default: "3d")
    
    enable_search : bool, optional
        Enable node search functionality (default: True)
    
    enable_filters : bool, optional
        Enable type/level filters (default: True)
    
    show_level_control : bool, optional
        Show level selection controls (default: True)
    
    score_threshold : float, optional
        Minimum score to display edges (default: 0.5)
    
    key : str, optional
        Unique key for the component
    
    Returns
    -------
    Optional[Dict]
        Component return value (currently None)
    """
    
    component_value = _component_func(
        vas_data=vas_data,
        height=height,
        camera_mode=camera_mode,
        enable_search=enable_search,
        enable_filters=enable_filters,
        show_level_control=show_level_control,
        score_threshold=score_threshold,
        key=key,
        default=None
    )
    
    return component_value


__all__ = ["vas_viewer"]
