"""
OPM Viewer Streamlit Component
Python API for the OPM 3D viewer component
"""

import os
import streamlit.components.v1 as components
from typing import Dict, Any, Optional

# Check if we're in development or production mode
_RELEASE = os.getenv("NETWORKMAPS_RELEASE", "").lower() == "true"

if not _RELEASE:
    _component_func = components.declare_component(
        "opm_viewer",
        url="http://localhost:3001",  # Dev server URL
    )
else:
    # Production: use the built component
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "opm_viewer", 
        path=build_dir
    )


def opm_viewer(
    opm_data: Dict[str, Any],
    height: int = 700,
    camera_mode: str = "3d",
    enable_edit: bool = False,
    enable_2d_view: bool = False,
    enable_edge_bundling: bool = False,
    key: Optional[str] = None
) -> Optional[Dict]:
    """
    Display an OPM 3D viewer component
    
    Parameters
    ----------
    opm_data : Dict[str, Any]
        OPM data structure containing layers, nodes, edges, etc.
        Expected structure:
        {
            "layers": List[{layer: str, color: str, isvisible: bool}],
            "nodes": List[{key: str, node: {name: str, layer: str}, position: {x, y, z}}],
            "edges": List[{key: str, type: str, edge: {...}, position: {...}}],
            "planeData": {m: int, d: int},
            ...
        }
    
    height : int, optional
        Height of the component in pixels (default: 700)
    
    camera_mode : str, optional
        Camera mode: "3d" or "2d" (default: "3d")
    
    enable_edit : bool, optional
        Enable node editing features (default: False)
    
    enable_2d_view : bool, optional
        Enable 2D projection view (default: False)
    
    enable_edge_bundling : bool, optional
        Enable edge bundling visualization (default: False)
    
    key : str, optional
        Unique key for the component
    
    Returns
    -------
    Optional[Dict]
        Component return value (currently None)
    """
    
    component_value = _component_func(
        opm_data=opm_data,
        height=height,
        camera_mode=camera_mode,
        enable_edit=enable_edit,
        enable_2d_view=enable_2d_view,
        enable_edge_bundling=enable_edge_bundling,
        key=key,
        default=None
    )
    
    return component_value


__all__ = ["opm_viewer"]
