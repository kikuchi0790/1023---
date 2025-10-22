/**
 * OPM Viewer Component Types
 * TypeScript type definitions for OPM data structures
 */

export interface Position3D {
  x: number;
  y: number;
  z: number;
}

export interface Layer {
  layer: string;
  color: string;
  isvisible?: boolean;
}

export interface Node {
  name: string;
  layer: string;
}

export interface OPMNode {
  key: string;
  node: Node;
  position?: Position3D;
  isvisible?: boolean;
  nodeType?: "Output" | "Mechanism" | "Input";
}

export interface EdgeInfo {
  fromkey: string;
  fromname: string;
  fromlayer: string;
  tokey: string;
  toname: string;
  tolayer: string;
}

export interface EdgePosition {
  from: Position3D;
  to: Position3D;
}

export interface EdgeDetails {
  direction?: string;
  weight?: string;
  score?: number;
}

export interface OPMEdge {
  key: string;
  type: string;
  edge: EdgeInfo;
  position?: EdgePosition;
  info?: EdgeDetails;
  isvisible?: boolean;
}

export interface PlaneData {
  m: number;  // plane size (XY)
  d: number;  // layer distance (Z)
}

export interface OPMData {
  projectName: string;
  projectNumber: number;
  version: string;
  colorList: string[];
  layers: Layer[];
  nodes: OPMNode[];
  edges: OPMEdge[];
  planeData: PlaneData;
  nodePositions?: Array<{
    key: string;
    position: Position3D;
    name: string;
    layer: string;
  }>;
  edgePositions?: Array<{
    key: string;
    position: EdgePosition;
    fromname: string;
    fromlayer: string;
    toname: string;
    tolayer: string;
    type: string;
  }>;
}

export interface OPMViewerProps {
  opmData: OPMData;
  height?: number;
  cameraMode?: "3d" | "2d";
  enableEdit?: boolean;
  enable2DView?: boolean;
  enableEdgeBundling?: boolean;
}

// Plotly specific types
export interface PlotlyData {
  x?: number[];
  y?: number[];
  z?: number[];
  type: string;
  mode?: string;
  marker?: {
    size?: number;
    color?: string | string[];
  };
  line?: {
    color?: string;
    width?: number;
    shape?: string;
  };
  name?: string;
  text?: string | string[];
  hoverinfo?: string;
  opacity?: number;
  color?: string;
  colorscale?: Array<[number, string]>;
  // mesh3d specific
  i?: number[];
  j?: number[];
  k?: number[];
  // cone specific
  u?: number[];
  v?: number[];
  w?: number[];
  sizemode?: string;
  sizeref?: number;
  anchor?: string;
  showscale?: boolean;
}

export interface PlotlyLayout {
  scene: {
    xaxis: { visible: boolean };
    yaxis: { visible: boolean };
    zaxis: { visible: boolean };
    camera?: PlotlyCamera;
  };
  hovermode: string | boolean;
  margin: {
    l: number;
    r: number;
    t: number;
    b: number;
  };
  showlegend: boolean;
  paper_bgcolor: string;
}

export interface PlotlyCamera {
  eye?: {
    x: number;
    y: number;
    z: number;
  };
  center?: {
    x: number;
    y: number;
    z: number;
  };
  up?: {
    x: number;
    y: number;
    z: number;
  };
}

export interface PlotlyConfig {
  displayModeBar?: boolean;
  displaylogo?: boolean;
  responsive?: boolean;
}
