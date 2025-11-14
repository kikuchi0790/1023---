/**
 * VAS System 3D Viewer Type Definitions
 */

export interface VASNode {
  id: string;
  labels: string[];
  properties: {
    name: string;
    pim_node_id: number;
    level: number;
    category?: string;
    community_id?: number;
    community_label?: string;
    source: string;
    number?: string;
    通しno?: number;
    pos_2d_x?: number;
    pos_2d_y?: number;
    idef0?: {
      type: string;
      category: string;
      function: string;
      description: string;
    };
  };
}

export interface VASLink {
  id: string;
  type: string;
  startNode: string;
  endNode: string;
  properties: {
    source: string;
    score: number;
    abs_score: number;
    direction: 'positive' | 'negative';
    is_intra_community?: boolean;
    source_community?: number;
    target_community?: number;
  };
}

export interface VASData {
  nodes: VASNode[];
  links: VASLink[];
}

export interface VASViewerProps {
  vasData: VASData;
  height?: number;
  cameraMode?: '3d' | '2d';
  enableSearch?: boolean;
  enableFilters?: boolean;
  showLevelControl?: boolean;
  scoreThreshold?: number;
}

export interface NodeFilterState {
  searchQuery: string;
  selectedLevels: Set<number>;
  selectedTypes: Set<string>;
  neighborDepth: number;
}

export interface SelectedNodeInfo {
  node: VASNode;
  connectedNodes: VASNode[];
  incomingLinks: VASLink[];
  outgoingLinks: VASLink[];
}
