export interface CytoscapeNodeData {
  id: string;
  name: string;
  label: string;
  level: number;
  category?: string;
  pagerank?: number;
  betweenness?: number;
  in_degree?: number;
  out_degree?: number;
}

export interface CytoscapeEdgeData {
  id: string;
  source: string;
  target: string;
  label: string;
  name: string;
}

export interface CytoscapeNode {
  data: CytoscapeNodeData;
  position?: {
    x: number;
    y: number;
  };
}

export interface CytoscapeEdge {
  data: CytoscapeEdgeData;
}

export interface GraphData {
  nodes: CytoscapeNode[];
  edges: CytoscapeEdge[];
}

export interface NetworkMetrics {
  [nodeName: string]: {
    pagerank?: number;
    betweenness?: number;
    in_degree?: number;
    out_degree?: number;
  };
}

export interface ComponentArgs {
  graph_data: GraphData;
  layout: string;
  height: number;
  threshold: number;
  network_metrics?: NetworkMetrics;
}
