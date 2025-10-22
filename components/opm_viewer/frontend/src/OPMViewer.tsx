/**
 * OPM Viewer Component
 * Main 3D visualization component using Plotly.js
 */

import React, { useEffect, useRef, useState } from 'react';
import Plotly from 'plotly.js-dist';
import { 
  OPMViewerProps, 
  PlotlyData, 
  PlotlyLayout, 
  PlotlyCamera,
  OPMNode,
  OPMEdge
} from './types';
import { 
  generatePlanes, 
  layerIndexToZ, 
  calculateNodeSize,
  getNodeColorByType
} from './utils/layoutEngine';
import { 
  generateCones, 
  generateSpheres, 
  generateCustomArrow,
  getEdgeColor,
  getEdgeWidth
} from './utils/arrowGen';

const OPMViewer: React.FC<OPMViewerProps> = ({
  opmData,
  height = 700,
  cameraMode = '3d',
  enableEdit = false,
  enable2DView = false,
  enableEdgeBundling = false,
}) => {
  const plotlyRef = useRef<HTMLDivElement>(null);
  const [camera, setCamera] = useState<PlotlyCamera | null>(null);
  const [plotData, setPlotData] = useState<PlotlyData[]>([]);
  const [selectedNodeIndex, setSelectedNodeIndex] = useState<number | null>(null);
  
  useEffect(() => {
    if (!plotlyRef.current || !opmData) {
      return;
    }
    
    // Generate plot data
    const data = generatePlotData();
    setPlotData(data);
    
    // Render plot
    renderPlot(data);
    
    // Cleanup
    return () => {
      if (plotlyRef.current) {
        Plotly.purge(plotlyRef.current);
      }
    };
  }, [opmData, enable2DView, enableEdgeBundling]);
  
  useEffect(() => {
    // Update camera when mode changes
    if (plotlyRef.current && plotData.length > 0) {
      const layout = createLayout();
      Plotly.relayout(plotlyRef.current, layout);
    }
  }, [cameraMode, camera]);
  
  const generatePlotData = (): PlotlyData[] => {
    const { layers, nodes, edges, planeData } = opmData;
    const { m, d } = planeData;
    const data: PlotlyData[] = [];
    
    // 1. Generate planes (layers)
    if (!enable2DView) {
      const planes = generatePlanes(layers, m, d);
      data.push(...planes);
    }
    
    // 2. Generate nodes
    const visibleNodes = nodes.filter(n => n.isvisible !== false);
    visibleNodes.forEach((node, index) => {
      const layerIndex = layers.findIndex(l => l.layer === node.node.layer);
      const defaultColor = layerIndex !== -1 ? layers[layerIndex].color : 'gray';
      const nodeColor = getNodeColorByType(node.nodeType, defaultColor);
      
      if (!node.position) {
        return;
      }
      
      let nodeZ = layerIndexToZ(node.position.z, layers.length, d);
      let nodeY = node.position.y;
      
      if (enable2DView) {
        nodeY = node.position.y + node.position.z * 5;
        nodeZ = 0;
      }
      
      // Calculate node size based on connections
      const connectedEdges = edges.filter(
        e => (e.edge.fromkey === node.key || e.edge.tokey === node.key) && e.isvisible !== false
      );
      const nodeSize = calculateNodeSize(connectedEdges.length);
      
      data.push({
        x: [node.position.x],
        y: [nodeY],
        z: [nodeZ],
        type: 'scatter3d',
        mode: 'markers+text',
        marker: { 
          size: nodeSize, 
          color: selectedNodeIndex === index ? 'red' : nodeColor 
        },
        name: node.node.name,
        text: node.node.name,
        hoverinfo: 'text',
      });
    });
    
    // 3. Generate edges
    const visibleEdges = edges.filter(e => e.isvisible !== false && e.position);
    visibleEdges.forEach((edge) => {
      if (!edge.position) {
        return;
      }
      
      const fromZ = layerIndexToZ(edge.position.from.z, layers.length, d);
      const toZ = layerIndexToZ(edge.position.to.z, layers.length, d);
      let fromY = edge.position.from.y;
      let toY = edge.position.to.y;
      
      if (enable2DView) {
        fromY = edge.position.from.y + edge.position.from.z * 5;
        toY = edge.position.to.y + edge.position.to.z * 5;
      }
      
      const edgeColor = getEdgeColor(edge.info?.score);
      const edgeWidth = getEdgeWidth(edge.info?.score);
      
      // Simple line edge
      data.push({
        x: [edge.position.from.x, edge.position.to.x],
        y: [enable2DView ? fromY : edge.position.from.y, enable2DView ? toY : edge.position.to.y],
        z: [enable2DView ? 0 : fromZ, enable2DView ? 0 : toZ],
        type: 'scatter3d',
        mode: 'lines',
        line: { color: edgeColor, width: edgeWidth },
        name: `${edge.key}`,
        hoverinfo: 'none',
        text: `${edge.type}`,
      });
      
      // Generate custom arrows if info is available
      if (edge.info?.direction && edge.info?.weight) {
        const arrows = generateCustomArrow(edge, layers, d, edgeColor);
        data.push(...arrows);
      }
    });
    
    return data;
  };
  
  const createLayout = (): PlotlyLayout => {
    return {
      scene: {
        xaxis: { visible: false },
        yaxis: { visible: false },
        zaxis: { visible: false },
        camera: camera || undefined,
      },
      hovermode: 'closest',
      margin: { l: 10, r: 10, t: 10, b: 10 },
      showlegend: false,
      paper_bgcolor: 'gray',
    };
  };
  
  const renderPlot = (data: PlotlyData[]) => {
    if (!plotlyRef.current || data.length === 0) {
      return;
    }
    
    const layout = createLayout();
    
    Plotly.newPlot(plotlyRef.current, data, layout, {
      displayModeBar: true,
      displaylogo: false,
      responsive: true,
    });
    
    // Setup click event
    if (plotlyRef.current) {
      (plotlyRef.current as any).on('plotly_click', (event: any) => {
        const pointData = event.points[0];
        const dataType = pointData.data.type;
        const mode = pointData.data.mode;
        
        if (dataType === 'scatter3d' && mode === 'markers+text') {
          setSelectedNodeIndex(event.points[0].curveNumber);
        }
      });
      
      // Setup relayout event to capture camera changes
      (plotlyRef.current as any).on('plotly_relayout', (eventData: any) => {
        if (eventData['scene.camera']) {
          setCamera(eventData['scene.camera']);
        }
      });
    }
  };
  
  return (
    <div 
      ref={plotlyRef} 
      style={{ 
        width: '100%', 
        height: `${height}px`,
        backgroundColor: 'gray',
        borderRadius: '8px',
      }}
    />
  );
};

export default OPMViewer;
