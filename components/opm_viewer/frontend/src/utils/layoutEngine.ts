/**
 * Layout Engine for OPM Visualization
 * Calculates positions, generates planes, and manages 3D layout
 */

import { Layer, PlaneData } from '../types';

/**
 * Calculate plane size based on node count
 */
export function calculatePlaneSize(nodeCount: number, baseSize: number = 9): number {
  if (nodeCount === 0) {
    return baseSize;
  }
  
  // Calculate max nodes per layer (assuming distributed)
  let m = 3;
  while (m * m / 5 < nodeCount) {
    m += 2;
  }
  m += 6;
  
  return m;
}

/**
 * Calculate distance between layers
 */
export function calculateDistance(layerCount: number): number {
  if (layerCount === 0) {
    return 3;
  }
  
  let d = 3;
  while (d * d / 2.5 < layerCount) {
    d += 1;
  }
  
  return d;
}

/**
 * Generate plane meshes for each layer
 */
export function generatePlanes(layers: Layer[], m: number, d: number) {
  const planes: any[] = [];
  const half = m / 2;
  
  for (let i = 0; i < layers.length; i++) {
    const { layer, color, isvisible } = layers[i];
    
    if (isvisible === false) {
      continue;
    }
    
    const z = d * (layers.length - i - 1);
    
    const plane = {
      x: [-half, -half, half, half, -half],
      y: [-half, half, half, -half, -half],
      z: [z, z, z, z, z],
      type: 'mesh3d',
      color: color,
      opacity: 0.5,
      name: `Layer ${layer}`,
      hoverinfo: 'none',
    };
    
    planes.push(plane);
  }
  
  return planes;
}

/**
 * Convert layer index to Z coordinate
 */
export function layerIndexToZ(layerIndex: number, layerCount: number, d: number): number {
  return d * (layerCount - layerIndex - 1);
}

/**
 * Check if position is available in grid
 */
export function isPositionAvailable(
  x: number,
  y: number,
  z: number,
  occupiedPositions: Set<string>
): boolean {
  const key = `${x},${y},${z}`;
  return !occupiedPositions.has(key);
}

/**
 * Generate available grid positions
 */
export function generateAvailableGrid(
  half: number,
  layerCount: number
): Array<{ x: number; y: number; z: number }> {
  const grid: Array<{ x: number; y: number; z: number }> = [];
  const size = Math.floor(half);
  
  for (let i = 0; i < layerCount; i++) {
    for (let j = Math.ceil(-half); j <= Math.floor(half); j++) {
      for (let k = Math.ceil(-half); k <= Math.floor(half); k++) {
        grid.push({ x: j, y: k, z: i });
      }
    }
  }
  
  return grid;
}

/**
 * Calculate node size based on connection count
 */
export function calculateNodeSize(connectionCount: number): number {
  return Math.max(4, Math.min(12, 4 + connectionCount * 0.8));
}

/**
 * Get node color based on type
 */
export function getNodeColorByType(nodeType: string | undefined, defaultColor: string): string {
  if (!nodeType) {
    return defaultColor;
  }
  
  const typeColors: Record<string, string> = {
    Output: '#70e483',     // Green
    Mechanism: '#3bc3ff',  // Blue
    Input: '#CC7F30',      // Orange
  };
  
  return typeColors[nodeType] || defaultColor;
}
