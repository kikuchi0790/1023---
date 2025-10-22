/**
 * Arrow Generation Utilities
 * Generate custom 3D arrows for edges
 */

import { EdgePosition, Layer } from '../types';

/**
 * Generate cone arrows for edges
 */
export function generateCones(
  edges: Array<{
    position?: EdgePosition;
    info?: { weight?: string; score?: number };
  }>,
  layers: Layer[],
  color: string,
  place: 'to' | 'from' | 'tomiddle' | 'frommiddle',
  d: number
) {
  const cones: any[] = [];
  
  edges.forEach((edge) => {
    if (!edge.position || !edge.position.from || !edge.position.to) {
      return;
    }
    
    const from = edge.position.from;
    const to = edge.position.to;
    const u = to.x - from.x;
    const v = to.y - from.y;
    const w = -(to.z - from.z) * d;
    const toZ = (layers.length - to.z - 1) * d;
    const fromZ = (layers.length - from.z - 1) * d;
    
    let cone: any = {
      type: 'cone',
      colorscale: [[0, color], [1, color]],
      sizemode: 'absolute',
      sizeref: 0.4,
      anchor: 'tip',
      showscale: false,
      hoverinfo: 'none',
      name: `cone {from {x: ${from.x}, y: ${from.y}, z: ${fromZ}}, to {x: ${to.x}, y: ${to.y}, z: ${toZ}}}${place}`,
    };
    
    if (place === 'to') {
      cone = { ...cone, x: [to.x], y: [to.y], z: [toZ], u: [u], v: [v], w: [w] };
    } else if (place === 'from') {
      cone = { ...cone, x: [from.x], y: [from.y], z: [fromZ], u: [-u], v: [-v], w: [-w] };
    } else if (place === 'tomiddle') {
      cone = { 
        ...cone, 
        x: [(from.x + to.x) / 2], 
        y: [(from.y + to.y) / 2], 
        z: [(toZ + fromZ) / 2],
        u: [u],
        v: [v],
        w: [w]
      };
    } else {
      cone = { 
        ...cone, 
        x: [(from.x + to.x) / 2], 
        y: [(from.y + to.y) / 2], 
        z: [(toZ + fromZ) / 2],
        u: [-u],
        v: [-v],
        w: [-w]
      };
    }
    
    cones.push(cone);
  });
  
  return cones;
}

/**
 * Generate sphere markers for edges
 */
export function generateSpheres(
  edges: Array<{
    position?: EdgePosition;
  }>,
  layers: Layer[],
  color: string,
  place: 'to' | 'from',
  d: number
) {
  const spheres: any[] = [];
  
  edges.forEach((edge) => {
    if (!edge.position || !edge.position.from || !edge.position.to) {
      return;
    }
    
    const from = edge.position.from;
    const to = edge.position.to;
    const vX = to.x - from.x;
    const vY = to.y - from.y;
    const vZ = -(to.z - from.z) * d;
    const toZ = (layers.length - to.z - 1) * d;
    const fromZ = (layers.length - from.z - 1) * d;
    const norm = Math.sqrt(vX ** 2 + vY ** 2 + vZ ** 2);
    const scale = 0.2 / norm;
    const adjX = vX * scale;
    const adjY = vY * scale;
    const adjZ = vZ * scale;
    
    const sphere: any = {
      type: 'scatter3d',
      u: [0],
      v: [0],
      w: [0],
      colorscale: [[0, color], [1, color]],
      sizemode: 'absolute',
      sizeref: 0.3,
      marker: { size: 3, color: color },
      anchor: 'tip',
      showscale: false,
      hoverinfo: 'none',
      name: `sphere {from {x: ${from.x}, y: ${from.y}, z: ${fromZ}}, to {x: ${to.x}, y: ${to.y}, z: ${toZ}}}${place}`,
    };
    
    if (place === 'from') {
      sphere.x = [from.x + adjX];
      sphere.y = [from.y + adjY];
      sphere.z = [fromZ + adjZ];
    } else {
      sphere.x = [to.x - adjX];
      sphere.y = [to.y - adjY];
      sphere.z = [toZ - adjZ];
    }
    
    spheres.push(sphere);
  });
  
  return spheres;
}

/**
 * Generate custom mesh arrow
 */
export function generateCustomArrow(
  edge: {
    position?: EdgePosition;
    info?: { direction?: string; weight?: string };
  },
  layers: Layer[],
  d: number,
  edgeColor: string = 'black'
) {
  const arrows: any[] = [];
  
  if (!edge.position || !edge.position.from || !edge.position.to) {
    return arrows;
  }
  
  const from = edge.position.from;
  const to = edge.position.to;
  const vX = to.x - from.x;
  const vY = to.y - from.y;
  const vZ = -(to.z - from.z) * d;
  const toZ = (layers.length - to.z - 1) * d;
  const fromZ = (layers.length - from.z - 1) * d;
  const norm = Math.sqrt(vX ** 2 + vY ** 2 + vZ ** 2);
  const scale = 0.2 / norm;
  const adjX = vX * scale;
  const adjY = vY * scale;
  const adjZ = vZ * scale;
  
  const createArrow = (baseX: number, baseY: number, baseZ: number, tipX: number, tipY: number, tipZ: number) => {
    const normA = Math.sqrt(2 * vX ** 2 + 2 * vY ** 2 + 2 * vZ ** 2 - 2 * vX * vY - 2 * vY * vZ - 2 * vZ * vX);
    const a = [(vY - vZ) / normA, (vZ - vX) / normA, (vX - vY) / normA];
    const normB = Math.sqrt(2 * vX ** 2 + 2 * vY ** 2 + 2 * vZ ** 2 - 2 * vX * vY - 2 * vY * vZ - 2 * vZ * vX) * Math.sqrt(vX ** 2 + vY ** 2 + vZ ** 2);
    const b = [(vY * (vX - vY) - vZ * (vZ - vY)) / normB, (vZ * (vY - vZ) - vX * (vX - vY)) / normB, (vX * (vZ - vX) - vY * (vY - vZ)) / normB];
    
    const vertices = [
      [baseX + 0.15 * a[0] + 0.15 * b[0], baseY + 0.15 * a[1] + 0.15 * b[1], baseZ + 0.15 * a[2] + 0.15 * b[2]],
      [baseX - 0.15 * a[0] + 0.15 * b[0], baseY - 0.15 * a[1] + 0.15 * b[1], baseZ - 0.15 * a[2] + 0.15 * b[2]],
      [baseX - 0.15 * a[0] - 0.15 * b[0], baseY - 0.15 * a[1] - 0.15 * b[1], baseZ - 0.15 * a[2] - 0.15 * b[2]],
      [baseX + 0.15 * a[0] - 0.15 * b[0], baseY + 0.15 * a[1] - 0.15 * b[1], baseZ + 0.15 * a[2] - 0.15 * b[2]],
      [tipX, tipY, tipZ],
    ];
    
    const faces = [
      [0, 1, 4],
      [1, 2, 4],
      [2, 3, 4],
      [3, 0, 4],
      [0, 1, 2],
      [0, 2, 3]
    ];
    
    arrows.push({
      type: 'mesh3d',
      x: vertices.map((v) => v[0]),
      y: vertices.map((v) => v[1]),
      z: vertices.map((v) => v[2]),
      i: faces.map((f) => f[0]),
      j: faces.map((f) => f[1]),
      k: faces.map((f) => f[2]),
      color: edgeColor,
      opacity: 1,
      hoverinfo: 'none',
      name: 'arrow_body',
    });
  };
  
  if (edge.info?.direction === 'forward' || edge.info?.direction === 'bidirectional') {
    const tipX = to.x - adjX;
    const tipY = to.y - adjY;
    const tipZ = toZ - adjZ;
    const baseX = to.x - 4 * adjX;
    const baseY = to.y - 4 * adjY;
    const baseZ = toZ - 4 * adjZ;
    createArrow(baseX, baseY, baseZ, tipX, tipY, tipZ);
  }
  
  if (edge.info?.direction === 'backward' || edge.info?.direction === 'bidirectional') {
    const tipX = from.x + adjX;
    const tipY = from.y + adjY;
    const tipZ = fromZ + adjZ;
    const baseX = from.x + 4 * adjX;
    const baseY = from.y + 4 * adjY;
    const baseZ = fromZ + 4 * adjZ;
    createArrow(baseX, baseY, baseZ, tipX, tipY, tipZ);
  }
  
  return arrows;
}

/**
 * Determine edge color based on score
 */
export function getEdgeColor(score: number | undefined): string {
  if (score === undefined || score === 0) {
    return 'black';
  }
  
  const absScore = Math.abs(score);
  
  if (absScore >= 7) {
    return score > 0 ? '#004563' : '#9f1e35';  // Strong: dark blue / dark red
  } else if (absScore >= 4) {
    return score > 0 ? '#588da2' : '#c94c62';  // Medium: medium blue / medium red
  } else {
    return score > 0 ? '#c3dde2' : '#e9c1c9';  // Weak: light blue / light red
  }
}

/**
 * Determine edge width based on score
 */
export function getEdgeWidth(score: number | undefined): number {
  if (score === undefined || score === 0) {
    return 2;
  }
  
  const absScore = Math.abs(score);
  
  if (absScore >= 7) {
    return 4;  // Thick
  } else if (absScore >= 4) {
    return 3;  // Medium
  } else {
    return 2;  // Thin
  }
}
