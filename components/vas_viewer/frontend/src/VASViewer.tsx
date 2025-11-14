import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
// @ts-ignore - OrbitControls型定義の問題を回避
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { VASViewerProps, VASNode, VASLink, NodeFilterState, SelectedNodeInfo } from './types';
import './styles.css';

const VASViewer: React.FC<VASViewerProps> = ({
  vasData,
  height = 700,
  cameraMode = '3d',
  enableSearch = true,
  enableFilters = true,
  showLevelControl = true,
  scoreThreshold = 0.5,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const nodeMeshesRef = useRef<Map<string, THREE.Mesh>>(new Map());
  const linkMeshesRef = useRef<Map<string, THREE.Line>>(new Map());

  const [filterState, setFilterState] = useState<NodeFilterState>({
    searchQuery: '',
    selectedLevels: new Set([0, 1, 2]),
    selectedTypes: new Set(['System', 'Process', 'Component', 'CommunityNode']),
    neighborDepth: 0,
  });

  const [selectedNode, setSelectedNode] = useState<SelectedNodeInfo | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    initThreeJS();
    createVisualization();

    const animate = () => {
      requestAnimationFrame(animate);
      if (controlsRef.current) controlsRef.current.update();
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
    animate();

    return () => {
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, [vasData, cameraMode]);

  useEffect(() => {
    updateVisibility();
  }, [filterState]);

  const initThreeJS = () => {
    const container = containerRef.current!;
    const width = container.clientWidth;
    const height = container.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(15, 15, 15);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 10);
    scene.add(directionalLight);

    const gridHelper = new THREE.GridHelper(50, 50, 0x444444, 0x222222);
    scene.add(gridHelper);
  };

  const createVisualization = () => {
    const scene = sceneRef.current!;
    
    nodeMeshesRef.current.forEach(mesh => scene.remove(mesh));
    linkMeshesRef.current.forEach(line => scene.remove(line));
    nodeMeshesRef.current.clear();
    linkMeshesRef.current.clear();

    const nodePositions = new Map<string, THREE.Vector3>();

    vasData.nodes.forEach((node, idx) => {
      const level = node.properties.level || 0;
      const pos2dX = node.properties.pos_2d_x || 0;
      const pos2dY = node.properties.pos_2d_y || 0;

      const position = new THREE.Vector3(
        pos2dX * 10 || (Math.cos((idx * 2 * Math.PI) / vasData.nodes.length) * 10),
        level * 5,
        pos2dY * 10 || (Math.sin((idx * 2 * Math.PI) / vasData.nodes.length) * 10)
      );
      nodePositions.set(node.id, position);

      const color = getNodeColor(node);
      const geometry = new THREE.SphereGeometry(0.5, 32, 32);
      const material = new THREE.MeshStandardMaterial({ color });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(position);
      mesh.userData = { node };

      scene.add(mesh);
      nodeMeshesRef.current.set(node.id, mesh);

      const sprite = createTextSprite(node.properties.name);
      sprite.position.set(position.x, position.y + 0.8, position.z);
      scene.add(sprite);
    });

    vasData.links.forEach(link => {
      const startPos = nodePositions.get(link.startNode);
      const endPos = nodePositions.get(link.endNode);

      if (!startPos || !endPos || Math.abs(link.properties.score) < scoreThreshold) return;

      const points = [startPos, endPos];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const color = link.properties.direction === 'positive' ? 0x4a9eff : 0xff4a6e;
      const material = new THREE.LineBasicMaterial({
        color,
        linewidth: Math.min(Math.abs(link.properties.score) / 3, 3),
        transparent: true,
        opacity: 0.6,
      });
      const line = new THREE.Line(geometry, material);
      line.userData = { link };

      scene.add(line);
      linkMeshesRef.current.set(link.id, line);
    });
  };

  const getNodeColor = (node: VASNode): number => {
    const type = node.labels[0];
    const colorMap: Record<string, number> = {
      System: 0x70e483,
      Process: 0x3bc3ff,
      Component: 0xcc7f30,
      CommunityNode: 0x9b59b6,
    };
    return colorMap[type] || 0x888888;
  };

  const createTextSprite = (text: string): THREE.Sprite => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d')!;
    canvas.width = 256;
    canvas.height = 64;

    context.fillStyle = 'rgba(0, 0, 0, 0.7)';
    context.fillRect(0, 0, canvas.width, canvas.height);

    context.font = '24px Arial';
    context.fillStyle = 'white';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text.substring(0, 20), canvas.width / 2, canvas.height / 2);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(2, 0.5, 1);

    return sprite;
  };

  const updateVisibility = () => {
    const { searchQuery, selectedLevels, selectedTypes } = filterState;

    nodeMeshesRef.current.forEach((mesh, nodeId) => {
      const node = mesh.userData.node as VASNode;
      const matchesSearch = !searchQuery || node.properties.name.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesLevel = selectedLevels.has(node.properties.level);
      const matchesType = selectedTypes.has(node.labels[0]);

      mesh.visible = matchesSearch && matchesLevel && matchesType;
    });

    linkMeshesRef.current.forEach((line, linkId) => {
      const link = line.userData.link as VASLink;
      const startMesh = nodeMeshesRef.current.get(link.startNode);
      const endMesh = nodeMeshesRef.current.get(link.endNode);

      line.visible = !!(startMesh?.visible && endMesh?.visible);
    });
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFilterState(prev => ({ ...prev, searchQuery: e.target.value }));
  };

  const toggleLevel = (level: number) => {
    setFilterState(prev => {
      const newLevels = new Set(prev.selectedLevels);
      if (newLevels.has(level)) {
        newLevels.delete(level);
      } else {
        newLevels.add(level);
      }
      return { ...prev, selectedLevels: newLevels };
    });
  };

  const toggleType = (type: string) => {
    setFilterState(prev => {
      const newTypes = new Set(prev.selectedTypes);
      if (newTypes.has(type)) {
        newTypes.delete(type);
      } else {
        newTypes.add(type);
      }
      return { ...prev, selectedTypes: newTypes };
    });
  };

  return (
    <div style={{ display: 'flex', height: `${height}px`, background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%)' }}>
      <div style={{ width: '30%', padding: '20px', color: 'white', overflowY: 'auto', borderRight: '1px solid rgba(74, 158, 255, 0.3)' }}>
        <h2 style={{ color: '#4a9eff', fontSize: '18px', marginBottom: '15px' }}>🔍 VAS 3D Viewer</h2>

        {enableSearch && (
          <div style={{ marginBottom: '20px' }}>
            <input
              type="text"
              placeholder="Search nodes..."
              value={filterState.searchQuery}
              onChange={handleSearchChange}
              style={{
                width: '100%',
                padding: '10px',
                background: 'rgba(20, 20, 35, 0.8)',
                border: '2px solid rgba(74, 158, 255, 0.3)',
                borderRadius: '8px',
                color: 'white',
                fontSize: '14px',
              }}
            />
          </div>
        )}

        {enableFilters && (
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ fontSize: '14px', marginBottom: '10px', color: '#4a9eff' }}>📊 Node Types</h3>
            {['System', 'Process', 'Component', 'CommunityNode'].map(type => (
              <label key={type} style={{ display: 'block', marginBottom: '8px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={filterState.selectedTypes.has(type)}
                  onChange={() => toggleType(type)}
                  style={{ marginRight: '8px' }}
                />
                {type}
              </label>
            ))}
          </div>
        )}

        {showLevelControl && (
          <div>
            <h3 style={{ fontSize: '14px', marginBottom: '10px', color: '#4a9eff' }}>📈 Levels</h3>
            {[0, 1, 2].map(level => (
              <label key={level} style={{ display: 'block', marginBottom: '8px', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={filterState.selectedLevels.has(level)}
                  onChange={() => toggleLevel(level)}
                  style={{ marginRight: '8px' }}
                />
                Level {level}
              </label>
            ))}
          </div>
        )}

        <div style={{ marginTop: '30px', fontSize: '12px', color: '#888' }}>
          <p>Nodes: {vasData.nodes.length}</p>
          <p>Links: {vasData.links.length}</p>
        </div>
      </div>

      <div ref={containerRef} style={{ flex: 1, position: 'relative' }} />
    </div>
  );
};

export default VASViewer;
