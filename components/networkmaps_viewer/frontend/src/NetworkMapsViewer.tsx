import React, { useEffect, useRef } from "react";
import { SceneManager } from "./engine/Scene";
import { GeometryFactory } from "./engine/Geometries";
import { NetworkMapsData } from "./types";
import * as THREE from "three";

interface Props {
  diagramData: NetworkMapsData;
  enableInteraction: boolean;
  cameraMode: string;
  onNodeClick: (nodeId: string, nodeName: string) => void;
}

const NetworkMapsViewer: React.FC<Props> = ({
  diagramData,
  enableInteraction,
  cameraMode,
  onNodeClick,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneManagerRef = useRef<SceneManager | null>(null);
  const raycasterRef = useRef<THREE.Raycaster>(new THREE.Raycaster());
  const mouseRef = useRef<THREE.Vector2>(new THREE.Vector2());

  useEffect(() => {
    if (!containerRef.current) return;

    const sceneManager = new SceneManager(containerRef.current);
    sceneManagerRef.current = sceneManager;

    return () => {
      sceneManager.dispose();
    };
  }, []);

  useEffect(() => {
    if (!sceneManagerRef.current || !diagramData) return;

    sceneManagerRef.current.clearObjects();

    Object.entries(diagramData.L2.base).forEach(([id, base]) => {
      const baseMesh = GeometryFactory.createBase(base);
      sceneManagerRef.current!.addObject(baseMesh);
    });

    Object.entries(diagramData.L2.device).forEach(([id, device]) => {
      const deviceMesh = GeometryFactory.createDevice(device);
      deviceMesh.userData.id = id;
      sceneManagerRef.current!.addObject(deviceMesh);

      const label = GeometryFactory.createDeviceLabel(device, device.name);
      sceneManagerRef.current!.addObject(label);
    });

    Object.entries(diagramData.L2.link).forEach(([id, link]) => {
      const srcDevice = diagramData.L2.device[link.src_device];
      const dstDevice = diagramData.L2.device[link.dst_device];

      if (srcDevice && dstDevice) {
        const linkLine = GeometryFactory.createLink(link, srcDevice, dstDevice);
        sceneManagerRef.current!.addObject(linkLine);
      }
    });
  }, [diagramData]);

  useEffect(() => {
    if (sceneManagerRef.current) {
      sceneManagerRef.current.setCameraMode(cameraMode);
    }
  }, [cameraMode]);

  const handleClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!enableInteraction || !sceneManagerRef.current) return;

    const rect = containerRef.current!.getBoundingClientRect();
    mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    const scene = sceneManagerRef.current.scene;
    const camera = sceneManagerRef.current.camera;

    raycasterRef.current.setFromCamera(mouseRef.current, camera);
    const intersects = raycasterRef.current.intersectObjects(
      scene.children,
      true
    );

    for (const intersect of intersects) {
      if (intersect.object.userData.type === "device") {
        onNodeClick(
          intersect.object.userData.id,
          intersect.object.userData.name
        );
        break;
      }
    }
  };

  return (
    <div
      ref={containerRef}
      onClick={handleClick}
      style={{
        width: "100%",
        height: "100%",
        cursor: enableInteraction ? "pointer" : "default",
      }}
    />
  );
};

export default NetworkMapsViewer;
