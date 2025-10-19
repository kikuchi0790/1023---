import * as THREE from "three";
import { Device, Link, Base } from "../types";

export class GeometryFactory {
  static createDevice(device: Device): THREE.Mesh {
    const geometry = new THREE.BoxGeometry(device.sx, device.sy, device.sz);

    const material = new THREE.MeshPhongMaterial({
      color: device.color1,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(device.px, device.py, device.pz);
    mesh.rotation.set(device.rx, device.ry, device.rz);

    mesh.userData = {
      type: "device",
      id: "",
      name: device.name,
    };

    return mesh;
  }

  static createLink(
    link: Link,
    srcDevice: Device,
    dstDevice: Device
  ): THREE.Line {
    const points = [
      new THREE.Vector3(srcDevice.px, srcDevice.py, srcDevice.pz),
      new THREE.Vector3(dstDevice.px, dstDevice.py, dstDevice.pz),
    ];

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: link.color,
      linewidth: 2,
    });

    const line = new THREE.Line(geometry, material);
    line.userData = {
      type: "link",
      data: link.data,
    };

    return line;
  }

  static createBase(base: Base): THREE.Mesh {
    const geometry = new THREE.BoxGeometry(base.sx, base.sy, base.sz);
    const material = new THREE.MeshPhongMaterial({
      color: base.color1,
      transparent: true,
      opacity: 0.3,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(base.px, base.py, base.pz);
    mesh.rotation.set(base.rx, base.ry, base.rz);

    return mesh;
  }

  static createDeviceLabel(device: Device, text: string): THREE.Sprite {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d")!;
    canvas.width = 256;
    canvas.height = 64;

    context.fillStyle = "rgba(255, 255, 255, 0.8)";
    context.fillRect(0, 0, canvas.width, canvas.height);

    context.fillStyle = "black";
    context.font = "24px Arial";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(text, canvas.width / 2, canvas.height / 2);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(material);

    sprite.position.set(device.px, device.py + 1.5, device.pz);
    sprite.scale.set(3, 0.75, 1);

    return sprite;
  }
}
