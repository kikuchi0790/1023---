/**
 * NetworkMaps データ型定義
 */

export interface NetworkMapsData {
  version: number;
  type: string;
  settings: {
    id_gen: number;
    bg_color: number;
    shapes: string[];
  };
  L2: {
    device: { [key: string]: Device };
    link: { [key: string]: Link };
    base: { [key: string]: Base };
    text?: any;
    symbol?: any;
  };
}

export interface Device {
  type: string;
  name: string;
  px: number;
  py: number;
  pz: number;
  rx: number;
  ry: number;
  rz: number;
  sx: number;
  sy: number;
  sz: number;
  color1: number;
  color2: number;
  base: string;
  vrfs?: any;
  infobox_type?: string;
  data?: any[];
  urls?: any;
}

export interface Link {
  src_device: string;
  dst_device: string;
  src_vrf?: string;
  dst_vrf?: string;
  type: string;
  color: number;
  shaft_color?: number;
  head_color?: number;
  shaft_type?: string;
  head_type?: string;
  tail_type?: string;
  data?: Array<{ key: string; value: string }>;
}

export interface Base {
  type: string;
  name: string;
  px: number;
  py: number;
  pz: number;
  rx: number;
  ry: number;
  rz: number;
  sx: number;
  sy: number;
  sz: number;
  color1: number;
  color2: number;
  t1name?: string;
  t2name?: string;
  tsx?: number;
  tsy?: number;
}

export interface ComponentArgs {
  diagram_data: NetworkMapsData;
  height: number;
  enable_interaction: boolean;
  camera_mode: string;
}

export interface SelectedNode {
  node_id: string;
  node_name: string;
}
