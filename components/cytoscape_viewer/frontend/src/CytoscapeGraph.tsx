import React, { useEffect, useRef } from "react";
import cytoscape, { Core, ElementDefinition } from "cytoscape";
import { GraphData } from "./types";

interface Props {
  graphData: GraphData;
  layout: string;
  onNodeClick: (nodeId: string, nodeName: string) => void;
}

const CytoscapeGraph: React.FC<Props> = ({ graphData, layout, onNodeClick }) => {
  const cyRef = useRef<Core | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // 既存のインスタンスを破棄
    if (cyRef.current) {
      cyRef.current.destroy();
    }

    // Cytoscape要素に変換（位置情報があれば含める）
    const elements: ElementDefinition[] = [
      ...graphData.nodes.map((node) => {
        const element: any = { data: node.data };
        if (node.position) {
          element.position = node.position;
        }
        return element;
      }),
      ...graphData.edges.map((edge) => ({
        data: edge.data,
      })),
    ];
    
    // ノードに位置情報がある場合はpresetレイアウトを使用
    const hasPositions = graphData.nodes.some(node => node.position !== undefined);
    const layoutName = hasPositions ? "preset" : layout;

    // Cytoscapeインスタンス生成
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: "node",
          style: {
            "background-color": (ele) => {
              const label = ele.data("label");
              if (label === "output") return "#2ECC40"; // 緑
              if (label === "mechanism") return "#0074D9"; // 青
              if (label === "input") return "#FF851B"; // オレンジ
              return "#888"; // デフォルト
            },
            label: "data(name)",
            "text-valign": "center",
            "text-halign": "center",
            color: "#fff",
            "text-outline-color": "#000",
            "text-outline-width": 2,
            "font-size": "10px",
            "font-weight": "bold",
            width: 30,
            height: 30,
          },
        },
        {
          selector: "edge",
          style: {
            width: 2,
            "line-color": (ele) => {
              const labelVal = parseFloat(ele.data("label"));
              if (labelVal >= 5) return "#2ECC40"; // 緑（強い正の関係）
              if (labelVal >= 2) return "#0074D9"; // 青（正の関係）
              if (labelVal >= -2) return "#AAAAAA"; // グレー（中立）
              if (labelVal >= -5) return "#FF851B"; // オレンジ（負の関係）
              return "#FF4136"; // 赤（強い負の関係）
            },
            "target-arrow-color": (ele) => {
              const labelVal = parseFloat(ele.data("label"));
              if (labelVal >= 5) return "#2ECC40";
              if (labelVal >= 2) return "#0074D9";
              if (labelVal >= -2) return "#AAAAAA";
              if (labelVal >= -5) return "#FF851B";
              return "#FF4136";
            },
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            label: "data(label)",
            "font-size": "10px",
            "text-background-color": "#fff",
            "text-background-opacity": 0.8,
            "text-background-padding": "2px",
            "text-background-shape": "roundrectangle",
          },
        },
      ],
      layout: {
        name: layoutName,
        fit: true,
        padding: 50,
      } as any,
    });

    // ズームイベント
    cy.on("zoom", () => {
      if (cy.zoom() < 0.8) {
        cy.nodes().style("label", "");
        cy.edges().style("label", "");
      } else {
        cy.nodes().style("label", "data(name)");
        cy.edges().style("label", "data(label)");
      }
    });

    // ノードクリックイベント
    cy.on("tap", "node", (evt) => {
      const node = evt.target;
      onNodeClick(node.data("id"), node.data("name"));
    });

    cyRef.current = cy;

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
      }
    };
  }, [graphData, layout]);

  return (
    <div
      style={{ width: "100%", height: "100%" }}
      ref={containerRef}
    />
  );
};

export default CytoscapeGraph;
