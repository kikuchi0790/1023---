import React, { useEffect, useRef, useState } from "react";
import cytoscape, { Core, ElementDefinition, NodeSingular } from "cytoscape";
import tippy, { Instance as TippyInstance, sticky } from "tippy.js";
import { GraphData } from "./types";
import "tippy.js/dist/tippy.css";
import "cytoscape-panzoom/cytoscape.js-panzoom.css";

// @ts-ignore - 型定義なしのプラグイン
import popper from "cytoscape-popper";
// @ts-ignore - 型定義なしのプラグイン
import panzoom from "cytoscape-panzoom";

// Cytoscapeプラグイン登録
cytoscape.use(popper);
cytoscape.use(panzoom);

interface Props {
  graphData: GraphData;
  layout: string;
  onNodeClick: (nodeId: string, nodeName: string) => void;
}

const CytoscapeGraph: React.FC<Props> = ({ graphData, layout, onNodeClick }) => {
  const cyRef = useRef<Core | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const currentTooltipRef = useRef<TippyInstance | null>(null);
  const [exportReady, setExportReady] = useState(false);

  // ツールチップを作成
  const createTooltip = (node: NodeSingular) => {
    const nodeName = node.data("name");
    const nodeType = node.data("label");
    const pagerank = node.data("pagerank") || 0;
    const betweenness = node.data("betweenness") || 0;
    const inDegree = node.data("in_degree") || 0;
    const outDegree = node.data("out_degree") || 0;

    const typeLabels: Record<string, string> = {
      output: "成果物",
      mechanism: "手段",
      input: "材料・情報",
    };

    const content = `
      <div style="padding: 8px; font-size: 12px; line-height: 1.6;">
        <strong style="font-size: 14px;">${nodeName}</strong><br/>
        <span style="color: #666;">タイプ: ${typeLabels[nodeType] || nodeType}</span><br/>
        ${pagerank > 0 ? `<span style="color: #0074D9;">PageRank: ${pagerank.toFixed(4)}</span><br/>` : ""}
        ${betweenness > 0 ? `<span style="color: #2ECC40;">媒介中心性: ${betweenness.toFixed(4)}</span><br/>` : ""}
        ${inDegree > 0 ? `<span style="color: #FF851B;">入次数: ${inDegree.toFixed(2)}</span><br/>` : ""}
        ${outDegree > 0 ? `<span style="color: #B10DC9;">出次数: ${outDegree.toFixed(2)}</span>` : ""}
      </div>
    `;

    const popperInstance = (node as any).popper({
      content: () => {
        const div = document.createElement("div");
        return div;
      },
      popper: {
        placement: "top",
        modifiers: [
          {
            name: "preventOverflow",
            options: {
              boundary: containerRef.current,
            },
          },
        ],
      },
    });

    const dummyElement = document.createElement("div");
    const tippyInstance = tippy(dummyElement, {
      getReferenceClientRect: () => {
        const pos = node.renderedPosition();
        return {
          width: 0,
          height: 0,
          top: pos.y - 10,
          bottom: pos.y - 10,
          left: pos.x,
          right: pos.x,
          x: pos.x,
          y: pos.y - 10,
          toJSON: () => ({}),
        };
      },
      content,
      trigger: "manual",
      arrow: true,
      placement: "top",
      hideOnClick: false,
      sticky: "reference" as any,
      plugins: [sticky],
      theme: "light-border",
      allowHTML: true,
      appendTo: containerRef.current || document.body,
    });

    return tippyInstance;
  };

  // PNG/SVGエクスポート関数（グローバルに公開）
  useEffect(() => {
    if (cyRef.current) {
      (window as any).exportCytoscapeImage = (format: "png" | "svg") => {
        if (!cyRef.current) return;

        if (format === "png") {
          const blob = cyRef.current.png({ full: true, scale: 2 });
          const link = document.createElement("a");
          link.href = blob;
          link.download = `network-graph-${Date.now()}.png`;
          link.click();
        } else if (format === "svg") {
          // @ts-ignore - svg()メソッドは実行時に存在
          const svgContent = cyRef.current.svg({ full: true, scale: 1 });
          const blob = new Blob([svgContent], { type: "image/svg+xml" });
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = `network-graph-${Date.now()}.svg`;
          link.click();
          URL.revokeObjectURL(url);
        }
      };
      setExportReady(true);
    }

    return () => {
      delete (window as any).exportCytoscapeImage;
      setExportReady(false);
    };
  }, [cyRef.current]);

  useEffect(() => {
    if (!containerRef.current) return;

    // 既存のツールチップをクリーンアップ
    if (currentTooltipRef.current) {
      currentTooltipRef.current.destroy();
      currentTooltipRef.current = null;
    }

    // 既存のインスタンスを破棄
    if (cyRef.current) {
      cyRef.current.destroy();
    }

    // ノードサイズを計算（PageRankに基づく）
    const calculateNodeSize = (pagerank: number | undefined): number => {
      const baseSize = 30;
      const scaleFactor = 150;
      if (!pagerank || pagerank === 0) return baseSize;
      return baseSize + pagerank * scaleFactor;
    };

    // Cytoscape要素に変換（位置情報があれば含める）
    const elements: ElementDefinition[] = [
      ...graphData.nodes.map((node) => {
        const element: any = { 
          data: {
            ...node.data,
            size: calculateNodeSize(node.data.pagerank),
          } 
        };
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

    // エッジの太さを計算（スコアに基づく）
    const calculateEdgeWidth = (label: string): number => {
      const score = Math.abs(parseFloat(label));
      if (score >= 7) return 6; // 太い
      if (score >= 4) return 3; // 中
      return 1; // 細い
    };

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
            width: "data(size)",
            height: "data(size)",
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-width": 4,
            "border-color": "#FFD700",
            "overlay-color": "#FFD700",
            "overlay-padding": 8,
            "overlay-opacity": 0.3,
          },
        },
        {
          selector: "edge",
          style: {
            width: (ele: any) => calculateEdgeWidth(ele.data("label")),
            "line-color": (ele: any) => {
              const labelVal = parseFloat(ele.data("label"));
              if (labelVal >= 5) return "#2ECC40"; // 緑（強い正の関係）
              if (labelVal >= 2) return "#0074D9"; // 青（正の関係）
              if (labelVal >= -2) return "#AAAAAA"; // グレー（中立）
              if (labelVal >= -5) return "#FF851B"; // オレンジ（負の関係）
              return "#FF4136"; // 赤（強い負の関係）
            },
            "target-arrow-color": (ele: any) => {
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
        {
          selector: "edge.highlighted",
          style: {
            width: (ele: any) => calculateEdgeWidth(ele.data("label")) * 2,
            "line-color": "#FFD700",
            "target-arrow-color": "#FFD700",
            "z-index": 999,
          },
        },
        {
          selector: "node.faded",
          style: {
            opacity: 0.3,
          },
        },
        {
          selector: "edge.faded",
          style: {
            opacity: 0.15,
          },
        },
      ],
      layout: {
        name: layoutName,
        fit: true,
        padding: 50,
        animate: true,
        animationDuration: 500,
      } as any,
    });

    // ドラッグ可能に設定
    cy.nodes().grabify();

    // Panzoomコントロール追加
    (cy as any).panzoom({
      zoomFactor: 0.05,
      zoomDelay: 45,
      minZoom: 0.1,
      maxZoom: 10,
      fitPadding: 50,
      panSpeed: 10,
      panDistance: 10,
      panInactiveArea: 8,
      panIndicatorMinOpacity: 0.5,
      panMinPercentSpeed: 0.25,
      autodisableForMobile: true,
    });

    // ズームイベント（ラベル表示/非表示）
    cy.on("zoom", () => {
      if (cy.zoom() < 0.8) {
        cy.nodes().style("label", "");
        cy.edges().style("label", "");
      } else {
        cy.nodes().style("label", "data(name)");
        cy.edges().style("label", "data(label)");
      }
    });

    // ノードホバーイベント（ツールチップ表示）
    cy.on("mouseover", "node", (evt) => {
      const node = evt.target;
      // 既存のツールチップがあれば破棄
      if (currentTooltipRef.current) {
        currentTooltipRef.current.destroy();
      }
      const tooltip = createTooltip(node);
      tooltip.show();
      currentTooltipRef.current = tooltip;
    });

    cy.on("mouseout", "node", () => {
      if (currentTooltipRef.current) {
        currentTooltipRef.current.hide();
        setTimeout(() => {
          if (currentTooltipRef.current) {
            currentTooltipRef.current.destroy();
            currentTooltipRef.current = null;
          }
        }, 100);
      }
    });

    // ノードクリックイベント（ハイライト）
    cy.on("tap", "node", (evt) => {
      const node = evt.target;
      
      // 既存のハイライトをクリア
      cy.elements().removeClass("highlighted faded");

      // 選択ノードとその接続エッジ・ノードをハイライト
      const connectedEdges = node.connectedEdges();
      const connectedNodes = connectedEdges.connectedNodes();

      // それ以外をフェード
      cy.elements().not(node).not(connectedEdges).not(connectedNodes).addClass("faded");
      
      // 接続エッジをハイライト
      connectedEdges.addClass("highlighted");

      // Streamlitにノード情報を返す
      onNodeClick(node.data("id"), node.data("name"));
    });

    // 背景クリックでハイライト解除
    cy.on("tap", (evt) => {
      if (evt.target === cy) {
        cy.elements().removeClass("highlighted faded");
      }
    });

    cyRef.current = cy;

    return () => {
      if (currentTooltipRef.current) {
        currentTooltipRef.current.destroy();
        currentTooltipRef.current = null;
      }
      if (cyRef.current) {
        cyRef.current.destroy();
      }
    };
  }, [graphData, layout]);

  return (
    <div
      style={{ width: "100%", height: "100%", position: "relative" }}
      ref={containerRef}
    />
  );
};

export default CytoscapeGraph;
