import React from "react";
import ReactDOM from "react-dom/client";
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";
import CytoscapeGraph from "./CytoscapeGraph";
import { ComponentArgs } from "./types";

interface State {
  args: ComponentArgs | null;
}

class StreamlitCytoscape extends StreamlitComponentBase<State> {
  state = {
    args: null as ComponentArgs | null,
  };

  componentDidMount() {
    Streamlit.setFrameHeight();
  }

  componentDidUpdate() {
    this.setState({ args: this.props.args as ComponentArgs });
    Streamlit.setFrameHeight();
  }

  handleNodeClick = (nodeId: string, nodeName: string) => {
    Streamlit.setComponentValue({
      node_id: nodeId,
      node_name: nodeName,
    });
  };

  render() {
    const { args } = this.state;

    if (!args || !args.graph_data) {
      return (
        <div style={{ padding: "20px", textAlign: "center", color: "#888" }}>
          データを読み込み中...
        </div>
      );
    }

    return (
      <div style={{ height: `${args.height}px`, width: "100%" }}>
        <CytoscapeGraph
          graphData={args.graph_data}
          layout={args.layout}
          onNodeClick={this.handleNodeClick}
        />
      </div>
    );
  }
}

const StreamlitComponent = withStreamlitConnection(StreamlitCytoscape);

const root = ReactDOM.createRoot(document.getElementById("root")!);
root.render(
  <React.StrictMode>
    <StreamlitComponent />
  </React.StrictMode>
);
