import React from "react";
import ReactDOM from "react-dom/client";
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";
import NetworkMapsViewer from "./NetworkMapsViewer";
import { ComponentArgs } from "./types";

interface State {
  args: ComponentArgs | null;
}

class StreamlitNetworkMaps extends StreamlitComponentBase<State> {
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

    if (!args || !args.diagram_data) {
      return (
        <div style={{ padding: "20px", textAlign: "center", color: "#888" }}>
          データを読み込み中...
        </div>
      );
    }

    return (
      <div style={{ height: `${args.height}px`, width: "100%" }}>
        <NetworkMapsViewer
          diagramData={args.diagram_data}
          enableInteraction={args.enable_interaction}
          cameraMode={args.camera_mode}
          onNodeClick={this.handleNodeClick}
        />
      </div>
    );
  }
}

const StreamlitComponent = withStreamlitConnection(StreamlitNetworkMaps);

const root = ReactDOM.createRoot(document.getElementById("root")!);
root.render(
  <React.StrictMode>
    <StreamlitComponent />
  </React.StrictMode>
);
