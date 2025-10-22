/**
 * Streamlit Component Entry Point
 */

import React from 'react';
import ReactDOM from 'react-dom';
import { Streamlit, RenderData } from 'streamlit-component-lib';
import OPMViewer from './OPMViewer';
import { OPMData } from './types';

interface ComponentState {
  opmData: OPMData | null;
  height: number;
  cameraMode: '3d' | '2d';
  enableEdit: boolean;
  enable2DView: boolean;
  enableEdgeBundling: boolean;
}

class StreamlitOPMViewer extends React.Component<{}, ComponentState> {
  constructor(props: {}) {
    super(props);
    this.state = {
      opmData: null,
      height: 700,
      cameraMode: '3d',
      enableEdit: false,
      enable2DView: false,
      enableEdgeBundling: false,
    };
  }
  
  componentDidMount() {
    Streamlit.setFrameHeight();
  }
  
  componentDidUpdate() {
    Streamlit.setFrameHeight();
  }
  
  public render(): React.ReactNode {
    const { opmData, height, cameraMode, enableEdit, enable2DView, enableEdgeBundling } = this.state;
    
    if (!opmData) {
      return (
        <div style={{ padding: '20px', textAlign: 'center' }}>
          <p>Loading OPM data...</p>
        </div>
      );
    }
    
    return (
      <OPMViewer
        opmData={opmData}
        height={height}
        cameraMode={cameraMode}
        enableEdit={enableEdit}
        enable2DView={enable2DView}
        enableEdgeBundling={enableEdgeBundling}
      />
    );
  }
}

function onRender(event: Event): void {
  const data = (event as CustomEvent<RenderData>).detail;
  const args = data.args;
  
  const component = (window as any).streamlitOPMViewer;
  if (component) {
    component.setState({
      opmData: args.opm_data || null,
      height: args.height || 700,
      cameraMode: args.camera_mode || '3d',
      enableEdit: args.enable_edit || false,
      enable2DView: args.enable_2d_view || false,
      enableEdgeBundling: args.enable_edge_bundling || false,
    });
  }
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();

const rootElement = document.getElementById('root');
if (rootElement) {
  const component = ReactDOM.render(<StreamlitOPMViewer />, rootElement);
  (window as any).streamlitOPMViewer = component;
}
