import React from 'react';
import ReactDOM from 'react-dom/client';
import { Streamlit, RenderData } from 'streamlit-component-lib';
import VASViewer from './VASViewer';
import { VASData } from './types';

interface StreamlitArgs {
  vas_data: VASData;
  height: number;
  camera_mode: '3d' | '2d';
  enable_search: boolean;
  enable_filters: boolean;
  show_level_control: boolean;
  score_threshold: number;
}

function StreamlitVASViewer() {
  const [args, setArgs] = React.useState<StreamlitArgs | null>(null);

  React.useEffect(() => {
    Streamlit.setFrameHeight();
  }, []);

  const onRender = React.useCallback((event: Event) => {
    const data = (event as CustomEvent<RenderData>).detail;
    setArgs(data.args as StreamlitArgs);
    Streamlit.setFrameHeight(data.args.height || 700);
  }, []);

  React.useEffect(() => {
    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
    Streamlit.setComponentReady();
    
    return () => {
      Streamlit.events.removeEventListener(Streamlit.RENDER_EVENT, onRender);
    };
  }, [onRender]);

  if (!args || !args.vas_data) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '400px',
        color: '#4a9eff',
        fontSize: '16px',
        fontFamily: 'Segoe UI, sans-serif',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%)'
      }}>
        Loading VAS 3D Viewer...
      </div>
    );
  }

  return (
    <VASViewer
      vasData={args.vas_data}
      height={args.height || 700}
      cameraMode={args.camera_mode || '3d'}
      enableSearch={args.enable_search !== false}
      enableFilters={args.enable_filters !== false}
      showLevelControl={args.show_level_control !== false}
      scoreThreshold={args.score_threshold || 0.5}
    />
  );
}

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(
  <React.StrictMode>
    <StreamlitVASViewer />
  </React.StrictMode>
);
