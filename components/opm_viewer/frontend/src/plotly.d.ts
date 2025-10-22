// Type declarations for plotly.js-dist
declare module 'plotly.js-dist' {
  export = Plotly;
}

declare namespace Plotly {
  function newPlot(
    root: HTMLElement,
    data: any[],
    layout?: any,
    config?: any
  ): Promise<any>;
  
  function relayout(
    root: HTMLElement,
    layout: any
  ): Promise<any>;
  
  function purge(root: HTMLElement): void;
  
  namespace events {
    function addEventListener(
      event: string,
      handler: (event: Event) => void
    ): void;
  }
  
  const RENDER_EVENT: string;
  
  function setComponentReady(): void;
  function setFrameHeight(height?: number): void;
}
