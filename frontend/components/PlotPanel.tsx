import Plot from "react-plotly.js";

interface PlotPanelProps {
  plots: Record<string, any>;
  selected: string;
  onSelectedChange: (value: string) => void;
}

export function PlotPanel({ plots, selected, onSelectedChange }: PlotPanelProps) {
  const keys = Object.keys(plots);
  const active = plots[selected] ?? plots[keys[0]];
  return (
    <section className="panel plotPanel">
      <div className="panelHeader">
        <h2>Interactive Dashboard</h2>
        <select value={selected} onChange={(event) => onSelectedChange(event.target.value)}>
          {keys.map((key) => <option key={key} value={key}>{key.replaceAll("_", " ")}</option>)}
        </select>
      </div>
      {active ? (
        <Plot
          data={active.data ?? []}
          layout={{ ...(active.layout ?? {}), autosize: true, paper_bgcolor: "transparent", plot_bgcolor: "transparent" }}
          config={{ responsive: true, displaylogo: false }}
          useResizeHandler
          className="plot"
        />
      ) : (
        <div className="emptyState">No chart has been generated yet.</div>
      )}
    </section>
  );
}
