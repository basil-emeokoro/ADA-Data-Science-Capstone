import { Activity, AlertTriangle, CheckCircle2, Download, Loader2, Moon, Play, Shield, Sun, Upload } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { MetricTable } from "../components/MetricTable";
import { PlotPanel } from "../components/PlotPanel";
import { useTheme } from "../hooks/useTheme";
import { API_DISPLAY_URL, checkBackendHealth, detectDataset, DetectionResponse, MetadataEntry, PredictionMode, processDataset, ProcessingResponse } from "../services/api";

const emptyMaxima: MetadataEntry = { subject_code: "", subject_name: "", p1_max: 40, p2_max: 60, p3_max: 100, p4_max: 100 };
const emptyPaperCount: MetadataEntry = { subject_code: "", subject_name: "", paper_count: 3 };

export function App() {
  const { theme, toggleTheme } = useTheme();
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<PredictionMode>("mode_a");
  const [detection, setDetection] = useState<DetectionResponse | null>(null);
  const [paperCounts, setPaperCounts] = useState<MetadataEntry[]>([emptyPaperCount]);
  const [maxScores, setMaxScores] = useState<MetadataEntry[]>([emptyMaxima]);
  const [result, setResult] = useState<ProcessingResponse | null>(null);
  const [selectedPlot, setSelectedPlot] = useState("actual_vs_predicted");
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState("");
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  const flattenedPlots = useMemo(() => {
    if (!result?.plots) return {};
    return { ...result.plots.eda, ...result.plots };
  }, [result]);

  useEffect(() => {
    checkBackendHealth().then(setBackendOnline);
  }, []);

  async function onDetect() {
    if (!file) return;
    setBusy(true);
    setBusyLabel("Detecting CSV structure");
    setError(null);
    try {
      const response = await detectDataset(file);
      setDetection(response);
      if (response.inferred_paper_count) {
        setPaperCounts([{ subject_code: "", subject_name: "", paper_count: response.inferred_paper_count }]);
      }
      setMaxScores([{ ...emptyMaxima, ...response.detected_max_scores }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detection failed");
    } finally {
      setBusy(false);
      setBusyLabel("");
    }
  }

  async function onRun() {
    if (!file) return;
    setBusy(true);
    setBusyLabel(mode === "mode_a" ? "Running benchmark pipeline" : "Predicting missing scores");
    setError(null);
    try {
      const response = await processDataset(file, mode, paperCounts, maxScores);
      setResult(response);
      const firstPlot = Object.keys({ ...response.plots?.eda, ...response.plots })[0];
      if (firstPlot) setSelectedPlot(firstPlot);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Processing failed");
    } finally {
      setBusy(false);
      setBusyLabel("");
    }
  }

  return (
    <main>
      <header className="topbar">
        <div>
          <h1>Predicting Missing Examination Component Scores</h1>
          <p>Privacy-preserving regression benchmarking and real missing-score prediction for 2-, 3-, and 4-paper examinations.</p>
          <div className="statusRow">
            <span className={`statusPill ${backendOnline ? "online" : backendOnline === false ? "offline" : ""}`}>
              {backendOnline ? <CheckCircle2 size={14} /> : <Activity size={14} />}
              Backend {backendOnline === null ? "checking" : backendOnline ? "online" : "offline"}
            </span>
            <span className="statusPill neutral">{API_DISPLAY_URL}</span>
          </div>
        </div>
        <button className="iconButton" onClick={toggleTheme} aria-label="Toggle theme" title="Toggle theme">
          {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
        </button>
      </header>

      <section className="workflow">
        <div className="panel controlPanel">
          <div className="panelHeader">
            <h2>Dataset Workflow</h2>
            <Shield size={20} />
          </div>
          <label className="dropzone">
            <Upload size={24} />
            <span>{file ? file.name : "Upload examination CSV"}</span>
            <input type="file" accept=".csv" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
          </label>
          <div className="segmented">
            <button className={mode === "mode_a" ? "active" : ""} onClick={() => setMode("mode_a")}>Mode A Benchmark</button>
            <button className={mode === "mode_b" ? "active" : ""} onClick={() => setMode("mode_b")}>Mode B Lite</button>
          </div>
          <div className="actions">
            <button onClick={onDetect} disabled={!file || busy}>{busy ? <Loader2 className="spin" size={16} /> : <Shield size={16} />} Detect</button>
            <button onClick={onRun} disabled={!file || busy}>{busy ? <Loader2 className="spin" size={16} /> : <Play size={16} />} Run Pipeline</button>
          </div>
          {busy && <div className="progressLine"><span /> {busyLabel}</div>}
          {error && <div className="alert"><AlertTriangle size={16} /> {error}</div>}
        </div>

        <div className="panel">
          <div className="panelHeader">
            <h2>Detection</h2>
            <span>{detection?.row_count ?? 0} rows</span>
          </div>
          <div className="kv">
            <span>Columns</span>
            <strong>{detection?.columns.join(", ") || "Pending"}</strong>
            <span>Sensitive fields</span>
            <strong>{detection?.sensitive_fields.join(", ") || "Pending"}</strong>
            <span>Inferred paper count</span>
            <strong>{detection?.inferred_paper_count ?? "Needs metadata if missing"}</strong>
            <span>Detected maxima</span>
            <strong>{detection ? JSON.stringify(detection.detected_max_scores) : "Pending"}</strong>
          </div>
        </div>

        <div className="panel">
          <div className="panelHeader">
            <h2>Metadata Recovery</h2>
          </div>
          <div className="gridForm">
            <input value={paperCounts[0]?.subject_code ?? ""} placeholder="Subject code" onChange={(e) => setPaperCounts([{ ...paperCounts[0], subject_code: e.target.value }])} />
            <input value={paperCounts[0]?.subject_name ?? ""} placeholder="Subject name" onChange={(e) => setPaperCounts([{ ...paperCounts[0], subject_name: e.target.value }])} />
            <input type="number" min={2} max={4} value={paperCounts[0]?.paper_count ?? 3} onChange={(e) => setPaperCounts([{ ...paperCounts[0], paper_count: Number(e.target.value) }])} />
            {["p1_max", "p2_max", "p3_max", "p4_max"].map((key) => (
              <input key={key} type="number" placeholder={key} value={(maxScores[0] as any)?.[key] ?? ""} onChange={(e) => setMaxScores([{ ...maxScores[0], [key]: Number(e.target.value) }])} />
            ))}
          </div>
        </div>
      </section>

      {!result && (
        <section className="readinessBand">
          <div>
            <strong>Mode A</strong>
            <span>Anonymize, export ADA-safe data, clean complete records, hide papers, train models, evaluate, explain.</span>
          </div>
          <div>
            <strong>Mode B Lite</strong>
            <span>Clean records, detect one missing paper, choose the scenario model, export predictions and reference exceptions.</span>
          </div>
          <div>
            <strong>Dashboard</strong>
            <span>Charts and ranking tables appear after detection and pipeline execution.</span>
          </div>
        </section>
      )}

      {result && (
        <>
          <section className="summaryBand">
            <div><span>Rows Processed</span><strong>{result.rows}</strong></div>
            <div><span>Metrics</span><strong>{result.metrics.length}</strong></div>
            <div><span>Warnings</span><strong>{result.warnings.length}</strong></div>
            <div><span>Errors</span><strong>{result.errors.length}</strong></div>
          </section>

          {(result.errors.length > 0 || result.warnings.length > 0) && (
            <section className="panel">
              <div className="panelHeader"><h2>Validation Messages</h2></div>
              {[...result.errors, ...result.warnings].map((message) => <div className="alert" key={message}><AlertTriangle size={16} /> {message}</div>)}
            </section>
          )}

          <section className="exports">
            {Object.entries(result.exports).map(([key, value]) => (
              <div className="exportItem" key={key}>
                <Download size={16} />
                <span>{key.replaceAll("_", " ")}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </section>

          <PlotPanel plots={flattenedPlots} selected={selectedPlot} onSelectedChange={setSelectedPlot} />
          <MetricTable rows={result.rankings} title="Model Ranking Table" />
          <MetricTable rows={result.metrics} title="Evaluation Metrics" />
        </>
      )}
    </main>
  );
}
