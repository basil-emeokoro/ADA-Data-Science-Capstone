import { Activity, AlertTriangle, CheckCircle2, Download, Loader2, Moon, Play, Shield, Sun, Upload } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { MetricTable } from "../components/MetricTable";
import { PlotPanel } from "../components/PlotPanel";
import { useTheme } from "../hooks/useTheme";
import {
  AdaSafeExportResponse,
  API_DISPLAY_URL,
  checkBackendHealth,
  detectDataset,
  DetectionResponse,
  exportAdaSafeDataset,
  MetadataEntry,
  PredictionMode,
  processDataset,
  ProcessingResponse
} from "../services/api";

const emptyMaxima: MetadataEntry = { subject_code: "", subject_name: "", p1_max: undefined, p2_max: undefined, p3_max: undefined, p4_max: undefined };
const emptyPaperCount: MetadataEntry = { subject_code: "", subject_name: "", paper_count: 3 };
const paperMaxKeys = ["p1_max", "p2_max", "p3_max", "p4_max"] as const;

export function App() {
  const { theme, toggleTheme } = useTheme();
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<PredictionMode>("mode_a");
  const [detection, setDetection] = useState<DetectionResponse | null>(null);
  const [metadataRows, setMetadataRows] = useState<MetadataEntry[]>([{ ...emptyPaperCount, ...emptyMaxima }]);
  const [adaExport, setAdaExport] = useState<AdaSafeExportResponse | null>(null);
  const [result, setResult] = useState<ProcessingResponse | null>(null);
  const [selectedPlot, setSelectedPlot] = useState("actual_vs_predicted");
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState("");
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  const flattenedPlots = useMemo(() => {
    if (!result?.plots) return {};
    const base: Record<string, any> = { ...(result.plots.eda ?? {}) };
    for (const [key, value] of Object.entries(result.plots)) {
      if (key === "eda" || key === "scenario_explainability") continue;
      base[key] = value;
    }
    const scenarioPlots = result.plots.scenario_explainability ?? {};
    for (const [scenario, charts] of Object.entries<Record<string, any>>(scenarioPlots)) {
      for (const [chartName, chart] of Object.entries(charts)) {
        base[`${scenario} ${chartName}`] = chart;
      }
    }
    return base;
  }, [result]);

  const exportRows = useMemo(() => {
    if (!result) return [];
    return Object.entries(result.exports ?? {}).map(([key, value]) => ({
      key,
      value,
      download: result.export_downloads?.[key]
    }));
  }, [result]);

  const qualityCounts = useMemo(() => {
    if (!result) return { clean: 0, invalid: 0, absent: 0, unpredictable: 0 };
    return {
      clean: Number(result.summary.clean_records ?? result.summary.total_rows ?? result.rows),
      invalid: Number(result.summary.invalid_records ?? 0),
      absent: Number(result.summary.absent_records ?? 0),
      unpredictable: Number(result.summary.unpredictable_records ?? 0)
    };
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
      setResult(null);
      setAdaExport(null);
      const subjects = response.subjects ?? [];
      const rows = subjects.length
        ? subjects.map((subject) => ({
            subject_code: subject.subject_code ?? "",
            subject_name: subject.subject_name ?? "",
            paper_count: subject.inferred_paper_count ?? response.inferred_paper_count ?? 3,
            ...emptyMaxima,
            ...subject.detected_max_scores
          }))
        : [{ ...emptyPaperCount, ...emptyMaxima, ...response.detected_max_scores }];
      setMetadataRows(rows);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detection failed");
    } finally {
      setBusy(false);
      setBusyLabel("");
    }
  }

  async function onRun() {
    if (!file) return;
    const validationMessage = metadataValidationMessage(metadataRows);
    if (validationMessage) {
      setError(validationMessage);
      return;
    }
    setBusy(true);
    setBusyLabel(mode === "mode_a" ? "Running benchmark pipeline" : "Predicting missing scores");
    setError(null);
    try {
      const response = await processDataset(file, mode, metadataRows, metadataRows);
      setResult(response);
      const firstPlot = Object.keys({ ...(response.plots?.eda ?? {}), ...(response.plots ?? {}) })[0];
      if (firstPlot) setSelectedPlot(firstPlot);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Processing failed");
    } finally {
      setBusy(false);
      setBusyLabel("");
    }
  }

  async function onExportAdaSafe() {
    if (!file) return;
    setBusy(true);
    setBusyLabel("Exporting ADA-safe dataset");
    setError(null);
    try {
      setAdaExport(await exportAdaSafeDataset(file));
    } catch (err) {
      setError(err instanceof Error ? err.message : "ADA-safe export failed");
    } finally {
      setBusy(false);
      setBusyLabel("");
    }
  }

  function updateMetadataRow(index: number, patch: MetadataEntry) {
    setMetadataRows((rows) => rows.map((row, rowIndex) => (rowIndex === index ? { ...row, ...patch } : row)));
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
            <button onClick={onExportAdaSafe} disabled={!file || busy || !detection || mode !== "mode_a"}><Download size={16} /> Export ADA-Safe Dataset</button>
            <button onClick={onRun} disabled={!file || busy}>{busy ? <Loader2 className="spin" size={16} /> : <Play size={16} />} Run Pipeline</button>
          </div>
          {busy && <div className="progressLine"><span /> {busyLabel}</div>}
          {adaExport && (
            <div className="successLine">
              <CheckCircle2 size={16} />
              ADA-safe export ready
              <a href={adaExport.download_url} download>Download CSV</a>
            </div>
          )}
          {error && <div className="alert"><AlertTriangle size={16} /> {error}</div>}
        </div>

        <div className="panel">
          <div className="panelHeader">
            <h2>Detection</h2>
            <span>{detection?.row_count ?? 0} rows</span>
          </div>
          <div className="kv">
            <span>Columns</span>
            <strong>{(detection?.columns ?? []).join(", ") || "Pending"}</strong>
            <span>Sensitive fields</span>
            <strong>{(detection?.sensitive_fields ?? []).join(", ") || "Pending"}</strong>
            <span>Inferred paper count</span>
            <strong>{detection?.inferred_paper_count ?? "Needs metadata if missing"}</strong>
            <span>Detected maxima</span>
            <strong>{detection ? formatDetectedMaxima(detection.detected_max_scores) : "Pending"}</strong>
          </div>
        </div>

        <div className="panel">
          <div className="panelHeader">
            <h2>Metadata Recovery</h2>
            <span>{metadataRows.length} subject batch{metadataRows.length === 1 ? "" : "es"}</span>
          </div>
          {detection && metadataValidationMessage(metadataRows) && (
            <div className="noticeLine"><AlertTriangle size={16} /> Maximum scores required before pipeline execution.</div>
          )}
          <div className="metadataTable">
            <div className="metadataHead">
              <span>Subject code</span>
              <span>Subject name</span>
              <span>Papers</span>
              <span>P1 max</span>
              <span>P2 max</span>
              <span>P3 max</span>
              <span>P4 max</span>
            </div>
            {metadataRows.map((row, index) => (
              <div className="metadataRow" key={`${row.subject_code}-${row.subject_name}-${index}`}>
                <input value={row.subject_code ?? ""} placeholder="Subject code" onChange={(event) => updateMetadataRow(index, { subject_code: event.target.value })} />
                <input value={row.subject_name ?? ""} placeholder="Subject name" onChange={(event) => updateMetadataRow(index, { subject_name: event.target.value })} />
                <input type="number" min={2} max={4} value={row.paper_count ?? 3} onChange={(event) => updateMetadataRow(index, normalizePaperCountPatch(Number(event.target.value)))} />
                {paperMaxKeys.map((key, paperIndex) => (
                  isApplicablePaper(row, paperIndex + 1) ? (
                    <input key={key} type="number" value={row[key] ?? ""} placeholder={key} onChange={(event) => updateMetadataRow(index, { [key]: event.target.value === "" ? undefined : Number(event.target.value) })} />
                  ) : (
                    <span className="metadataInactive" key={key}>N/A</span>
                  )
                ))}
              </div>
            ))}
            {metadataRows.length === 0 && <div className="emptyState compact">No subject metadata detected. Add subject metadata before running the pipeline.</div>}
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
          <section className="resultHeader">
            <div>
              <span>{result.mode === "mode_a" ? "Mode A Benchmark Results" : "Mode B Lite Prediction Results"}</span>
              <h2>{result.mode === "mode_a" ? "Experimental model comparison" : "Completed missing-score prediction"}</h2>
            </div>
            <strong>{Object.keys(result.exports ?? {}).length} exports available</strong>
          </section>

          <section className="summaryBand">
            <div><span>Total Rows</span><strong>{result.summary.total_rows ?? result.rows}</strong></div>
            <div><span>Subjects</span><strong>{result.summary.subjects_detected ?? 0}</strong></div>
            <div><span>Scenarios</span><strong>{result.summary.scenarios_run ?? 0}</strong></div>
            <div><span>Exports</span><strong>{result.summary.export_files_available ?? Object.keys(result.exports).length}</strong></div>
          </section>

          <section className="insightGrid">
            <div className="panel highlightPanel">
              <div className="panelHeader"><h2>Best Model</h2></div>
              <strong>{result.summary.best_overall_model ?? "N/A"}</strong>
              <span>Best RMSE: {formatNumber(result.summary.best_rmse)}</span>
            </div>
            <div className="panel qualityPanel">
              <div className="panelHeader"><h2>Dataset Quality</h2></div>
              <div className="qualityGrid">
                <span>Clean/training<strong>{qualityCounts.clean}</strong></span>
                <span>Invalid<strong>{qualityCounts.invalid}</strong></span>
                <span>Absent<strong>{qualityCounts.absent}</strong></span>
                <span>Unpredictable<strong>{qualityCounts.unpredictable}</strong></span>
              </div>
            </div>
          </section>

          {modeBCompleteRecordMessage(result) && (
            <section className="panel infoPanel">
              <div className="panelHeader"><h2>Mode B Guidance</h2></div>
              <p>No predictable missing scores found. Dataset contains complete valid records only.</p>
              <p>Use Mode A Benchmark to evaluate predictive performance.</p>
            </section>
          )}

          {((result.errors ?? []).length > 0 || (result.warnings ?? []).length > 0) && (
            <section className="panel">
              <div className="panelHeader"><h2>Validation Messages</h2></div>
              {[...(result.errors ?? []), ...(result.warnings ?? [])].map((message) => <div className="alert" key={message}><AlertTriangle size={16} /> {message}</div>)}
            </section>
          )}

          <section className="panel">
            <div className="panelHeader">
              <h2>{result.mode === "mode_a" ? "Mode A Export Package" : "Mode B Export Package"}</h2>
            </div>
            <div className="exports">
            {exportRows.map(({ key, value, download }) => (
              <div className="exportItem" key={key}>
                <Download size={16} />
                <span>{key.replaceAll("_", " ")}</span>
                <strong>{value}</strong>
                {download && <a href={download} download>Download</a>}
              </div>
            ))}
            </div>
          </section>

          <PlotPanel plots={flattenedPlots} selected={selectedPlot} onSelectedChange={setSelectedPlot} />
          <MetricTable rows={result.rankings} title="Model Ranking Table" />
          <MetricTable rows={result.metrics} title="Evaluation Metrics" />
        </>
      )}
    </main>
  );
}

function metadataValidationMessage(rows: MetadataEntry[]): string | null {
  if (!rows.length) return "Maximum scores required before pipeline execution.";
  for (const row of rows) {
    const paperCount = Number(row.paper_count ?? 0);
    if (![2, 3, 4].includes(paperCount)) return "Maximum scores required before pipeline execution.";
    for (let index = 1; index <= paperCount; index += 1) {
      const key = `p${index}_max` as keyof MetadataEntry;
      const value = Number(row[key]);
      if (!Number.isFinite(value) || value <= 0) return "Maximum scores required before pipeline execution.";
    }
  }
  return null;
}

function normalizePaperCountPatch(paperCount: number): MetadataEntry {
  const patch: MetadataEntry = { paper_count: paperCount };
  for (let index = paperCount + 1; index <= 4; index += 1) {
    patch[`p${index}_max` as keyof MetadataEntry] = undefined;
  }
  return patch;
}

function isApplicablePaper(row: MetadataEntry, paperIndex: number): boolean {
  return Number(row.paper_count ?? 0) >= paperIndex;
}

function formatDetectedMaxima(maxima: Record<string, number | null> | null | undefined): string {
  if (!maxima || Object.keys(maxima).length === 0) return "Missing - use metadata recovery";
  return JSON.stringify(maxima);
}

function modeBCompleteRecordMessage(result: ProcessingResponse | null): boolean {
  if (!result || result.mode !== "mode_b") return false;
  return (result.warnings ?? []).some((warning) => warning.includes("No predictable missing scores found"));
}

function formatNumber(value: string | number | null | undefined): string {
  if (typeof value === "number") return value.toFixed(3);
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(3) : value;
  }
  return "N/A";
}
