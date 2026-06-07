export type PredictionMode = "mode_a" | "mode_b";

export interface DetectionResponse {
  filename: string;
  columns: string[];
  sensitive_fields: string[];
  inferred_paper_count: number | null;
  detected_max_scores: Record<string, number | null>;
  row_count: number;
  subjects: SubjectDetection[];
}

export interface SubjectDetection {
  subject_key: string;
  subject_code: string | null;
  subject_name: string | null;
  inferred_paper_count: number | null;
  row_count: number;
  detected_max_scores: Record<string, number | null>;
}

export interface MetadataEntry {
  subject_name?: string;
  subject_code?: string;
  paper_count?: number;
  p1_max?: number;
  p2_max?: number;
  p3_max?: number;
  p4_max?: number;
}

export interface ProcessingResponse {
  mode: PredictionMode;
  rows: number;
  exports: Record<string, string>;
  metrics: Array<Record<string, string | number | null>>;
  rankings: Array<Record<string, string | number | null>>;
  plots: Record<string, any>;
  summary: Record<string, string | number | null>;
  warnings: string[];
  errors: string[];
}

export interface AdaSafeExportResponse {
  rows: number;
  export_path: string;
  sensitive_fields: string[];
  columns: string[];
}

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
export const API_DISPLAY_URL = API_BASE_URL || "Vite proxy -> http://127.0.0.1:8002";

function apiUrl(path: string): string {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    return payload.detail ?? JSON.stringify(payload);
  } catch {
    return response.text();
  }
}

function networkErrorMessage(error: unknown): string {
  if (error instanceof TypeError) {
    return `Could not reach backend through ${API_DISPLAY_URL}. Check that Uvicorn is running on port 8002 and restart the Vite frontend after config changes.`;
  }
  return error instanceof Error ? error.message : "Request failed";
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(apiUrl("/api/health"));
    return response.ok;
  } catch {
    return false;
  }
}

export async function detectDataset(file: File): Promise<DetectionResponse> {
  const form = new FormData();
  form.append("file", file);
  try {
    const response = await fetch(apiUrl("/api/detect"), { method: "POST", body: form });
    if (!response.ok) throw new Error(await readError(response));
    return response.json();
  } catch (error) {
    throw new Error(networkErrorMessage(error));
  }
}

export async function exportAdaSafeDataset(file: File): Promise<AdaSafeExportResponse> {
  const form = new FormData();
  form.append("file", file);
  try {
    const response = await fetch(apiUrl("/api/export/ada-safe"), { method: "POST", body: form });
    if (!response.ok) throw new Error(await readError(response));
    return response.json();
  } catch (error) {
    throw new Error(networkErrorMessage(error));
  }
}

export async function processDataset(
  file: File,
  mode: PredictionMode,
  paperCounts: MetadataEntry[],
  maxScores: MetadataEntry[]
): Promise<ProcessingResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append(
    "payload",
    JSON.stringify({
      mode,
      paper_counts: paperCounts.filter((entry) => entry.paper_count),
      max_scores: maxScores
    })
  );
  try {
    const response = await fetch(apiUrl("/api/process"), { method: "POST", body: form });
    if (!response.ok) throw new Error(await readError(response));
    return response.json();
  } catch (error) {
    throw new Error(networkErrorMessage(error));
  }
}
