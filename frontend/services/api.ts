export type PredictionMode = "mode_a" | "mode_b";

export interface DetectionResponse {
  filename: string;
  columns: string[];
  sensitive_fields: string[];
  inferred_paper_count: number | null;
  detected_max_scores: Record<string, number | null>;
  row_count: number;
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
  warnings: string[];
  errors: string[];
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function detectDataset(file: File): Promise<DetectionResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE_URL}/api/detect`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
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
  const response = await fetch(`${API_BASE_URL}/api/process`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
