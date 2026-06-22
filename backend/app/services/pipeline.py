from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config.settings import get_settings
from backend.app.ml.explainability.plots import (
    actual_vs_predicted,
    eda_plots,
    feature_importance,
    partial_dependence_plot,
    residual_plot,
    shap_summary,
)
from backend.app.ml.preprocessing.cleaning import PAPER_SCORE_COLUMNS, clean_dataset
from backend.app.ml.preprocessing.scenarios import build_features_for_target, scenario_targets
from backend.app.ml.training.trainer import TrainedScenario, train_scenario
from backend.app.schemas.pipeline import (
    AdaSafeExportResponse,
    CleaningPreviewResponse,
    ProcessingRequest,
    ProcessingResponse,
    PredictionMode,
)
from backend.app.services.anonymization import anonymize_dataset, detect_sensitive_fields
from backend.app.services.csv_parser import parse_examination_csv
from backend.app.services.privacy_exports import build_public_research_dataset, private_subject_mapping_path
from backend.app.utils.files import timestamped_name
from backend.app.utils.json import make_json_safe


def detect_upload(upload_bytes: bytes, filename: str, column_mapping: dict[str, str] | None = None) -> dict[str, Any]:
    df, detected_max = parse_examination_csv(upload_bytes)
    df = _apply_column_mapping(df, column_mapping or {})
    inferred = None
    if "paper_count" in df.columns:
        counts = pd.to_numeric(df["paper_count"], errors="coerce").dropna().astype(int).unique()
        if len(counts) == 1 and int(counts[0]) in {2, 3, 4}:
            inferred = int(counts[0])
    if inferred is None and "subject_code" in df.columns and not df["subject_code"].dropna().empty:
        code = str(df["subject_code"].dropna().iloc[0]).strip()
        inferred = int(code[-1]) if code[-1:] in {"2", "3", "4"} else None
    subjects = _detect_subjects(df, detected_max)
    display_maxima = detected_max if len(subjects) <= 1 else {}
    return {
        "filename": filename,
        "columns": list(df.columns),
        "sensitive_fields": detect_sensitive_fields(df),
        "inferred_paper_count": inferred,
        "detected_max_scores": display_maxima,
        "row_count": len(df),
        "subjects": subjects,
    }


CANONICAL_MAPPING_FIELDS = {
    "subject_id",
    "subject_code",
    "subject_name",
    "candidate_number",
    "candidate_id",
    "anonymized_candidate_id",
    "paper_count",
    *PAPER_SCORE_COLUMNS,
    "p1_max",
    "p2_max",
    "p3_max",
    "p4_max",
}


def export_ada_safe_dataset(
    upload_bytes: bytes,
    filename: str,
    request: ProcessingRequest | None = None,
) -> AdaSafeExportResponse:
    raw_df, detected_max = parse_examination_csv(upload_bytes)
    raw_df = _apply_column_mapping(raw_df, request.column_mapping if request else {})
    sensitive_fields = detect_sensitive_fields(raw_df)
    anonymized = anonymize_dataset(raw_df)
    cleaned = clean_dataset(
        anonymized,
        detected_max_scores=detected_max,
        paper_counts=request.paper_counts if request else [],
        max_scores=request.max_scores if request else [],
        require_complete_scores=True,
    )
    if cleaned.errors:
        raise ValueError("ADA-safe public export requires complete metadata and valid cleaned records: " + "; ".join(cleaned.errors))
    settings = get_settings()
    public_dataset, _ = build_public_research_dataset(cleaned.data, private_subject_mapping_path(settings.project_root))
    export_path = settings.export_dir / timestamped_name("mode_a_ada_safe_public_dataset")
    public_dataset.to_csv(export_path, index=False)
    return AdaSafeExportResponse(
        rows=len(public_dataset),
        export_path=str(export_path),
        download_url=_download_url(export_path),
        sensitive_fields=sensitive_fields,
        columns=list(public_dataset.columns),
    )


def run_pipeline(upload_bytes: bytes, filename: str, request: ProcessingRequest) -> ProcessingResponse:
    raw_df, detected_max = parse_examination_csv(upload_bytes)
    raw_df = _apply_column_mapping(raw_df, request.column_mapping)
    if request.mode == PredictionMode.mode_a:
        return _run_mode_a(raw_df, detected_max, request)
    return _run_mode_b(raw_df, detected_max, request)


def preview_cleaning(upload_bytes: bytes, filename: str, request: ProcessingRequest) -> CleaningPreviewResponse:
    raw_df, detected_max = parse_examination_csv(upload_bytes)
    raw_df = _apply_column_mapping(raw_df, request.column_mapping)
    total_rows = len(raw_df)
    duplicate_rows = int(raw_df.duplicated().sum())
    working = anonymize_dataset(raw_df) if request.mode == PredictionMode.mode_a else raw_df
    cleaned = clean_dataset(
        working,
        detected_max_scores=detected_max,
        paper_counts=request.paper_counts,
        max_scores=request.max_scores,
        require_complete_scores=request.mode == PredictionMode.mode_a,
    )
    reasons = cleaned.invalid_records.get("record_reason", pd.Series(dtype="object")).astype(str)
    incomplete_rows = int(reasons.str.contains("incomplete applicable", case=False, na=False).sum())
    invalid_rows = max(0, len(cleaned.invalid_records) - incomplete_rows)
    predictable_missing_rows = 0
    unpredictable_rows = 0
    if request.mode == PredictionMode.mode_b and not cleaned.data.empty:
        for _, subject_df in _subject_groups(cleaned.data):
            paper_count = int(subject_df["paper_count"].iloc[0])
            active = [f"p{index}_score" for index in range(1, paper_count + 1)]
            missing = subject_df[active].isna().sum(axis=1)
            predictable_missing_rows += int((missing == 1).sum())
            unpredictable_rows += int((missing > 1).sum())
    return CleaningPreviewResponse(
        total_rows=total_rows,
        duplicate_rows=duplicate_rows,
        clean_rows=len(cleaned.data),
        invalid_rows=invalid_rows,
        absent_rows=len(cleaned.absent_records),
        incomplete_rows=incomplete_rows,
        predictable_missing_rows=predictable_missing_rows,
        unpredictable_rows=unpredictable_rows,
        canonical_columns=list(cleaned.data.columns),
        warnings=cleaned.warnings,
        errors=cleaned.errors,
    )


def _apply_column_mapping(data: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    output = data.copy()
    renames: dict[str, str] = {}
    selected_sources: set[str] = set()
    for target, source in mapping.items():
        target = str(target).strip()
        source = str(source).strip()
        if not source:
            continue
        if target not in CANONICAL_MAPPING_FIELDS:
            raise ValueError(f"Unsupported canonical field in column mapping: {target}.")
        if source not in output.columns:
            raise ValueError(f"Mapped source column was not found: {source}.")
        if source in selected_sources:
            raise ValueError(f"Source column is mapped more than once: {source}.")
        if target in output.columns and source != target:
            raise ValueError(f"Cannot map {source} to {target}; the canonical field already exists.")
        selected_sources.add(source)
        if source != target:
            renames[source] = target
    return output.rename(columns=renames)


def _run_mode_a(raw_df: pd.DataFrame, detected_max: dict[str, float | None], request: ProcessingRequest) -> ProcessingResponse:
    settings = get_settings()
    anonymized = anonymize_dataset(raw_df)

    cleaned = clean_dataset(
        anonymized,
        detected_max_scores=detected_max,
        paper_counts=request.paper_counts,
        max_scores=request.max_scores,
        require_complete_scores=True,
    )
    if cleaned.errors:
        return ProcessingResponse(mode=request.mode, rows=len(cleaned.data), warnings=cleaned.warnings, errors=cleaned.errors)

    public_dataset, _ = build_public_research_dataset(cleaned.data, private_subject_mapping_path(settings.project_root))
    public_path = settings.export_dir / timestamped_name("mode_a_ada_safe_public_dataset")
    public_dataset.to_csv(public_path, index=False)
    exports = {"ada_safe_dataset": str(public_path)}

    cleaned_path = settings.export_dir / timestamped_name("mode_a_clean_training_records")
    cleaned.data.to_csv(cleaned_path, index=False)
    exports["clean_training_records"] = str(cleaned_path)
    exports["cleaned_dataset"] = str(cleaned_path)
    invalid_path = _export_records("mode_a_invalid_records", cleaned.invalid_records)
    absent_path = _export_records("mode_a_absent_records", cleaned.absent_records)
    unpredictable_path = _export_records("mode_a_unpredictable_records", pd.DataFrame())
    exports["invalid_records"] = str(invalid_path)
    exports["absent_records"] = str(absent_path)
    exports["unpredictable_records"] = str(unpredictable_path)

    scenarios, metrics, rankings, plots, warnings, model_summary = _train_all_scenarios(cleaned.data)
    warnings.extend(cleaned.warnings)
    metrics_path = settings.export_dir / timestamped_name("mode_a_metrics")
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    exports["metrics"] = str(metrics_path)
    summary_csv_path, summary_json_path = _export_model_summary("mode_a_model_summary", model_summary)
    exports["model_summary_csv"] = str(summary_csv_path)
    exports["model_summary_json"] = str(summary_json_path)
    summary = _executive_summary(
        cleaned.data,
        scenarios,
        rankings,
        exports,
        invalid_count=len(cleaned.invalid_records),
        absent_count=len(cleaned.absent_records),
        unpredictable_count=0,
    )
    return ProcessingResponse(
        mode=request.mode,
        rows=len(cleaned.data),
        exports=exports,
        export_downloads=_download_map(exports),
        metrics=make_json_safe(metrics),
        rankings=make_json_safe(rankings),
        plots=make_json_safe(plots),
        summary=make_json_safe(summary),
        warnings=warnings,
        errors=[],
    )


def _run_mode_b(raw_df: pd.DataFrame, detected_max: dict[str, float | None], request: ProcessingRequest) -> ProcessingResponse:
    settings = get_settings()
    cleaned = clean_dataset(
        raw_df,
        detected_max_scores=detected_max,
        paper_counts=request.paper_counts,
        max_scores=request.max_scores,
        require_complete_scores=False,
    )
    if cleaned.errors:
        return ProcessingResponse(mode=request.mode, rows=len(cleaned.data), warnings=cleaned.warnings, errors=cleaned.errors)

    data = cleaned.data.copy()
    data["prediction_status"] = "original"
    unpredictable_rows: list[pd.DataFrame] = []
    warnings = list(cleaned.warnings)
    metrics: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    scenarios: list[TrainedScenario] = []
    model_summary: list[dict[str, Any]] = []
    clean_training_frames: list[pd.DataFrame] = []
    plots: dict[str, Any] = {"eda": eda_plots(data)}
    single_missing_rows = 0

    for subject_key, subject_df in _subject_groups(data):
        paper_count = int(subject_df["paper_count"].iloc[0])
        active_cols = [f"p{idx}_score" for idx in range(1, paper_count + 1)]
        missing_counts = subject_df[active_cols].isna().sum(axis=1)
        invalid = subject_df[missing_counts > 1]
        if not invalid.empty:
            warnings.append(f"{len(invalid)} row(s) rejected for {subject_key}: insufficient information due to multiple missing papers.")
            rejected = invalid.copy()
            rejected["prediction_status"] = "unpredictable"
            rejected["unpredictable_reason"] = "multiple missing paper scores"
            unpredictable_rows.append(rejected)

        complete_rows = subject_df[missing_counts == 0].copy()
        if not complete_rows.empty:
            clean_training_frames.append(complete_rows)
        if len(complete_rows) < 6:
            entirely_missing_targets = [column for column in active_cols if subject_df[column].isna().all()]
            untrained_missing = subject_df[missing_counts == 1].copy()
            if not untrained_missing.empty:
                single_missing_rows += len(untrained_missing)
                data.loc[untrained_missing.index, "prediction_status"] = "unpredictable"
                untrained_missing["prediction_status"] = "unpredictable"
                untrained_missing["unpredictable_reason"] = "no complete training rows available for target paper"
                unpredictable_rows.append(untrained_missing)
            if entirely_missing_targets:
                warnings.append(
                    "No complete training rows available for this target paper. Provide complete records in the same "
                    "file or use a trained reference dataset in a future version. "
                    f"Subject: {subject_key}; target(s): {', '.join(entirely_missing_targets)}."
                )
            else:
                warnings.append(f"Not enough complete rows to train Mode B models for {subject_key}.")
            continue

        trained: dict[str, TrainedScenario] = {}
        for target in scenario_targets(paper_count):
            X, y = build_features_for_target(complete_rows, target)
            try:
                trained[target] = train_scenario(subject_key, paper_count, target, X, y)
                scenarios.append(trained[target])
                metrics.extend(trained[target].metrics)
                rankings.extend(trained[target].ranking)
                model_summary.append(_scenario_summary(trained[target]))
                _add_scenario_explainability(plots, trained[target], X)
            except ValueError as exc:
                warnings.append(str(exc))

        valid_missing = subject_df[missing_counts == 1]
        single_missing_rows += len(valid_missing)
        for idx, row in valid_missing.iterrows():
            target = next(col for col in active_cols if pd.isna(row[col]))
            scenario = trained.get(target)
            if scenario is None:
                warnings.append(f"No trained model available for missing {target} in {subject_key}.")
                rejected = pd.DataFrame([row.to_dict()])
                rejected["prediction_status"] = "unpredictable"
                rejected["unpredictable_reason"] = f"no trained model available for {target}"
                unpredictable_rows.append(rejected)
                continue
            prepared = _prepare_single_prediction(row, scenario.feature_columns)
            prediction = float(scenario.best_model.predict(prepared)[0])
            data.loc[idx, target] = prediction
            data.loc[idx, "prediction_status"] = "predicted"

    if single_missing_rows == 0 and len(data) > 0:
        warnings.append(
            "No predictable missing scores found. Dataset contains complete valid records only. "
            "Use Mode A Benchmark to evaluate predictive performance."
        )

    absent_output = cleaned.absent_records.copy()
    if not absent_output.empty:
        absent_output["prediction_status"] = "absent"
    predictable_output = data[data["prediction_status"] != "unpredictable"].copy()
    output_path = settings.export_dir / timestamped_name("mode_b_completed_predictions")
    predictable_output.to_csv(output_path, index=False)
    exports = {"completed_prediction_file": str(output_path)}
    clean_training = pd.concat(clean_training_frames, ignore_index=True, sort=False) if clean_training_frames else pd.DataFrame()
    clean_training_path = _export_records("mode_b_clean_training_records", clean_training)
    invalid_path = _export_records("mode_b_invalid_records", cleaned.invalid_records)
    absent_path = _export_records("mode_b_absent_records", absent_output)
    exports["clean_training_records"] = str(clean_training_path)
    exports["invalid_records"] = str(invalid_path)
    exports["absent_records"] = str(absent_path)
    if unpredictable_rows:
        reference = pd.concat(unpredictable_rows, ignore_index=True, sort=False)
        reference_path = _export_records("mode_b_unpredictable_records", reference)
    else:
        reference_path = _export_records("mode_b_unpredictable_records", pd.DataFrame())
    exports["unpredictable_records"] = str(reference_path)
    exports["unpredictable_reference_file"] = str(reference_path)
    summary_csv_path, summary_json_path = _export_model_summary("mode_b_model_summary", model_summary)
    exports["model_summary_csv"] = str(summary_csv_path)
    exports["model_summary_json"] = str(summary_json_path)
    unpredictable_count = sum(len(frame) for frame in unpredictable_rows)
    summary = _executive_summary(
        data,
        scenarios,
        rankings,
        exports,
        invalid_count=len(cleaned.invalid_records),
        absent_count=len(absent_output),
        unpredictable_count=unpredictable_count,
    )
    return ProcessingResponse(
        mode=request.mode,
        rows=len(predictable_output),
        exports=exports,
        export_downloads=_download_map(exports),
        metrics=make_json_safe(metrics),
        rankings=make_json_safe(rankings),
        plots=make_json_safe(plots),
        summary=make_json_safe(summary),
        warnings=warnings,
        errors=[],
    )


def _train_all_scenarios(data: pd.DataFrame) -> tuple[list[TrainedScenario], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str], list[dict[str, Any]]]:
    scenarios: list[TrainedScenario] = []
    metrics: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    warnings: list[str] = []
    model_summary: list[dict[str, Any]] = []
    plots: dict[str, Any] = {"eda": eda_plots(data)}

    for subject_key, subject_df in _subject_groups(data):
        paper_count = int(subject_df["paper_count"].iloc[0])
        for target in scenario_targets(paper_count):
            X, y = build_features_for_target(subject_df, target)
            try:
                scenario = train_scenario(subject_key, paper_count, target, X, y)
            except ValueError as exc:
                warnings.append(str(exc))
                continue
            scenarios.append(scenario)
            metrics.extend(scenario.metrics)
            rankings.extend(scenario.ranking)
            model_summary.append(_scenario_summary(scenario))
            _add_scenario_explainability(plots, scenario, X)
            if "actual_vs_predicted" not in plots:
                plots["actual_vs_predicted"] = actual_vs_predicted(scenario.y_test, scenario.y_pred)
                plots["residual_plot"] = residual_plot(scenario.y_test, scenario.y_pred)
                plots["feature_importance"] = feature_importance(scenario.best_model, scenario.feature_columns)
                plots["shap"] = shap_summary(scenario.best_model, X[scenario.feature_columns])
                plots["partial_dependence"] = partial_dependence_plot(scenario.best_model, X[scenario.feature_columns])
    return scenarios, metrics, rankings, plots, warnings, model_summary


def _detect_subjects(df: pd.DataFrame, detected_max: dict[str, float | None]) -> list[dict[str, Any]]:
    if "subject_code" in df.columns and df["subject_code"].notna().any():
        group_cols = ["subject_code"]
    elif "subject_id" in df.columns and df["subject_id"].notna().any():
        group_cols = ["subject_id"]
    elif "subject_name" in df.columns:
        group_cols = ["subject_name"]
    else:
        return [
            {
                "subject_key": "dataset",
                "subject_id": None,
                "subject_code": None,
                "subject_name": None,
                "inferred_paper_count": None,
                "row_count": len(df),
                "detected_max_scores": detected_max,
            }
        ]
    subjects: list[dict[str, Any]] = []
    for key, group in df.groupby(group_cols[0], dropna=False):
        subject_code = str(key).strip() if group_cols[0] == "subject_code" and pd.notna(key) else None
        subject_id = str(key).strip() if group_cols[0] == "subject_id" and pd.notna(key) else None
        subject_name = str(key).strip() if group_cols[0] == "subject_name" and pd.notna(key) else None
        if "subject_name" in group.columns and subject_name is None and group["subject_name"].notna().any():
            subject_name = str(group["subject_name"].dropna().iloc[0]).strip()
        inferred = None
        if "paper_count" in group.columns:
            counts = pd.to_numeric(group["paper_count"], errors="coerce").dropna().astype(int).unique()
            if len(counts) == 1 and int(counts[0]) in {2, 3, 4}:
                inferred = int(counts[0])
        if inferred is None and subject_code and subject_code[-1:] in {"2", "3", "4"}:
            inferred = int(subject_code[-1])
        subject_maxima = _detected_maxima_for_group(group, detected_max)
        subjects.append(
            {
                "subject_key": subject_code or subject_id or subject_name or "dataset",
                "subject_id": subject_id,
                "subject_code": subject_code,
                "subject_name": subject_name,
                "inferred_paper_count": inferred,
                "row_count": len(group),
                "detected_max_scores": subject_maxima,
            }
        )
    return subjects


def _scenario_key(scenario: TrainedScenario) -> str:
    target = scenario.target.replace("_score", "").upper()
    return f"{scenario.subject_key} | Hide {target}"


def _scenario_summary(scenario: TrainedScenario) -> dict[str, Any]:
    best = next((row for row in scenario.ranking if row["model"] == scenario.best_model_name), scenario.ranking[0])
    return {
        "subject": scenario.subject_key,
        "paper_count": scenario.paper_count,
        "scenario": f"Hide {scenario.target.replace('_score', '').upper()}",
        "target": scenario.target,
        "best_model": scenario.best_model_name,
        "RMSE": best.get("RMSE"),
        "MAE": best.get("MAE"),
        "MSE": best.get("MSE"),
        "R2": best.get("R2"),
        "CV_RMSE": best.get("CV_RMSE"),
        "feature_columns": ", ".join(scenario.feature_columns),
    }


def _add_scenario_explainability(plots: dict[str, Any], scenario: TrainedScenario, X: pd.DataFrame) -> None:
    explainability = plots.setdefault("scenario_explainability", {})
    key = _scenario_key(scenario)
    explainability[key] = {
        "feature_importance": feature_importance(scenario.best_model, scenario.feature_columns),
        "shap": shap_summary(scenario.best_model, X[scenario.feature_columns]),
    }


def _export_model_summary(prefix: str, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    settings = get_settings()
    csv_path = settings.export_dir / timestamped_name(prefix, ".csv")
    json_path = settings.export_dir / timestamped_name(prefix, ".json")
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    json_path.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    return csv_path, json_path


def _export_records(prefix: str, frame: pd.DataFrame) -> Path:
    path = get_settings().export_dir / timestamped_name(prefix, ".csv")
    frame.to_csv(path, index=False)
    return path


def _download_url(path: Path | str) -> str:
    return f"/api/download/{Path(path).name}"


def _download_map(exports: dict[str, str]) -> dict[str, str]:
    return {key: _download_url(value) for key, value in exports.items()}


def _executive_summary(
    data: pd.DataFrame,
    scenarios: list[TrainedScenario],
    rankings: list[dict[str, Any]],
    exports: dict[str, str],
    invalid_count: int = 0,
    absent_count: int = 0,
    unpredictable_count: int = 0,
) -> dict[str, Any]:
    best = sorted(rankings, key=lambda item: (item.get("RMSE", float("inf")), item.get("MAE", float("inf"))))[0] if rankings else {}
    return {
        "total_rows": int(len(data)),
        "subjects_detected": int(_subject_identity_series(data).nunique()) if len(data) else 0,
        "scenarios_run": len(scenarios),
        "best_overall_model": best.get("model"),
        "best_rmse": best.get("RMSE"),
        "export_files_available": len(exports),
        "clean_records": int(len(data)),
        "invalid_records": int(invalid_count),
        "absent_records": int(absent_count),
        "unpredictable_records": int(unpredictable_count),
    }


def _subject_groups(data: pd.DataFrame):
    if data["subject_code"].notna().any():
        key_col = "subject_code"
    elif "subject_id" in data.columns and data["subject_id"].notna().any():
        key_col = "subject_id"
    else:
        key_col = "subject_name"
    for key, group in data.groupby(key_col, dropna=False):
        yield str(key), group.copy()


def _subject_identity_series(data: pd.DataFrame) -> pd.Series:
    identity = data["subject_code"].copy()
    if "subject_id" in data.columns:
        identity = identity.fillna(data["subject_id"])
    return identity.fillna(data["subject_name"])


def _detected_maxima_for_group(group: pd.DataFrame, fallback: dict[str, float | None]) -> dict[str, float | None]:
    maxima: dict[str, float | None] = {}
    for max_col in ("p1_max", "p2_max", "p3_max", "p4_max"):
        if max_col in group.columns:
            values = pd.to_numeric(group[max_col], errors="coerce").dropna().unique()
            if len(values) == 1:
                maxima[max_col] = float(values[0])
            continue
        if fallback.get(max_col) is not None:
            maxima[max_col] = fallback[max_col]
    return maxima


def _prepare_single_prediction(row: pd.Series, feature_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame([row.to_dict()])
    missing = [column for column in feature_columns if column not in frame.columns]
    for column in missing:
        frame[column] = 0.0
    return frame[feature_columns].fillna(0.0)
