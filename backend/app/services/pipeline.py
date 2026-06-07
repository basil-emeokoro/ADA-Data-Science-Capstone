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
from backend.app.schemas.pipeline import AdaSafeExportResponse, ProcessingRequest, ProcessingResponse, PredictionMode
from backend.app.services.anonymization import anonymize_dataset, detect_sensitive_fields
from backend.app.services.csv_parser import parse_examination_csv
from backend.app.utils.files import timestamped_name
from backend.app.utils.json import make_json_safe


def detect_upload(upload_bytes: bytes, filename: str) -> dict[str, Any]:
    df, detected_max = parse_examination_csv(upload_bytes)
    inferred = None
    if "subject_code" in df.columns and not df["subject_code"].dropna().empty:
        code = str(df["subject_code"].dropna().iloc[0]).strip()
        inferred = int(code[-1]) if code[-1:] in {"2", "3", "4"} else None
    return {
        "filename": filename,
        "columns": list(df.columns),
        "sensitive_fields": detect_sensitive_fields(df),
        "inferred_paper_count": inferred,
        "detected_max_scores": detected_max,
        "row_count": len(df),
        "subjects": _detect_subjects(df, detected_max),
    }


def export_ada_safe_dataset(upload_bytes: bytes, filename: str) -> AdaSafeExportResponse:
    raw_df, _ = parse_examination_csv(upload_bytes)
    sensitive_fields = detect_sensitive_fields(raw_df)
    anonymized = anonymize_dataset(raw_df)
    export_path = get_settings().export_dir / timestamped_name("mode_a_ada_safe_preclean")
    anonymized.to_csv(export_path, index=False)
    return AdaSafeExportResponse(
        rows=len(anonymized),
        export_path=str(export_path),
        sensitive_fields=sensitive_fields,
        columns=list(anonymized.columns),
    )


def run_pipeline(upload_bytes: bytes, filename: str, request: ProcessingRequest) -> ProcessingResponse:
    raw_df, detected_max = parse_examination_csv(upload_bytes)
    if request.mode == PredictionMode.mode_a:
        return _run_mode_a(raw_df, detected_max, request)
    return _run_mode_b(raw_df, detected_max, request)


def _run_mode_a(raw_df: pd.DataFrame, detected_max: dict[str, float | None], request: ProcessingRequest) -> ProcessingResponse:
    settings = get_settings()
    anonymized = anonymize_dataset(raw_df)
    anonymized_path = settings.export_dir / timestamped_name("mode_a_ada_safe_anonymized")
    anonymized.to_csv(anonymized_path, index=False)

    cleaned = clean_dataset(
        anonymized,
        detected_max_scores=detected_max,
        paper_counts=request.paper_counts,
        max_scores=request.max_scores,
        require_complete_scores=True,
    )
    exports = {"ada_safe_dataset": str(anonymized_path)}
    if cleaned.errors:
        return ProcessingResponse(mode=request.mode, rows=len(cleaned.data), exports=exports, warnings=cleaned.warnings, errors=cleaned.errors)

    cleaned_path = settings.export_dir / timestamped_name("mode_a_cleaned")
    cleaned.data.to_csv(cleaned_path, index=False)
    exports["cleaned_dataset"] = str(cleaned_path)

    scenarios, metrics, rankings, plots, warnings, model_summary = _train_all_scenarios(cleaned.data)
    warnings.extend(cleaned.warnings)
    metrics_path = settings.export_dir / timestamped_name("mode_a_metrics")
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    exports["metrics"] = str(metrics_path)
    summary_csv_path, summary_json_path = _export_model_summary("mode_a_model_summary", model_summary)
    exports["model_summary_csv"] = str(summary_csv_path)
    exports["model_summary_json"] = str(summary_json_path)
    summary = _executive_summary(cleaned.data, scenarios, rankings, exports)
    return ProcessingResponse(
        mode=request.mode,
        rows=len(cleaned.data),
        exports=exports,
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
    plots: dict[str, Any] = {"eda": eda_plots(data)}

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
        if len(complete_rows) < 6:
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

    predictable_output = data[data["prediction_status"] != "unpredictable"].copy()
    output_path = settings.export_dir / timestamped_name("mode_b_completed_predictions")
    predictable_output.to_csv(output_path, index=False)
    exports = {"completed_prediction_file": str(output_path)}
    if unpredictable_rows:
        reference = pd.concat(unpredictable_rows, ignore_index=True, sort=False)
        reference_path = settings.export_dir / timestamped_name("mode_b_unpredictable_reference")
        reference.to_csv(reference_path, index=False)
        exports["unpredictable_reference_file"] = str(reference_path)
    summary_csv_path, summary_json_path = _export_model_summary("mode_b_model_summary", model_summary)
    exports["model_summary_csv"] = str(summary_csv_path)
    exports["model_summary_json"] = str(summary_json_path)
    summary = _executive_summary(data, scenarios, rankings, exports)
    return ProcessingResponse(
        mode=request.mode,
        rows=len(predictable_output),
        exports=exports,
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
    elif "subject_name" in df.columns:
        group_cols = ["subject_name"]
    else:
        return [
            {
                "subject_key": "dataset",
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
        subject_name = str(key).strip() if group_cols[0] == "subject_name" and pd.notna(key) else None
        if "subject_name" in group.columns and subject_name is None and group["subject_name"].notna().any():
            subject_name = str(group["subject_name"].dropna().iloc[0]).strip()
        inferred = None
        if subject_code and subject_code[-1:] in {"2", "3", "4"}:
            inferred = int(subject_code[-1])
        subjects.append(
            {
                "subject_key": subject_code or subject_name or "dataset",
                "subject_code": subject_code,
                "subject_name": subject_name,
                "inferred_paper_count": inferred,
                "row_count": len(group),
                "detected_max_scores": detected_max,
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


def _executive_summary(
    data: pd.DataFrame,
    scenarios: list[TrainedScenario],
    rankings: list[dict[str, Any]],
    exports: dict[str, str],
) -> dict[str, Any]:
    best = sorted(rankings, key=lambda item: (item.get("RMSE", float("inf")), item.get("MAE", float("inf"))))[0] if rankings else {}
    return {
        "total_rows": int(len(data)),
        "subjects_detected": int(data["subject_code"].fillna(data["subject_name"]).nunique()) if len(data) else 0,
        "scenarios_run": len(scenarios),
        "best_overall_model": best.get("model"),
        "best_rmse": best.get("RMSE"),
        "export_files_available": len(exports),
    }


def _subject_groups(data: pd.DataFrame):
    key_col = "subject_code" if data["subject_code"].notna().any() else "subject_name"
    for key, group in data.groupby(key_col, dropna=False):
        yield str(key), group.copy()


def _prepare_single_prediction(row: pd.Series, feature_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame([row.to_dict()])
    missing = [column for column in feature_columns if column not in frame.columns]
    for column in missing:
        frame[column] = 0.0
    return frame[feature_columns].fillna(0.0)
