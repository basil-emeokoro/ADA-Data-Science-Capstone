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
from backend.app.schemas.pipeline import ProcessingRequest, ProcessingResponse, PredictionMode
from backend.app.services.anonymization import anonymize_dataset, detect_sensitive_fields
from backend.app.services.csv_parser import parse_examination_csv
from backend.app.utils.files import timestamped_name


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
    }


def run_pipeline(upload_bytes: bytes, filename: str, request: ProcessingRequest) -> ProcessingResponse:
    raw_df, detected_max = parse_examination_csv(upload_bytes)
    if request.mode == PredictionMode.mode_a:
        return _run_mode_a(raw_df, detected_max, request)
    return _run_mode_b(raw_df, detected_max, request)


def _run_mode_a(raw_df: pd.DataFrame, detected_max: dict[str, float | None], request: ProcessingRequest) -> ProcessingResponse:
    settings = get_settings()
    anonymized = anonymize_dataset(raw_df)
    anonymized_path = settings.export_dir / timestamped_name("ada_safe_anonymized")
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

    scenarios, metrics, rankings, plots, warnings = _train_all_scenarios(cleaned.data)
    warnings.extend(cleaned.warnings)
    metrics_path = settings.export_dir / timestamped_name("mode_a_metrics")
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    exports["metrics"] = str(metrics_path)
    return ProcessingResponse(
        mode=request.mode,
        rows=len(cleaned.data),
        exports=exports,
        metrics=metrics,
        rankings=rankings,
        plots=plots,
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
                metrics.extend(trained[target].metrics)
                rankings.extend(trained[target].ranking)
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
    return ProcessingResponse(
        mode=request.mode,
        rows=len(predictable_output),
        exports=exports,
        metrics=metrics,
        rankings=rankings,
        plots=plots,
        warnings=warnings,
        errors=[],
    )


def _train_all_scenarios(data: pd.DataFrame) -> tuple[list[TrainedScenario], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    scenarios: list[TrainedScenario] = []
    metrics: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    warnings: list[str] = []
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
            if "actual_vs_predicted" not in plots:
                plots["actual_vs_predicted"] = actual_vs_predicted(scenario.y_test, scenario.y_pred)
                plots["residual_plot"] = residual_plot(scenario.y_test, scenario.y_pred)
                plots["feature_importance"] = feature_importance(scenario.best_model, scenario.feature_columns)
                plots["shap"] = shap_summary(scenario.best_model, X[scenario.feature_columns])
                plots["partial_dependence"] = partial_dependence_plot(scenario.best_model, X[scenario.feature_columns])
    return scenarios, metrics, rankings, plots, warnings


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
