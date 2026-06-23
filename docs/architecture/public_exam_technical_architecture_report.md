# Public Examination Score Prediction System: Technical Architecture Report

## Purpose

This document replaces the earlier internal system specification with a public-safe technical architecture report. It describes the implemented application without referencing restricted organization names, confidential source documents, raw candidate identifiers, or private subject mappings.

The project is a capstone-grade machine learning web application for predicting missing examination component scores and benchmarking regression models under controlled experimental conditions.

## System Objective

The application supports two workflows:

- **Mode A: Experimental Benchmarking**  
  Complete valid examination records are used for controlled hide-and-predict experiments. One paper score is hidden at a time, models are trained using the remaining applicable scores, and predictions are compared against the known hidden score.

- **Mode B Lite: Missing-Score Prediction**  
  Uploaded records with one row-level missing paper score are processed for prediction using complete valid records in the same uploaded file. The mode is intentionally limited to one missing applicable paper per candidate.

## Architecture

The system follows the approved research-first architecture:

- Frontend: React and TypeScript
- Backend: FastAPI
- Machine learning engine: Python scientific stack
- Visualization: Plotly
- Explainability: SHAP and feature importance
- Deployment: Docker-compatible FastAPI service serving built frontend assets

The implementation is organized into separate frontend, backend, data, reports, and documentation folders. Backend services separate API routing, schema validation, preprocessing, training, evaluation, explainability, exports, and utility logic.

## Data Workflow

The application starts with CSV upload and dataset detection. It identifies columns, subject grouping fields, paper counts, score columns, maximum score metadata, and sensitive identifiers where present.

If the file uses the public schema, the system can work directly with:

- `subject_id`
- `paper_count`
- anonymized candidate identifier
- `p1_score` to `p4_score`
- `p1_max` to `p4_max`

If a raw dataset uses different column names, the user can configure column mapping before detection, preview cleaning, or pipeline execution.

## Privacy Model

Public exports do not expose original candidate identifiers or original subject codes. Candidate identifiers are anonymized, and subjects are represented with stable public subject IDs such as `SUBJ_001`.

The private mapping between original subject metadata and public subject IDs is kept in an ignored confidential folder and is not included in the public repository or submission package.

## Metadata Rules

Paper maximum scores are required metadata. The application does not assume default maxima. When maxima are missing, the user must enter them before cleaning, training, benchmarking, prediction, or public export.

Paper applicability is determined by the subject paper count:

- 2-paper subjects require P1 and P2.
- 3-paper subjects require P1, P2, and P3.
- 4-paper subjects require P1, P2, P3, and P4.

Non-applicable paper columns are ignored for validation and modelling.

## Cleaning Rules

The cleaning workflow classifies records into evidence categories rather than silently discarding them. Outputs include clean training records, invalid records, absent records, incomplete records where applicable, and unpredictable records.

Training and benchmarking use complete valid records for applicable papers only. Invalid, absent, and insufficient records are isolated into export files for review.

## Feature Engineering

The approved feature set includes normalized scores, partial totals, mean score, score spread, and score standard deviation. For Mode A benchmark scenarios, engineered aggregate features are recalculated after hiding the target paper so the hidden score cannot leak into the model through derived totals or averages.

## Model Evaluation

The system compares the approved regression models:

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- CatBoost Regressor
- Support Vector Regressor

Evaluation metrics include MAE, MSE, RMSE, and R-squared. Ranking tables and model summaries are exported for report writing.

## Explainability and Dashboard

The dashboard provides executive summary cards, dataset quality summaries, model ranking tables, evaluation metrics, export links, and selectable chart/explainability panels. SHAP and feature importance outputs are generated for selected best-model scenarios where supported by the model and data.

## Deployment

The application can be run locally with Python and Node, or as a Docker container. The Docker image builds the frontend, installs backend dependencies, exposes the FastAPI service, and serves the browser application from the same container.

Local Docker usage:

```powershell
cd "C:\ADA_Data_Science\Projects\Capstone Project"
docker build -t ada-score-prediction:latest .
docker run --rm -p 8000:8000 ada-score-prediction:latest
```

If port `8000` is already in use:

```powershell
docker run --rm -p 8001:8000 ada-score-prediction:latest
```

Then open `http://localhost:8001`.

## Reproducibility

The repository includes Python dependencies, frontend build configuration, Docker deployment files, tests, documentation, and notebook/report artifacts. Confidential source data and private mappings are excluded from Git tracking.

The project is suitable for capstone submission, reproducible local demonstration, Docker deployment, and final report evidence generation.
