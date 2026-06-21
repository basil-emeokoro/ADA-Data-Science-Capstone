# Predicting Missing Examination Component Scores

Capstone-grade machine learning web application for predicting missing examination paper component scores in high-stakes educational assessment.

The project implements the frozen specification:

- Mode A: privacy-preserving experimental benchmarking with anonymization, deliberate score hiding, model comparison, explainability, and exports.
- Mode B Lite: real missing paper prediction with one missing paper per candidate, dynamic model selection, and completed dataset export.

## Architecture

- Frontend: React + TypeScript
- Backend: FastAPI
- ML Engine: pandas, numpy, scikit-learn, XGBoost, CatBoost, SHAP, Plotly
- Data layout: raw, anonymized, cleaned, processed, and export directories

## Runtime

Use Python 3.12 for the backend and machine learning environment.

Python 3.12 is the supported runtime because it has broad wheel compatibility across the scientific Python stack used here, especially CatBoost, SHAP, XGBoost, scikit-learn, pandas, and NumPy.

## Installation

```powershell
cd "C:\ADA_Data_Science\Projects\Capstone Project"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running Backend

```powershell
cd "C:\ADA_Data_Science\Projects\Capstone Project"
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8002
```

The API is available at `http://127.0.0.1:8002`.

## Running Frontend

```powershell
cd "C:\ADA_Data_Science\Projects\Capstone Project\frontend"
npm.cmd install
npm.cmd run dev -- --host 127.0.0.1 --port 5175
```

Open the local URL printed by Vite, usually `http://127.0.0.1:5175`.

The frontend calls the FastAPI backend through same-origin `/api` routes during local development.

For local development, the Vite dev server proxies `/api` requests to `http://127.0.0.1:8002`, so no browser CORS configuration is needed when both servers are running locally.

For deployment, set `VITE_API_BASE_URL` to the public backend URL.

## Docker Deployment

The repository includes a Dockerfile for local deployment. The image builds the React frontend, installs the Python 3.12 backend dependencies, and serves the FastAPI app on port `8000`.

```powershell
cd "C:\ADA_Data_Science\Projects\Capstone Project"
docker build -t ada-score-prediction:latest .
docker run --rm -p 8000:8000 ada-score-prediction:latest
```

Open `http://localhost:8000` for the frontend or `http://localhost:8000/api/health` for the API health check.

To stop a foreground Docker container, press `Ctrl+C` in the terminal where it is running. If the container was started in detached mode, list containers with `docker ps` and stop it with `docker stop <container_name_or_id>`.

The `.dockerignore` file excludes confidential inputs, `Working Documents/`, generated reports, exported CSVs, virtual environments, and frontend build folders from the Docker build context.

## Verification Commands

Run these from the project root after activating `.venv`:

```powershell
python -m pytest backend/tests -q
python -m pip check
cd frontend
npm.cmd run build
```

## Upload Workflow

1. Upload a CSV examination dataset.
2. Review detected sensitive fields.
3. Click `Detect` to inspect columns, subjects, paper counts, sensitive fields, and maxima.
4. Recover missing metadata by subject batch where needed. If paper maxima are absent from the CSV, enter the required maxima manually before running any pipeline.
5. For Mode A, click `Export ADA-Safe Dataset` to export the public privacy-preserving dataset when required.
6. Run Mode A benchmarking or Mode B Lite prediction.

## Privacy Model

The project separates public research artifacts from private examination metadata.

Candidate identifiers are replaced with synthetic IDs such as `CAND_000001`. Candidate number, centre number, candidate ID, and other direct identifiers are removed from public outputs.

Subject codes are pseudonymized in public/ADA-safe exports. Original values such as `512003` are replaced with stable public IDs such as `SUBJ_001`. The public dataset contains `subject_id`, not `subject_code`, and does not include original subject names.

The private mapping file is generated at:

```text
Working Documents/subject_mapping_private.csv
```

It contains:

- `original_subject_code`
- `subject_id`
- `subject_name`
- `paper_count`

`Working Documents/` is excluded by `.gitignore`, so the private mapping is not committed or included in ADA-safe downloads.

Paper maxima are included in the public dataset because they are required metadata for reproducibility. Without maxima, normalized score features and score-validation decisions cannot be independently reproduced. Non-applicable paper maxima remain blank.

Public ADA-safe datasets include only:

- `subject_id`
- `paper_count`
- `anonymized_candidate_id`
- `p1_score`, `p2_score`, `p3_score`, `p4_score`
- `p1_max`, `p2_max`, `p3_max`, `p4_max`

## WAEC Paper Rules

Paper applicability is inferred from the final digit of `subject_code`:

- Codes ending in `2`: P1 and P2 are applicable. P3 and P4 are ignored completely.
- Codes ending in `3`: P1, P2, and P3 are applicable. P4 is ignored completely.
- Codes ending in `4`: P1, P2, P3, and P4 are applicable.

Invalid values in non-applicable papers do not invalidate a record. For example, `-99` in P3/P4 for a 2-paper subject is ignored.

Paper maximum scores are required metadata. The application never assumes a default maximum score. If maxima are missing from the uploaded CSV, processing stops until the user supplies maxima for each applicable paper:

- Codes ending in `2`: provide P1 max and P2 max only.
- Codes ending in `3`: provide P1 max, P2 max, and P3 max only.
- Codes ending in `4`: provide P1 max, P2 max, P3 max, and P4 max.

Do not provide maxima for non-applicable papers. If a user manually enters `100`, that is treated as user-supplied metadata, not a system assumption.

Applicable-paper values of `-99`, `B`, `null`, or blank are isolated into `invalid_records.csv`.

Applicable-paper values of `A`, `AB`, or `ABS` are treated as absent:

- Mode A excludes absent candidates from training.
- Mode B does not predict absent papers and exports those rows with `prediction_status = absent`.

## Mode A Usage

Mode A requires anonymization. The system:

1. Detects identifiers such as candidate and centre fields.
2. Replaces candidate identifiers with synthetic IDs such as `CAND_000001`.
3. Replaces confidential subject codes with stable public IDs such as `SUBJ_001` in ADA-safe exports.
4. Writes the private subject-code mapping only to `Working Documents/subject_mapping_private.csv`.
5. Supports an explicit `Export ADA-Safe Dataset` action.
6. Exports the cleaned public ADA-safe dataset again during Mode A processing for reproducibility.
7. Cleans and validates score data, removing incomplete active-paper records before benchmarking.
8. Generates 2-, 3-, and 4-paper scenarios by hiding each available paper component.
9. Trains Random Forest, Gradient Boosting, XGBoost, CatBoost, and SVR models.
10. Evaluates with MAE, MSE, RMSE, R2, 80/20 split, and 5-fold cross validation.
11. Produces ranking tables, model summaries, and Plotly explainability visuals for best scenario models.

Mode A is the benchmark/research mode. It requires complete valid records because the system deliberately hides known paper scores and compares predictions against the real values. It is used for model comparison, research evidence, reproducibility, and report figures. It is not primarily the routine operational prediction workflow.

## Mode B Usage

Mode B Lite does not require anonymization. The system:

1. Cleans uploaded data.
2. Detects candidates with exactly one missing paper score.
3. Rejects candidates with multiple missing papers.
4. Trains scenario models from complete rows for the same subject.
5. Selects the correct model for each missing paper.
6. Exports completed predictions with `prediction_status`.
7. Exports unpredictable cases separately for reference.

Mode B is the real missing-score prediction workflow. It is used when the uploaded dataset already contains genuine missing paper scores. It trains from complete valid rows in the same loaded prediction dataset for the same subject/scenario, then predicts only one missing applicable paper per candidate. It does not predict absent candidates.

## Training and Validation Strategy

The modelling pipeline is deterministic and reproducible:

- Train/test split is used for every trained scenario.
- The split ratio is 80 percent training and 20 percent testing for datasets with enough rows.
- 5-fold cross-validation is also used on the training partition where possible.
- `random_state = 42` is fixed in the backend settings.
- Mode A trains benchmark models from the uploaded benchmark dataset after cleaning and experimental hiding.
- Mode B trains from complete valid rows inside the loaded prediction dataset, not from a separate external training file. Rows with one genuine missing applicable paper are then predicted using the matching trained scenario model.

## Codebase Map

- `frontend/` - React/Vite user interface.
- `backend/` - FastAPI backend.
- `backend/app/services/` - upload handling, CSV parsing, privacy export, and pipeline orchestration.
- `backend/app/ml/preprocessing/` - WAEC cleaning rules, metadata recovery support, and feature engineering.
- `backend/app/ml/models/` - model registry and model definitions.
- `backend/app/ml/evaluation/` - regression metrics and ranking logic.
- `backend/app/ml/explainability/` - SHAP, feature importance, residual, actual-vs-predicted, and Plotly chart helpers.
- `backend/notebooks/` - reproducible demo notebook.
- `data/` - runtime data and generated exports; not the public submission source.
- `Working Documents/` - confidential/private workspace; never public.
- `ADA_Data_Science_Submission_Basil_Emeokoro/` - generated clean submission package when prepared locally.

## Deployment Guidance

### Render

Use the Docker deployment path for Render:

1. Push the repository to GitHub without confidential files.
2. Create a Render Web Service.
3. Select Docker as the environment.
4. Use the repository root as the build context.
5. Expose port `8000`.
6. Health check path: `/api/health`.

Equivalent local validation commands:

```powershell
docker build -t ada-score-prediction .
docker run --rm -p 8000:8000 ada-score-prediction
```

### GitHub

GitHub hosts the source code and documentation, but it does not directly run a FastAPI + React application as a live web app without deployment infrastructure. Use GitHub together with Render, a VM, Docker hosting, or Codespaces for execution.

### GitHub Codespaces

Codespaces can run the app for demonstration:

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Forward the backend and frontend ports from the Codespaces Ports tab.

### Streamlit Community Cloud

This is not a Streamlit app. It uses FastAPI and React, so Streamlit Community Cloud is not the correct deployment target unless the UI is rewritten in Streamlit.

## Export Process

Exports are written to `data/exports/` and include:

- Mode A ADA-safe public dataset CSV with candidate anonymization, subject pseudonymization, paper counts, cleaned scores, and maxima
- Mode A clean training records CSV
- Mode A metrics CSV
- Mode A model summary CSV and JSON
- Mode A invalid records CSV
- Mode A absent records CSV
- Mode A unpredictable records CSV
- Mode B completed prediction CSV
- Mode B clean training records CSV
- Mode B invalid records CSV
- Mode B absent records CSV
- Mode B unpredictable records CSV
- Mode B model summary CSV and JSON

CSV exports are intentionally ignored by git because source and generated datasets may contain confidential records.

After a pipeline run, each export appears in the frontend export package with a `Download` button. Downloads are served by the backend from `/api/download/{filename}` and are restricted to files generated inside `data/exports/`.

Use these downloads for report evidence:

- ADA-safe public dataset for submission-safe review, publication, demonstrations, and screenshots
- clean training records used for modelling
- invalid records isolated from applicable-paper invalid values
- absent records isolated from true absences
- unpredictable records that could not be predicted
- metrics CSV and model summary CSV/JSON
- Mode B completed prediction file

The private subject mapping file is not exposed through the download API and should remain only in `Working Documents/`.

## Notebook

The publication/demo notebook is:

```text
backend/notebooks/Exam_Score_Prediction_Demo.ipynb
```

It demonstrates dataset loading, detection, metadata recovery, privacy-preserving preprocessing, cleaning, EDA, feature engineering, Mode A benchmarking, Mode B controlled validation, model comparison, explainability, exports, and summary reporting.

## Submission Package

For ADA/GitHub submission, use the clean generated package:

```text
ADA_Data_Science_Submission_Basil_Emeokoro/
```

This folder excludes confidential raw data, `Working Documents/`, private subject mappings, `.venv`, logs, caches, and old development exports. Its public sample dataset is:

```text
sample_data/ADA_Public_Dataset.csv
```

That dataset contains pseudonymized `subject_id` values, anonymized candidate IDs, paper counts, cleaned scores, and TASS/marks-distribution maxima for all applicable papers.

## Troubleshooting

- If CatBoost or XGBoost installation fails, install Microsoft Visual C++ Build Tools and retry.
- If `py -3.12` is not found on Windows, install Python 3.12 and select "Add python.exe to PATH" during installation.
- If the frontend cannot reach the backend, confirm Uvicorn is running on `http://127.0.0.1:8002` and restart the Vite dev server so proxy settings reload.
- If Vite reports the requested port is in use, use the alternate URL Vite prints in the terminal.
- If SHAP plots are slow, run on a smaller sample during development.
- If score validation fails, check that paper scores do not exceed maximum scores.
- If paper count cannot be inferred from subject code, provide it by subject name in the metadata recovery panel.
- If maximum scores are missing, provide max scores per applicable paper in the metadata recovery panel. The app does not default missing maxima to `100` or any other value.

## Reproducibility Notes

- The training pipeline uses fixed random seeds.
- The supported Python runtime is pinned to 3.12 in `.python-version` and `runtime.txt`.
- Mode A always runs an 80/20 split and 5-fold cross validation.
- Generated artifacts are stored under `data/` and `reports/`.
- Tests cover anonymization, cleaning, metadata recovery, Mode A, Mode B, exports, and invalid data handling.
- Public subject IDs are deterministic and preserved through the private mapping file in `Working Documents/`.
