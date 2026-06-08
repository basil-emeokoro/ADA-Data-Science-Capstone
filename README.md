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
5. For Mode A, click `Export ADA-Safe Dataset` to export anonymized data before cleaning.
6. Run Mode A benchmarking or Mode B Lite prediction.

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
3. Supports an explicit `Export ADA-Safe Dataset` action before cleaning.
4. Exports the anonymized ADA-safe dataset again during Mode A processing for reproducibility.
5. Cleans and validates score data, removing incomplete active-paper records before benchmarking.
6. Generates 2-, 3-, and 4-paper scenarios by hiding each available paper component.
7. Trains Random Forest, Gradient Boosting, XGBoost, CatBoost, and SVR models.
8. Evaluates with MAE, MSE, RMSE, R2, 80/20 split, and 5-fold cross validation.
9. Produces ranking tables, model summaries, and Plotly explainability visuals for best scenario models.

## Mode B Usage

Mode B Lite does not require anonymization. The system:

1. Cleans uploaded data.
2. Detects candidates with exactly one missing paper score.
3. Rejects candidates with multiple missing papers.
4. Trains scenario models from complete rows for the same subject.
5. Selects the correct model for each missing paper.
6. Exports completed predictions with `prediction_status`.
7. Exports unpredictable cases separately for reference.

## Export Process

Exports are written to `data/exports/` and include:

- Mode A ADA-safe anonymized CSV
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

- ADA-safe anonymized dataset for submission-safe review
- clean training records used for modelling
- invalid records isolated from applicable-paper invalid values
- absent records isolated from true absences
- unpredictable records that could not be predicted
- metrics CSV and model summary CSV/JSON
- Mode B completed prediction file

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
