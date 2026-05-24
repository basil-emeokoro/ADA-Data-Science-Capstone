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

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running Backend

```powershell
uvicorn backend.app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`.

## Running Frontend

```powershell
cd frontend
npm install
npm run dev
```

The frontend calls the FastAPI backend through `VITE_API_BASE_URL`, defaulting to `http://127.0.0.1:8000`.

## Upload Workflow

1. Upload a CSV examination dataset.
2. Review detected sensitive fields.
3. For Mode A, anonymize before cleaning.
4. Export the ADA-safe anonymized dataset before cleaning if needed.
5. Recover missing metadata such as paper count or maximum scores.
6. Run Mode A benchmarking or Mode B Lite prediction.

## Mode A Usage

Mode A requires anonymization. The system:

1. Detects identifiers such as candidate and centre fields.
2. Replaces candidate identifiers with synthetic IDs such as `CAND_000001`.
3. Exports the anonymized ADA-safe dataset before cleaning.
4. Cleans and validates score data.
5. Generates scenarios by hiding each available paper component.
6. Trains Random Forest, Gradient Boosting, XGBoost, CatBoost, and SVR models.
7. Evaluates with MAE, MSE, RMSE, R², 80/20 split, and 5-fold cross validation.
8. Produces ranking tables and Plotly explainability visuals.

## Mode B Usage

Mode B Lite does not require anonymization. The system:

1. Cleans uploaded data.
2. Detects candidates with exactly one missing paper score.
3. Rejects candidates with multiple missing papers.
4. Trains scenario models from complete rows for the same subject.
5. Selects the correct model for each missing paper.
6. Exports completed predictions with `prediction_status`.

## Export Process

Exports are written to `data/exports/` and include:

- ADA-safe anonymized datasets
- Cleaned datasets
- Mode A metrics and ranking tables
- Mode B completed prediction files

## Troubleshooting

- If CatBoost or XGBoost installation fails, install Microsoft Visual C++ Build Tools and retry.
- If SHAP plots are slow, run on a smaller sample during development.
- If score validation fails, check that paper scores do not exceed maximum scores.
- If paper count cannot be inferred, provide metadata through the request payload or UI.

## Reproducibility Notes

- The training pipeline uses fixed random seeds.
- Mode A always runs an 80/20 split and 5-fold cross validation.
- Generated artifacts are stored under `data/` and `reports/`.
- Tests cover anonymization, cleaning, metadata recovery, Mode A, Mode B, exports, and invalid data handling.
