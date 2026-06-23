# Submission Reflection and User Manual

## Project Overview

This project, **Predicting Missing Examination Component Scores: A Comparative Study of Machine Learning Models for High-Stakes Educational Assessment**, implements a reproducible machine learning web application for evaluating and predicting missing examination paper component scores. The system was built as a research-first capstone application rather than an enterprise production platform. Its main purpose is to support transparent experimentation, privacy-preserving data handling, explainable modelling, and publication-quality evidence generation.

The application supports two operating modes. **Mode A** is the primary research mode. It performs controlled experimental benchmarking by hiding one known paper score at a time, training models using the remaining scores and engineered features, and comparing predictions against the actual hidden score. **Mode B Lite** is the operational demonstration mode. It handles candidate-level missing scores where one applicable paper is missing for a candidate and predicts the missing value using complete valid records in the uploaded file.

The implementation uses a React and TypeScript frontend, a FastAPI backend, and a Python machine learning stack built around pandas, NumPy, scikit-learn, XGBoost, CatBoost, SHAP, and Plotly. The application is also Dockerized so it can be run locally through a single container.

## Reflection on the Design

The core design decision was to prioritize reproducibility and interpretability over unnecessary system complexity. This is appropriate because the project is a data science capstone, not a commercial scoring platform. The workflow therefore makes data inspection, cleaning, anonymization, metadata validation, model comparison, and export generation visible to the user.

Privacy preservation was treated as a first-class requirement. Candidate identifiers are anonymized, and public submission datasets use stable subject pseudonyms instead of original subject codes. A private mapping file is kept outside the public submission package in an ignored confidential folder. This separation allows the project to produce public evidence while preserving the ability to trace subject mappings privately if needed for internal validation.

The modelling pipeline compares multiple regression models under the same validation rules. Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor, CatBoost Regressor, and Support Vector Regressor are evaluated using MAE, MSE, RMSE, and R-squared. The application also provides model ranking tables and explainability outputs such as feature importance and SHAP-based summaries. These outputs support the research objective of comparing models, rather than simply producing a single prediction.

One important engineering hardening step was correcting potential target leakage in Mode A. Aggregate features are recalculated after the target paper has been hidden for a scenario. This prevents the hidden score from indirectly entering the model through totals, means, spreads, or other derived features. This correction is important for research validity because benchmark performance should reflect realistic predictive information, not accidental access to the target value.

## Data Privacy Model

The project separates confidential working files from public submission artifacts.

Confidential files remain in ignored folders such as `Working Documents/`. These may include raw source files, private subject mappings, and any original identifiers required for internal traceability. These files are not committed to GitHub and are not included in the public submission package.

Public exports use safe fields only:

- `subject_id`
- `paper_count`
- anonymized candidate identifier
- applicable paper scores
- paper maximum score metadata

The public dataset does not expose original subject codes or original candidate identifiers. Paper maximum scores are included because they are required metadata for score validation, normalization, reproducibility, and interpretation of model results.

## User Manual

### Local Docker Startup

From PowerShell, run:

```powershell
cd "C:\ADA_Data_Science\Projects\Capstone Project"
docker build -t ada-score-prediction:latest .
docker run --rm -p 8000:8000 ada-score-prediction:latest
```

Open:

```text
http://localhost:8000
```

If port `8000` is already in use, run the container on another host port:

```powershell
docker run --rm -p 8001:8000 ada-score-prediction:latest
```

Then open:

```text
http://localhost:8001
```

The health endpoint is available at:

```text
http://localhost:8000/api/health
```

or, if using the alternate port:

```text
http://localhost:8001/api/health
```

### Upload and Detection

The first step is to upload a CSV dataset. After upload, click `Detect`. The application inspects the dataset and reports:

- row count
- detected columns
- sensitive fields
- subject grouping
- inferred paper counts
- detected maximum score metadata

If the file uses the public schema, the system can use `subject_id`, `paper_count`, `p1_score` to `p4_score`, and `p1_max` to `p4_max` directly. If the file uses a raw schema, the user can configure column mapping before running the pipeline.

### Metadata Recovery

Paper maxima are required metadata. The application must not assume that missing maximum scores are `100`. If maxima are missing from the dataset, the metadata editor requires the user to enter the maximum score for each applicable paper.

Applicable paper rules are based on paper count:

- 2-paper subject: P1 and P2 only
- 3-paper subject: P1, P2, and P3 only
- 4-paper subject: P1, P2, P3, and P4

Non-applicable papers are ignored for validation and modelling. The metadata editor highlights pending rows, supports filtering, and provides bulk fill tools for subjects that share the same maximum score structure.

### Optional Cleaning Configuration

The `Configure Cleaning` workflow allows a user to map source columns to canonical application fields before processing. This is useful when a raw dataset uses different column names for subject, candidate, paper score, or maximum score fields.

The cleaning preview reports how many rows are expected to be clean, invalid, absent, incomplete, predictable, or unpredictable under the examination cleaning rules. This gives the user a way to inspect data quality before starting model training.

### Mode A Benchmarking

Mode A is the research benchmarking workflow. It should be used when the dataset contains complete valid paper scores. The pipeline:

1. anonymizes sensitive identifiers
2. validates required metadata
3. applies cleaning rules
4. isolates invalid, absent, incomplete, and unpredictable records
5. generates hide-one-paper scenarios
6. trains the approved regression models
7. evaluates performance with MAE, MSE, RMSE, and R-squared
8. produces ranking tables, explainability outputs, and export files

Mode A can take time on large multi-subject datasets because it trains several models across many subject and paper scenarios. For demonstration deployments with limited resources, smaller public datasets are recommended. Full benchmark evidence is best generated locally.

### Mode B Lite Prediction

Mode B Lite is for row-level missing-score prediction. Users should keep all applicable paper columns in the file and mark missing candidate scores as blank or missing values. Users should not delete an entire paper column.

Mode B trains from complete valid records in the uploaded file. If all rows are missing the same target paper, the application cannot train a model for that target from that file alone. In that case, the user must provide complete records in the same file or treat the case as a future extension requiring a stored reference model.

Mode B output includes completed prediction rows, unpredictable records, invalid records, absent records, and a model summary.

### Chart Selection and Explainability

After the pipeline completes, the dashboard provides chart and explainability options. The chart selector lets the user switch among available result views, including model comparison, residual/diagnostic plots, and explainability panels where available. The best model card, dataset quality card, model ranking table, and export package are intended to support screenshots and report writing.

### Export Files

The application produces downloadable export files from the browser. Depending on the mode, these include:

- ADA-safe public dataset
- clean training records
- invalid records
- absent records
- unpredictable records
- completed Mode B predictions
- metrics CSV
- model summary CSV and JSON

Generated export files are placed under the export directory during local execution. Confidential source folders and generated exports are ignored by Git unless intentionally packaged into a public-safe submission folder.

## Deployment Notes

The application can be deployed to a Docker-compatible hosting service. For Render, create a Web Service from the GitHub repository and select Docker deployment. Use the repository Dockerfile and set the health check path to:

```text
/api/health
```

After Render builds the service, open the public Render URL in the browser and test the upload page with a small public-safe CSV. Because hosted free-tier resources are limited, the deployed version is best used for demonstration, detection, metadata validation, small pipeline runs, and export workflows. Large benchmarking runs should be executed locally.

## Submission Reflection

This project demonstrates an end-to-end research software workflow for missing examination component score prediction. The strongest aspect of the implementation is that it connects data privacy, metadata recovery, data cleaning, model benchmarking, explainability, and export generation into one reproducible interface. The application is not just a model script; it is a usable research system with a clear workflow from upload to evidence generation.

The project also illustrates why metadata matters in educational assessment modelling. Maximum scores cannot be guessed safely because different subjects and paper components may have different maxima. The application therefore requires maximum score metadata before cleaning, normalization, training, benchmarking, or prediction. This reduces the risk of invalid score validation and improves reproducibility.

Another important lesson is that benchmarking pipelines must guard against target leakage. In a hide-and-predict experiment, aggregate features must be derived only from visible papers. The final implementation recalculates engineered features after hiding the target paper, so model performance reflects legitimate predictive information.

The current system is suitable for capstone submission, local demonstration, Docker deployment, and report evidence generation. Future versions could add persisted trained reference models, asynchronous background jobs for long benchmarks, richer user-managed cleaning rules, and a more scalable deployment architecture. Those extensions are intentionally outside the current frozen scope.

## Suggested Reflection Message

Live application: https://ada-data-science-capstone.onrender.com/

The deployed application demonstrates a privacy-preserving machine learning workflow for examination component score prediction. It supports public-safe dataset upload, subject pseudonymization, metadata validation, configurable cleaning, experimental model benchmarking, missing-score prediction, explainability, dashboard visualization, and export generation. The Docker deployment is hosted on Render, with the health endpoint available at `/api/health`.

For large multi-subject benchmarking, local Docker execution is recommended.
