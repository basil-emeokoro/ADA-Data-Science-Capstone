from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import PartialDependenceDisplay, partial_dependence


def actual_vs_predicted(y_true: list[float], y_pred: list[float]) -> dict[str, Any]:
    frame = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    fig = px.scatter(frame, x="Actual", y="Predicted", title="Actual vs Predicted")
    if not frame.empty:
        lower = float(frame[["Actual", "Predicted"]].min().min())
        upper = float(frame[["Actual", "Predicted"]].max().max())
        fig.add_trace(go.Scatter(x=[lower, upper], y=[lower, upper], mode="lines", name="Ideal"))
    return fig.to_dict()


def residual_plot(y_true: list[float], y_pred: list[float]) -> dict[str, Any]:
    residuals = np.array(y_true) - np.array(y_pred)
    frame = pd.DataFrame({"Predicted": y_pred, "Residual": residuals})
    fig = px.scatter(frame, x="Predicted", y="Residual", title="Residual Analysis")
    fig.add_hline(y=0)
    return fig.to_dict()


def feature_importance(model: Any, feature_columns: list[str]) -> dict[str, Any]:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("svr", model)
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        values = np.zeros(len(feature_columns))
    frame = pd.DataFrame({"Feature": feature_columns, "Importance": values})
    fig = px.bar(frame.sort_values("Importance", ascending=False), x="Feature", y="Importance", title="Feature Importance")
    return fig.to_dict()


def shap_summary(model: Any, X: pd.DataFrame) -> dict[str, Any]:
    try:
        import shap

        sample = X.head(min(10, len(X)))
        explainer = shap.Explainer(model.predict, sample)
        values = explainer(sample)
        mean_abs = np.abs(values.values).mean(axis=0)
        frame = pd.DataFrame({"Feature": sample.columns, "MeanAbsSHAP": mean_abs})
    except Exception:
        frame = pd.DataFrame({"Feature": X.columns, "MeanAbsSHAP": np.zeros(len(X.columns))})
    fig = px.bar(frame.sort_values("MeanAbsSHAP", ascending=False), x="Feature", y="MeanAbsSHAP", title="SHAP Mean Absolute Impact")
    return fig.to_dict()


def partial_dependence_plot(model: Any, X: pd.DataFrame) -> dict[str, Any]:
    if X.empty:
        return go.Figure().to_dict()
    feature = X.columns[0]
    try:
        result = partial_dependence(model, X, [feature], grid_resolution=min(20, max(2, len(X))))
        grid = result["grid_values"][0]
        average = result["average"][0]
        fig = px.line(x=grid, y=average, labels={"x": feature, "y": "Partial dependence"}, title="Partial Dependence Plot")
    except Exception:
        fig = px.scatter(X, x=feature, y=feature, title="Partial Dependence Plot")
    return fig.to_dict()


def eda_plots(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=["number"])
    plots: dict[str, Any] = {}
    if not numeric.empty:
        first = numeric.columns[0]
        plots["histogram"] = px.histogram(df, x=first, title=f"Histogram of {first}").to_dict()
        plots["heatmap"] = px.imshow(numeric.corr(numeric_only=True), title="Correlation Heatmap").to_dict()
        if len(numeric.columns) >= 2:
            plots["scatterplot"] = px.scatter(df, x=numeric.columns[0], y=numeric.columns[1], title="Score Scatterplot").to_dict()
            plots["pairplot"] = px.scatter_matrix(df, dimensions=list(numeric.columns[:4]), title="Pairplot").to_dict()
    return plots
