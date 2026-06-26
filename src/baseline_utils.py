from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def build_random_forest_baseline(
    *,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """crea el modelo de baseline de random forest con los hiperparámetros especificados."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def evaluate_split(model, X, y_true, split_name: str) -> dict:
    """devuelve un diccionario con las métricas de evaluación para un split dado."""
    y_pred = model.predict(X)
    return {
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def classification_report_df(model, X, y_true) -> pd.DataFrame:
    """devuelve el informe de clasificación como un dataframe."""
    report = classification_report(y_true, model.predict(X), output_dict=True)
    return pd.DataFrame(report).transpose()


def plot_confusion_for_split(
    model,
    X,
    y_true,
    labels: Iterable[str],
    title: str,
    *,
    ax=None,
    cmap: str = "Blues",
):
    """plotea la matriz de confusión para un split dado."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_true, model.predict(X), labels=list(labels))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    disp.plot(ax=ax, cmap=cmap, colorbar=False, xticks_rotation=45)
    ax.set_title(title)
    return ax


def top_feature_importances(model, feature_names: Iterable[str], top_n: int = 20) -> pd.DataFrame:
    """devuelve un dataframe con las características más importantes del modelo."""
    importances = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": model.feature_importances_,
        }
    )
    return importances.sort_values("importance", ascending=False).head(top_n)
