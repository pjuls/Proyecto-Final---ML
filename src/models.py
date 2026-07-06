from __future__ import annotations

from typing import Any, Callable, Iterable
import numpy as np
import seaborn as sns
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid



# Model builders
def build_random_forest_baseline(*, n_estimators: int = 300, max_depth: int | None = None, min_samples_leaf: int = 1, random_state: int = 42, n_jobs: int = -1,) -> RandomForestClassifier:
    """Crea el modelo de baseline de Random Forest con los hiperparámetros especificados."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def build_mlp(*, hidden_layer_sizes: tuple[int, ...] = (128, 64), activation: str = "relu", alpha: float = 1e-4, learning_rate_init: float = 1e-3, max_iter: int = 200, random_state: int = 42,) -> Pipeline:
    """Crea el pipeline (StandardScaler + MLPClassifier) con los hiperparámetros especificados.

    Se usa un Pipeline porque el MLP (a diferencia de Random Forest) es sensible
    a la escala de las features, especialmente relevante para eGeMAPS donde
    las features tienen rangos muy distintos entre sí.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=False,
                validation_fraction=0.1,
            ),
        ),
    ])




# Evaluación (compartida por todos los modelos)

def evaluate_split(model, X, y_true, split_name: str) -> dict:
    """Devuelve un diccionario con las métricas de evaluación para un split dado."""
    y_pred = model.predict(X)
    return {
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def classification_report_df(model, X, y_true) -> pd.DataFrame:
    """Devuelve el informe de clasificación como un DataFrame."""
    report = classification_report(y_true, model.predict(X), output_dict=True)
    return pd.DataFrame(report).transpose()


def plot_confusion_for_split( model, X, y_true, labels: Iterable[str], title: str, *, ax=None, cmap: str = "Blues",):
    """Plotea la matriz de confusión para un split dado."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(y_true, model.predict(X), labels=list(labels))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    disp.plot(ax=ax, cmap=cmap, colorbar=False, xticks_rotation=45)
    ax.set_title(title)
    return ax


def top_feature_importances(model, feature_names: Iterable[str], top_n: int = 20) -> pd.DataFrame:
    """Devuelve un DataFrame con las características más importantes del modelo."""
    importances = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": model.feature_importances_,
        }
    )
    return importances.sort_values("importance", ascending=False).head(top_n)


def _get_underlying_estimator(model):
    """Si model es un Pipeline, devuelve el último step (el estimador real)."""
    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]
    return model





# Búsqueda de hiperparámetros

def tune_hyperparameters(build_fn: Callable[..., Any],param_grid: dict,X_train,y_train, X_val, y_val,*,scoring: str = "macro_f1",verbose: bool = True,) -> tuple[Any, pd.DataFrame]:
    """Búsqueda manual de hiperparámetros, evaluando en el split de validación.

    No usamos GridSearchCV/cv de sklearn a propósito: como los splits ya están
    armados de forma independiente (ej. por actor/hablante), hacer K-Fold sobre
    train mezclaría esa independencia y daría una estimación optimista. En cambio,
    para cada combinación del grid se entrena en train y se mide en validation,
    igual que se va a usar el modelo final.

    Parameters
    ----------
    build_fn : build_random_forest_baseline o build_mlp
    param_grid : dict de {hiperparametro: [valores a probar]}
    scoring : "accuracy" o "macro_f1"

    Returns
    -------
    (mejor_modelo_ya_entrenado, dataframe_con_todas_las_combinaciones_probadas)
    """
    results = []
    best_score = -np.inf
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        model = build_fn(**params)
        model.fit(X_train, y_train)

        val_metrics = evaluate_split(model, X_val, y_val, "validation")
        score = val_metrics[scoring]

        results.append(
            {
                **params,
                "accuracy": val_metrics["accuracy"],
                "macro_f1": val_metrics["macro_f1"],
            }
        )

        if verbose:
            print(f"{params} -> val_{scoring}={score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    results_df = (
        pd.DataFrame(results)
        .sort_values(scoring, ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        print(f"\nMejores hiperparámetros: {best_params}")
        print(f"Mejor {scoring} en validación: {best_score:.4f}")

    return best_model, results_df


# Entrenamiento + evaluación genérica (sirve para RF, MLP, o cualquier otro)

def run_model( model, X_train, y_train, X_val, y_val, X_test, y_test, nombre_modelo: str, nombre_features: str, *, feature_cols=None, plot_importances: bool = True,):
    """Entrena y evalúa un modelo ya construido (RF, MLP/Pipeline, u otro) de forma genérica.

    Si el estimador subyacente tiene `feature_importances_` (ej. RandomForest) y
    se pasa `feature_cols`, además grafica el top de importancias.
    """
    print(f"{nombre_modelo} - {nombre_features}")


    emotion_labels = sorted(pd.unique(y_train))

    model.fit(X_train, y_train)

    metrics = pd.DataFrame(
        [
            evaluate_split(model, X_train, y_train, "train"),
            evaluate_split(model, X_val, y_val, "validation"),
            evaluate_split(model, X_test, y_test, "test"),
        ]
    ).set_index("split")

    print(metrics.round(4))

    print(f"\nClassification report - Validation ({nombre_features})")
    print(classification_report_df(model, X_val, y_val).round(3))

    print(f"\nClassification report - Test ({nombre_features})")
    print(classification_report_df(model, X_test, y_test).round(3))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_confusion_for_split(model, X_val, y_val, emotion_labels, "Validation", ax=axes[0])
    plot_confusion_for_split(model, X_test, y_test, emotion_labels, "Test", ax=axes[1])

    plt.suptitle(f"{nombre_modelo} - {nombre_features}")
    plt.tight_layout()
    plt.show()

    underlying = _get_underlying_estimator(model)

    if plot_importances and feature_cols is not None and hasattr(underlying, "feature_importances_"):
        importances = top_feature_importances(underlying, feature_cols, top_n=20)
        print(importances)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importances, x="importance", y="feature", color="#2a9d8f")
        plt.title(f"Top 20 Feature Importances ({nombre_features})")
        plt.tight_layout()
        plt.show()

    return model, metrics




# Experimentos por canal (Speech / Song / Speech+Song / cross-channel)

def subset_split(X, y, mask: np.ndarray):
    """Aplica una máscara booleana a X e y."""
    return X[mask], y[mask]

def run_channel_experiment( build_fn: Callable[..., Any], channel: str | None, feature_splits: dict, X_splits: dict, y_splits: dict, nombre_modelo: str, nombre_features: str, *, feature_cols=None,):
    """Corre un experimento filtrado por canal (Experimento A: 'speech', B: 'song').
 
    channel=None corre sobre todo el dataset 
 
    feature_splits: dict {"train": train_df1, "val": val_df1, "test": test_df1}
    X_splits, y_splits: dicts con claves "train", "val", "test".
    """
    data = {}
    for split_name in ("train", "val", "test"):
        X, y = X_splits[split_name], y_splits[split_name]
        if channel is not None:
            mask = (feature_splits[split_name]["channel"].values == channel)
            X, y = subset_split(X, y, mask)
        data[split_name] = (X, y)
 
    etiqueta_canal = channel if channel is not None else "Speech+Song"
 
    return run_model(
        build_fn(),
        *data["train"], *data["val"], *data["test"],
        nombre_modelo=nombre_modelo,
        nombre_features=f"{nombre_features} - {etiqueta_canal}",
        feature_cols=feature_cols,
    )



def run_cross_channel_experiment( build_fn: Callable[..., Any], train_channel: str, eval_channel: str, feature_splits: dict, X_splits: dict, y_splits: dict, nombre_modelo: str, nombre_features: str, *, feature_cols=None,):
    """Entrena en un canal y evalúa en el otro (domain shift Speech<->Song).
 
    Entrena con train filtrado por `train_channel`. Val y test se filtran por
    `eval_channel`, para medir qué tan bien generaliza el modelo al canal no visto.
    """
    train_mask = (feature_splits["train"]["channel"].values == train_channel)
    X_train, y_train = subset_split(X_splits["train"], y_splits["train"], train_mask)
 
    val_mask = (feature_splits["val"]["channel"].values == eval_channel)
    X_val, y_val = subset_split(X_splits["val"], y_splits["val"], val_mask)
 
    test_mask = (feature_splits["test"]["channel"].values == eval_channel)
    X_test, y_test = subset_split(X_splits["test"], y_splits["test"], test_mask)
 
    nombre_exp = f"{nombre_features} - train:{train_channel} -> eval:{eval_channel}"
 
    return run_model(
        build_fn(),
        X_train, y_train, X_val, y_val, X_test, y_test,
        nombre_modelo=nombre_modelo,
        nombre_features=nombre_exp,
        feature_cols=feature_cols,
    )
 
