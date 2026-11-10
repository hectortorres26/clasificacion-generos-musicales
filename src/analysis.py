"""Clasificación musical con preprocesamiento ajustado dentro de cada partición."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from .common import configure_plots, save_json, provenance, save_figure

FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "speechiness",
    "tempo",
    "valence",
]
LABELS = ["Hip-Hop", "Rock"]
SEED = 42


def load_data(tracks_path, metrics_path):
    tracks = pd.read_csv(tracks_path)
    metrics = pd.read_json(metrics_path, precise_float=True)
    for frame, cols in [
        (tracks, ["track_id", "genre_top"]),
        (metrics, ["track_id"] + FEATURES),
    ]:
        if not set(cols).issubset(frame.columns):
            raise ValueError(f"Faltan columnas de {cols}")
        if frame.track_id.isna().any() or frame.track_id.duplicated().any():
            raise ValueError("track_id debe ser único y no nulo en cada archivo.")
    data = metrics.merge(
        tracks[["track_id", "genre_top"]], on="track_id", validate="one_to_one"
    )
    data = data.loc[data.genre_top.isin(LABELS)].copy()
    if set(data.genre_top.unique()) != set(LABELS):
        raise ValueError("Se requieren ambos géneros.")
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="raise")
    if np.isinf(data[FEATURES].to_numpy()).any():
        raise ValueError("Las variables contienen infinitos.")
    if data[FEATURES].isna().all().any():
        raise ValueError("Existe una variable sin observaciones.")
    if data.duplicated(subset=FEATURES, keep=False).any():
        raise ValueError(
            "Hay pistas con características idénticas: agrúpelas antes de separar los datos."
        )
    return data


def candidates():
    def pipeline(model):
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, svd_solver="full")),
                ("model", model),
            ]
        )

    return {
        "base_mayoritaria": pipeline(DummyClassifier(strategy="most_frequent")),
        "arbol_balanceado": pipeline(
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=SEED,
            )
        ),
        "regresion_logistica": pipeline(
            LogisticRegression(
                C=1, class_weight="balanced", max_iter=3000, random_state=SEED
            )
        ),
    }


def evaluate(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "por_clase": classification_report(
            y, pred, labels=LABELS, output_dict=True, zero_division=0
        ),
    }


def run(tracks_path, metrics_path, output_dir="reports"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    configure_plots()
    data = load_data(tracks_path, metrics_path)
    X = data[FEATURES]
    y = data.genre_top
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    assert set(X_train.index).isdisjoint(X_test.index)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = candidates()
    cv_rows = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={"f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
            n_jobs=1,
            error_score="raise",
        )
        cv_rows.append(
            {
                "modelo": name,
                "f1_macro_media": float(scores["test_f1_macro"].mean()),
                "f1_macro_desviacion": float(scores["test_f1_macro"].std(ddof=1)),
                "balanced_accuracy_media": float(
                    scores["test_balanced_accuracy"].mean()
                ),
            }
        )
    # La elección se realiza solo con entrenamiento; nunca con el resultado de prueba.
    chosen = max(cv_rows, key=lambda r: r["f1_macro_media"])["modelo"]
    fitted = clone(models[chosen]).fit(X_train, y_train)
    predicted = fitted.predict(X_test)
    baseline = clone(models["base_mayoritaria"]).fit(X_train, y_train)
    selected_metrics = evaluate(y_test, predicted)
    rng = np.random.default_rng(SEED)
    boot = []
    yt = y_test.to_numpy()
    for _ in range(500):
        ids = rng.integers(0, len(yt), len(yt))
        boot.append(f1_score(yt[ids], predicted[ids], average="macro", zero_division=0))
    summary = {
        "n_pistas": len(data),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "seed": SEED,
        "clases": y.value_counts().to_dict(),
        "clases_test": y_test.value_counts().to_dict(),
        "features": FEATURES,
        "cv": cv_rows,
        "modelo_seleccionado": chosen,
        "test": selected_metrics,
        "baseline_test": evaluate(y_test, baseline.predict(X_test)),
        "f1_macro_ic95_bootstrap": np.quantile(boot, [0.025, 0.975]).tolist(),
        "componentes_pca": int(fitted.named_steps["pca"].n_components_),
        "varianza_pca": float(
            fitted.named_steps["pca"].explained_variance_ratio_.sum()
        ),
        "fuentes": [provenance(tracks_path), provenance(metrics_path)],
    }
    save_json(output / "metrics.json", summary)
    pd.DataFrame(cv_rows).to_csv(output / "validacion_cruzada.csv", index=False)
    class_rows = {
        k: v for k, v in selected_metrics["por_clase"].items() if isinstance(v, dict)
    }
    pd.DataFrame(class_rows).T.to_csv(output / "clasificacion.csv", index_label="clase")
    cm = confusion_matrix(y_test, predicted, labels=LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    counts = y.value_counts().reindex(LABELS)
    axes[0].bar(LABELS, counts, color=["#d66a38", "#23788e"])
    axes[0].set(title="Composición del conjunto", ylabel="Pistas")
    for i, n in enumerate(counts):
        axes[0].text(i, n + 30, f"{n:,}", ha="center")
    names = ["Base", "Árbol", "Regresión\nlogística"]
    axes[1].bar(
        names,
        [r["f1_macro_media"] for r in cv_rows],
        yerr=[r["f1_macro_desviacion"] for r in cv_rows],
        capsize=4,
        color="#23788e",
    )
    axes[1].set(
        title="Selección: CV de 5 particiones",
        ylabel="F1 macro (media ± DE)",
        ylim=(0, 1),
    )
    axes[2].imshow(cm, cmap="Blues")
    axes[2].set(
        xticks=[0, 1],
        xticklabels=LABELS,
        yticks=[0, 1],
        yticklabels=LABELS,
        title="Prueba reservada · matriz de confusión",
        xlabel="Predicción",
        ylabel="Clase real",
    )
    for i in range(2):
        for j in range(2):
            axes[2].text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "#172b4d",
                fontsize=16,
            )
    fig.suptitle(
        f"Géneros musicales | F1 macro en prueba: {selected_metrics['f1_macro']:.3f}",
        fontsize=17,
    )
    save_figure(fig, output / "evaluacion.png")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", default="data/songs.csv")
    p.add_argument("--metrics", default="data/echonest-metrics.json")
    p.add_argument("--output", default="reports")
    a = p.parse_args()
    r = run(a.tracks, a.metrics, a.output)
    print(f"Modelo: {r['modelo_seleccionado']}; F1 macro: {r['test']['f1_macro']:.4f}")
