"""Utilidades reproducibles de trazabilidad y visualización."""

from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def configure_plots():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#172b4d",
            "text.color": "#172b4d",
            "figure.facecolor": "white",
            "axes.facecolor": "#f5f8fa",
            "axes.grid": False,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_json(path, data):
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=convert, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def provenance(path):
    path = Path(path)
    return {
        "archivo": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "paquetes": {
            p: importlib.metadata.version(p)
            for p in ["pandas", "numpy", "matplotlib", "scikit-learn"]
        },
    }
