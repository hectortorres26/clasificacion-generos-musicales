"""Ejecuta el cuaderno en un kernel nuevo con el mismo Python de este comando."""

from pathlib import Path
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MPLBACKEND", "Agg")
root = Path(__file__).resolve().parents[1]
path = root / "clasificador_canciones.ipynb"
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [
    sys.executable,
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}",
]
client = NotebookClient(notebook, km=manager, timeout=600, allow_errors=False)
try:
    client.execute(cwd=str(root))
finally:
    # Un gestor entregado por el llamador requiere cierre explícito.
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
for cell in notebook.cells:
    cell.metadata.pop("execution", None)
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"Cuaderno ejecutado y validado: {path.name}")
