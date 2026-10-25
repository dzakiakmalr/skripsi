"""Execute from a clean kernel, saving notebook progress after every cell."""
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'notebooks/master_notebook_v2_smote_nc_classweight.ipynb'
nb = nbformat.read(path, as_version=4)
start = time.perf_counter()

def save_cell(cell, cell_index, **kwargs):
    nbformat.write(nb, path)
    print(f'Completed cell {cell_index}, elapsed {time.perf_counter()-start:.1f}s', flush=True)

client = NotebookClient(nb, timeout=None, kernel_name='python3',
                        resources={'metadata': {'path': str(ROOT / 'notebooks')}},
                        on_cell_executed=save_cell)
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print('Execution complete', flush=True)
