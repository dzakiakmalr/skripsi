"""Audit hasil aktual, kelengkapan eksekusi, dan pelestarian kode master."""
from pathlib import Path
import json
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

root = Path(__file__).resolve().parents[1]
out = root / 'results/master_v2'
old = nbformat.read(root / 'notebooks/master_notebook.ipynb', 4)
nb = nbformat.read(root / 'notebooks/master_notebook_v2_smote_nc_classweight.ipynb', 4)
nbformat.validate(nb)
code_cells = [c for c in nb.cells if c.cell_type == 'code']
assert all(c.execution_count is not None for c in code_cells)
assert [c.execution_count for c in code_cells] == list(range(1, len(code_cells) + 1))
assert not any(o.output_type == 'error' for c in code_cells for o in c.outputs)
assert all(nb.cells[i].source == old.cells[i].source for i in range(3, 48)
           if old.cells[i].cell_type == 'code')
assert not any('SMOTEENN' in c.source or 'SMOTE-ENN' in c.source for c in nb.cells)
results = pd.read_csv(out / 'comparison.csv')
pred = pd.read_csv(out / 'test_predictions.csv')
manifest = json.loads((out / 'run_manifest.json').read_text(encoding='utf-8'))
assert len(results) == 8 and len(pred) == manifest['test_rows']
assert pred.row_index.tolist() == manifest['test_indices']
assert not set(manifest['train_indices']) & set(manifest['test_indices'])
methods = ['Baseline', 'Class Weight', 'SMOTE-NC', 'SMOTE-NC + ENN']
assert set(zip(results.Model, results['Imbalance Method'])) == {
    (a, m) for a in ['XGBoost', 'LightGBM'] for m in methods}
for _, row in results.iterrows():
    label = f"{row['Model']} — {row['Imbalance Method']}"
    truth = pred.y_true
    prediction = pred[label + ' prediction']
    probability = pred[label + ' probability']
    computed = {'Accuracy': accuracy_score(truth, prediction),
                'Precision': precision_score(truth, prediction),
                'Recall': recall_score(truth, prediction),
                'F1': f1_score(truth, prediction),
                'ROC-AUC': roc_auc_score(truth, probability)}
    for metric, value in computed.items():
        assert np.isclose(row[metric], value, rtol=0, atol=1e-12), (label, metric)
    assert np.array_equal(row[['TN','FP','FN','TP']].to_numpy(dtype=int),
                          confusion_matrix(truth, prediction, labels=[0, 1]).ravel())
    slug = row['Model'].lower() + '_' + str(methods.index(row['Imbalance Method']) + 1)
    cv = pd.read_csv(out / f'cv_{slug}.csv')
    assert len(cv) == 81
    assert cv[[f'split{i}_test_score' for i in range(5)]].notna().all().all()
    assert np.isclose(cv.mean_test_score.max(), row['Best CV F1'])
report = {'executed_code_cells': len(code_cells), 'notebook_error_outputs': 0,
          'eda_cleaning_code_preserved': True, 'experiments': 8,
          'cv_fits': 8 * 81 * 5, 'test_rows': len(pred),
          'metrics_recomputed_from_saved_predictions': 'passed'}
(out / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
