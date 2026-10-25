"""Verify SHAP extraction with mathematical fixtures, without training any model."""
from pathlib import Path
import ast
import json
import hashlib
import nbformat
import numpy as np
import pandas as pd
import shap
from scipy.special import expit

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/master_notebook_v2_smote_nc_classweight.ipynb'
nb = nbformat.read(path, 4)
start = next(i for i,c in enumerate(nb.cells) if c.source.startswith('# 10. Interpretabilitas'))
section = nb.cells[start:]
for cell in section:
    if cell.cell_type != 'code':
        continue
    tree = ast.parse(cell.source)
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in {'fit','fit_resample','fit_transform'} for node in ast.walk(tree))
helper_cell = next(c for c in section if c.cell_type == 'code' and 'def shap_positive_explanation' in c.source)
function = next(node for node in ast.parse(helper_cell.source).body
                if isinstance(node, ast.FunctionDef) and node.name == 'shap_positive_explanation')
features = ['Gender', 'Age (Month)', 'Weight', 'Height']
env = {'shap': shap, 'np': np, 'FEATURES': features, 'expit': expit}
exec(compile(ast.Module(body=[function], type_ignores=[]), '<extraction-helper>', 'exec'), env)
extract = env['shap_positive_explanation']
# Analytic fixtures test shape handling only: these are not empirical model/SHAP results.
values = np.arange(28).reshape(7, 4) / 100 - 0.12
frame = pd.DataFrame(np.zeros((7,4)), columns=features)
base = 0.23
margin = base + values.sum(axis=1)
probability = expit(margin)
multi = np.stack([-values, values], axis=2)
fixtures = [
    ('2D scalar base', values, base),
    ('2D singleton base', values, [base]),
    ('list per class', [-values, values], [-base, base]),
    ('3D output-last', multi, [-base, base]),
    ('Explanation single', shap.Explanation(values=values, base_values=np.full(7,base)), None),
    ('Explanation multi', shap.Explanation(values=multi, base_values=np.tile([-base,base],(7,1))), None),
]
passed = []
for name, raw, expected in fixtures:
    result = extract(raw, expected, frame, [0,1], margin, probability)
    np.testing.assert_array_equal(result.values, values)
    np.testing.assert_array_equal(result.base_values, np.full(7,base))
    passed.append(name)
# Single local sample: verify waterfall-ready shape/base.
local = extract(values[:1], base, frame.iloc[:1], [0,1], margin[:1], probability[:1])
assert local[0].values.shape == (4,) and np.ndim(local[0].base_values) == 0
passed.append('single local sample')
rejected = []
for name, raw, expected, target_margin, target_probability in [
    ('class-first 3D', multi.transpose(2,0,1), [-base,base], margin, probability),
    ('wrong feature count', values[:,:3], base, margin, probability),
    ('ambiguous base', values, [-base,base], margin, probability),
    ('wrong class direction', -values, -base, margin, probability),
    ('wrong probability', values, base, margin, 1-probability),
]:
    try:
        extract(raw, expected, frame, [0,1], target_margin, target_probability)
    except (ValueError, AssertionError):
        rejected.append(name)
    else:
        raise AssertionError(f'Invalid fixture accepted: {name}')

# Selection check on actual saved metrics; no model fitting or old cells executed.
results = pd.read_csv(root / 'results/master_v2/comparison.csv')
selected = results.sort_values(['Model','F1','Recall','ROC-AUC','Imbalance Method'],
    ascending=[True,False,False,False,True], kind='mergesort').drop_duplicates('Model')
assert len(selected) == 2
for _,row in selected.iterrows():
    assert row['F1'] == results.loc[results.Model == row.Model, 'F1'].max()

report = {
    'section_code_status': 'syntax and mathematical shape/additivity fixture checks passed',
    'empirical_shap_status': 'not executed: fitted searches and original experiment state unavailable',
    'no_retraining_or_resampling_calls_in_new_section': True,
    'existing_cell_count': start, 'appended_cell_count': len(section),
    'original_executed_code_cells': sum(c.get('execution_count') is not None for c in nb.cells[:start]),
    'new_executed_code_cells': sum(c.get('execution_count') is not None for c in section),
    'supported_fixtures_passed': passed, 'invalid_fixtures_rejected': rejected,
    'selection_from_actual_saved_metrics': selected[['Model','Imbalance Method','F1','Recall','ROC-AUC']].to_dict('records'),
    'shap_version_checked': shap.__version__,
    'old_cells_sha256': hashlib.sha256(json.dumps(nb.cells[:start],sort_keys=True).encode()).hexdigest(),
}
(root / 'results/master_v2/shap_section_review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
