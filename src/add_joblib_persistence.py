"""Insert persistence cells without changing any previous cell or output."""
from pathlib import Path
import copy
import hashlib
import nbformat

root = Path(__file__).resolve().parents[1]
path = root / 'notebooks/master_notebook_v2_smote_nc_classweight.ipynb'
nb = nbformat.read(path, 4)
before = copy.deepcopy(nb.cells)
assert not any(c.source.startswith('# 9. Penyimpanan Model') for c in before)
index = next(i for i,c in enumerate(nb.cells) if c.source.startswith('# 10. Interpretabilitas'))
artifacts = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (root / 'results/master_v2').rglob('*') if p.is_file()}
markdown = '''# 9. Penyimpanan Model Terlatih dengan Joblib

Bagian ini menyimpan **delapan pipeline terbaik yang sudah di-fit**, bukan melatih ulang.
Pipeline disimpan lengkap supaya konfigurasi model dan sampler tetap tercatat; saat prediksi,
sampler imbalanced-learn tidak dijalankan. Semua berkas berada di `RUN_DIR / "models"`:

- `best_models.joblib`: dictionary `(algorithm, method) → best_estimator_` untuk inference.
- `experiment_state.joblib`: dictionary `state` berisi `searches`, hasil evaluasi, prediksi,
  probabilitas, data train/test asli beserta snapshot dan konfigurasi yang diperlukan SHAP.
  Objek `GridSearchCV` mencakup estimator terlatih dan hasil CV, bukan hanya best parameters.
- `model_manifest.json`: versi dependency, daftar model, parameter terbaik, dan hash arsip.

Prasyarat: `searches` dan state eksperimen masih tersedia di kernel. **Output notebook dan CSV
bukan pengganti objek model.** Jika kernel kosong, cell berhenti dan tidak menjalankan training.
Jalankan cell ini setelah evaluasi selesai, sebelum menutup/reset kernel. Cell dapat dijalankan
sendiri bila state sudah tersedia; tidak perlu menjalankan ulang GridSearchCV.

Arsip ditulis sementara, dimuat kembali, lalu diverifikasi sebelum menggantikan berkas tujuan.
Prediksi kedelapan pipeline setelah load harus identik dengan prediksi test sebelumnya.
Joblib bergantung pada kompatibilitas versi Python/library; gunakan versi yang tercatat dan
muat hanya arsip yang tepercaya. Objek fungsi notebook tidak diserialisasi.
'''
code = '''from pathlib import Path
from tempfile import TemporaryDirectory
from importlib.metadata import version
from datetime import datetime, timezone
import sys
import json
import hashlib
import joblib
import numpy as np
import pandas as pd

persistence_state_names = [
    "searches", "results", "predictions", "probabilities", "FEATURES", "RANDOM_STATE",
    "TEST_SIZE", "POS_LABEL", "LABELS", "X_train", "y_train", "X_test", "y_test",
    "X_train_snapshot", "y_train_snapshot", "X_test_snapshot", "y_test_snapshot",
    "n_test_original", "categorical_indices", "scale_pos_weight",
]
persistence_missing = [name for name in persistence_state_names + ["RUN_DIR", "assert_original_data"]
                       if name not in globals()]
if persistence_missing:
    raise RuntimeError("Model belum dapat disimpan: state kernel tidak tersedia: "
                       + ", ".join(persistence_missing)
                       + ". Gunakan kernel eksperimen yang sudah selesai; tidak ada training otomatis.")
assert_original_data()
assert len(searches) == len(results) == 8
assert set(searches) == set(predictions) == set(probabilities)
assert set(searches) == set(zip(results["Model"], results["Imbalance Method"]))
persisted_models = {key: search.best_estimator_ for key, search in searches.items()}
persistence_dir = Path(RUN_DIR) / "models"
persistence_dir.mkdir(parents=True, exist_ok=True)
persistence_metadata = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version, "random_state": RANDOM_STATE, "positive_class": POS_LABEL,
    "features": list(FEATURES), "test_rows": len(X_test),
    "packages": {name: version(name) for name in ["joblib", "numpy", "pandas",
                  "scikit-learn", "imbalanced-learn", "xgboost", "lightgbm"]},
    "models": [{"algorithm": key[0], "method": key[1],
                "best_parameters": search.best_params_, "best_cv_f1": float(search.best_score_)}
               for key, search in searches.items()],
}
persistence_bundle = {
    "format_version": 1,
    "metadata": persistence_metadata.copy(),
    "state": {name: globals()[name] for name in persistence_state_names},
}
with TemporaryDirectory(prefix="joblib_save_", dir=persistence_dir) as persistence_temp:
    persistence_temp = Path(persistence_temp)
    joblib.dump(persisted_models, persistence_temp / "best_models.joblib", compress=3)
    joblib.dump(persistence_bundle, persistence_temp / "experiment_state.joblib", compress=3)
    persistence_loaded_models = joblib.load(persistence_temp / "best_models.joblib")
    persistence_loaded_state = joblib.load(persistence_temp / "experiment_state.joblib")["state"]
    assert set(persistence_loaded_models) == set(searches)
    assert set(persistence_loaded_state["searches"]) == set(searches)
    pd.testing.assert_frame_equal(persistence_loaded_state["results"], results, check_exact=True)
    for persistence_name in ["X_train", "X_test", "X_train_snapshot", "X_test_snapshot"]:
        pd.testing.assert_frame_equal(persistence_loaded_state[persistence_name], globals()[persistence_name], check_exact=True)
    for persistence_name in ["y_train", "y_test", "y_train_snapshot", "y_test_snapshot"]:
        pd.testing.assert_series_equal(persistence_loaded_state[persistence_name], globals()[persistence_name], check_exact=True)
    for persistence_key in searches:
        np.testing.assert_array_equal(persistence_loaded_state["predictions"][persistence_key], predictions[persistence_key])
        np.testing.assert_array_equal(persistence_loaded_state["probabilities"][persistence_key], probabilities[persistence_key])
        # Verifikasi kedua arsip menyimpan estimator yang dapat dipakai tanpa fit ulang.
        for persistence_model in [persistence_loaded_models[persistence_key],
                                  persistence_loaded_state["searches"][persistence_key].best_estimator_]:
            assert np.array_equal(persistence_model.classes_, [0, 1])
            np.testing.assert_array_equal(persistence_model.predict(X_test), predictions[persistence_key])
            np.testing.assert_allclose(persistence_model.predict_proba(X_test)[:, 1],
                                       probabilities[persistence_key], rtol=1e-7, atol=1e-8)
    assert_original_data()
    persistence_metadata["validation"] = "passed: both archives reproduce all 8 test predictions and probabilities"
    persistence_metadata["sha256"] = {
        name: hashlib.sha256((persistence_temp / name).read_bytes()).hexdigest()
        for name in ["best_models.joblib", "experiment_state.joblib"]}
    (persistence_temp / "model_manifest.json").write_text(
        json.dumps(persistence_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    for persistence_filename in ["best_models.joblib", "experiment_state.joblib", "model_manifest.json"]:
        (persistence_temp / persistence_filename).replace(persistence_dir / persistence_filename)
del persistence_loaded_models, persistence_loaded_state, persistence_model
print("Delapan model tersimpan dan lolos verifikasi load tanpa training ulang:", persistence_dir)
'''
usage = '''### Memuat Kembali Model tanpa Training

Untuk inference, jalankan contoh berikut di kernel yang baru. Isi `MODEL_DIR` sesuai lokasi proyek.
Data prediksi harus sudah memakai empat fitur dan encoding yang sama seperti penelitian.

```python
from pathlib import Path
import joblib
MODEL_DIR = Path("D:/Kuliah/Semester9/Notebook/results/master_v2/models")
loaded_models = joblib.load(MODEL_DIR / "best_models.joblib")
print(list(loaded_models))
# Pilih key dari daftar tersebut; tidak diperlukan fit().
# probability = loaded_models[key].predict_proba(X_input)[:, 1]
```

Untuk melanjutkan **section 10 SHAP** di kernel baru tanpa mengulang eksperimen, muat state
tepercaya dan definisikan kembali helper validasi berikut. Path `RUN_DIR` ditentukan dari lokasi
arsip agar tidak bergantung pada direktori kerja kernel. Seluruh snapshot berasal dari arsip
eksperimen, bukan dibuat ulang dari data yang mungkin telah berubah.

```python
from pathlib import Path
import joblib
MODEL_DIR = Path("D:/Kuliah/Semester9/Notebook/results/master_v2/models")
saved_experiment = joblib.load(MODEL_DIR / "experiment_state.joblib")
assert saved_experiment["format_version"] == 1
globals().update(saved_experiment["state"])
RUN_DIR = MODEL_DIR.parent

def assert_original_data():
    assert X_test.equals(X_test_snapshot) and y_test.equals(y_test_snapshot)
    assert X_train.equals(X_train_snapshot) and y_train.equals(y_train_snapshot)
    assert len(X_test) == len(y_test) == n_test_original
    assert list(X_train.columns) == list(X_test.columns) == FEATURES
    assert X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index)
    assert X_train.index.intersection(X_test.index).empty

assert_original_data()
# Berikutnya jalankan section 10 saja.
```
'''
compile(code, '<joblib persistence>', 'exec')
nb.cells[index:index] = [nbformat.v4.new_markdown_cell(markdown),
                         nbformat.v4.new_code_cell(code), nbformat.v4.new_markdown_cell(usage)]
assert nb.cells[:index] + nb.cells[index+3:] == before
nbformat.validate(nb)
nbformat.write(nb, path)
reloaded = nbformat.read(path, 4)
assert reloaded.cells[:index] + reloaded.cells[index+3:] == before
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p,digest in artifacts.items())
print('Added 3 cells; all', len(before), 'original cells/outputs and existing artifacts unchanged.')
