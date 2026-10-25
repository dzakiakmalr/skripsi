"""Append section 10 without replaying or changing any existing notebook cell."""
from pathlib import Path
import copy
import hashlib
import json
import nbformat

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'notebooks/master_notebook_v2_smote_nc_classweight.ipynb'
nb = nbformat.read(PATH, as_version=4)
original = copy.deepcopy(nb.cells)
assert not any(c.source.startswith('# 10. Interpretabilitas') for c in original), 'Section sudah ada.'
artifact_dir = ROOT / 'results/master_v2'
old_artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in artifact_dir.iterdir() if p.is_file() and not p.name.startswith('shap_')}

def md(text):
    nb.cells.append(nbformat.v4.new_markdown_cell(text.strip()))

def code(text):
    compile(text.strip(), '<SHAP section>', 'exec')
    nb.cells.append(nbformat.v4.new_code_cell(text.strip()))

md('''# 10. Interpretabilitas Model dengan SHAP

SHAP (*SHapley Additive exPlanations*) menjelaskan kontribusi fitur terhadap output model
secara global dan lokal. Bagian ini **hanya menggunakan estimator hasil tuning yang sudah
di-fit**; tidak memanggil fit, GridSearchCV, atau resampling. Seluruh hasil eksperimen sebelumnya
dipertahankan. Nomor 10 mengikuti permintaan penambahan tanpa mengubah penomoran bagian lama.

**Prasyarat eksekusi:** kernel masih memiliki `searches`, `results`, `predictions`, `probabilities`,
`X_test`, `y_test`, `FEATURES`, `RUN_DIR`, dan `assert_original_data` dari eksperimen selesai.
File notebook menyimpan kode/output, bukan objek estimator di memori. CSV metrik, best parameters,
dan prediksi saja tidak dapat merekonstruksi pohon terlatih. Jika kernel telah direset dan model
belum diserialisasi, bagian ini berhenti dengan pesan prasyarat; tidak melakukan retraining otomatis.

**Status saat penambahan:** objek model terlatih belum tersedia pada kernel yang diperiksa.
Karena itu output SHAP empiris belum diisi; kode berikut harus dijalankan pada kernel yang memiliki
objek eksperimen asli. Tidak ada ranking atau grafik SHAP yang direkayasa dari metrik sebelumnya.

## 10.1 Library, Prasyarat, dan Snapshot Integritas
Gunakan TreeExplainer dengan `model_output="raw"` dan `feature_perturbation="tree_path_dependent"`.
Tidak ada background baru yang di-fit: referensi berasal dari statistik lintasan pohon terlatih.
Untuk model biner ini, unit kontribusi adalah **log-odds Stunted**, bukan persentase probabilitas.
Probabilitas diperoleh dari sigmoid(nilai dasar + jumlah SHAP), bukan dari sigmoid tiap kontribusi.

API diperiksa pada SHAP 0.52.0. Rujukan:
[TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) dan
[Explanation](https://shap.readthedocs.io/en/latest/generated/shap.Explanation.html).
Kompatibilitas model terlatih tetap harus divalidasi saat objek aslinya tersedia.''')
code('''import shap
import copy
import hashlib
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.utils.validation import check_is_fitted
from IPython.display import display

shap_required = ["searches", "results", "predictions", "probabilities", "X_test", "y_test",
                 "FEATURES", "RUN_DIR", "RANDOM_STATE", "assert_original_data"]
shap_missing = [name for name in shap_required if name not in globals()]
if shap_missing:
    raise RuntimeError(
        "SHAP membutuhkan state eksperimen asli yang sudah selesai. Tidak tersedia: "
        + ", ".join(shap_missing)
        + ". Hubungkan kernel yang menyimpan objek tersebut atau muat arsip model/state asli "
          "yang tepercaya. CSV metrik/parameter bukan model terlatih. Tidak ada training otomatis.")

assert_original_data()
assert FEATURES == ["Gender", "Age (Month)", "Weight", "Height"]
assert X_test.columns.tolist() == FEATURES
assert X_test.index.is_unique and X_test.index.equals(y_test.index)
assert set(X_test["Gender"].unique()).issubset({0, 1})
assert set(y_test.unique()) == {0, 1}
assert len(results) == len(searches) == len(predictions) == len(probabilities) == 8
assert not results.duplicated(["Model", "Imbalance Method"]).any()
assert set(zip(results["Model"], results["Imbalance Method"])) == set(searches)
assert set(searches) == set(predictions) == set(probabilities)
shap_results_snapshot = copy.deepcopy(results)
shap_predictions_snapshot = {key: np.asarray(value).copy() for key, value in predictions.items()}
shap_probabilities_snapshot = {key: np.asarray(value).copy() for key, value in probabilities.items()}
shap_X_test_snapshot = X_test.copy(deep=True)
shap_y_test_snapshot = y_test.copy(deep=True)
shap_output_dir = Path(RUN_DIR)
assert shap_output_dir.is_dir()
shap_existing_artifact_hashes = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in shap_output_dir.iterdir() if p.is_file() and not p.name.startswith("shap_")}
print("Versi SHAP:", shap.__version__)
print("Test asli:", X_test.shape, "| unit SHAP: log-odds Stunted")''')
md('''## 10.2 Pemilihan Satu Model per Algoritma dan Pengambilan Estimator
Pilih F1 **test** tertinggi untuk masing-masing algoritma. Jika seri, gunakan Recall, ROC-AUC,
lalu nama metode sebagai pemecah seri deterministik. Pemilihan ini hanya untuk interpretasi
setelah evaluasi, bukan hyperparameter tuning atau estimasi kinerja baru tanpa bias seleksi.
Nama skenario tidak di-hard-code. Semua delapan estimator dicatat fingerprint-nya untuk
memastikan analisis tidak mengubah model. Pipeline yang berisi transformasi tambahan ditolak
agar fitur mentah tidak diam-diam dijelaskan pada ruang input yang salah.''')
code('''def shap_final_model(best_estimator):
    if hasattr(best_estimator, "named_steps"):
        assert "model" in best_estimator.named_steps
        assert best_estimator.steps[-1][0] == "model"
        for name, step in best_estimator.steps[:-1]:
            if name not in {"smotenc", "enn"} or not hasattr(step, "fit_resample"):
                raise ValueError("Ada preprocessing tambahan; input SHAP harus ditinjau: " + name)
        final_model = best_estimator.named_steps["model"]
    else:
        final_model = best_estimator
    check_is_fitted(final_model)
    assert np.array_equal(final_model.classes_, [0, 1]), "Urutan kelas tidak sesuai protokol."
    assert final_model.n_features_in_ == len(FEATURES)
    return final_model

def shap_model_fingerprint(final_model):
    if hasattr(final_model, "get_booster"):
        payload = bytes(final_model.get_booster().save_raw())
    elif hasattr(final_model, "booster_"):
        payload = final_model.booster_.model_to_string().encode("utf-8")
    else:
        raise TypeError("Fingerprint hanya mendukung XGBoost/LightGBM terlatih.")
    return hashlib.sha256(payload).hexdigest()

shap_model_hashes_before = {
    key: shap_model_fingerprint(shap_final_model(search.best_estimator_))
    for key, search in searches.items()}
shap_selected = (results.sort_values(
    ["Model", "F1", "Recall", "ROC-AUC", "Imbalance Method"],
    ascending=[True, False, False, False, True], kind="mergesort")
    .drop_duplicates("Model").set_index("Model").loc[["XGBoost", "LightGBM"]].reset_index())
shap_models = {}
shap_selected_parameters = []
for shap_row in shap_selected.to_dict("records"):
    shap_key = (shap_row["Model"], shap_row["Imbalance Method"])
    shap_search = searches[shap_key]
    shap_estimator = shap_final_model(shap_search.best_estimator_)
    # Native XGBoost menyimpan nama asli; LightGBM dapat mengganti spasi dengan underscore.
    shap_native_names = (shap_estimator.get_booster().feature_names
                         if hasattr(shap_estimator, "get_booster")
                         else shap_estimator.booster_.feature_name())
    assert list(shap_native_names) in [FEATURES, [f.replace(" ", "_") for f in FEATURES]]
    np.testing.assert_array_equal(shap_search.best_estimator_.predict(X_test), predictions[shap_key])
    np.testing.assert_allclose(shap_search.best_estimator_.predict_proba(X_test)[:, 1],
                               probabilities[shap_key], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(shap_estimator.predict_proba(X_test)[:, 1],
                               probabilities[shap_key], rtol=1e-6, atol=1e-7)
    shap_models[shap_key] = shap_estimator
    shap_selected_parameters.append(copy.deepcopy(shap_search.best_params_))
shap_selected["Best Hyperparameters"] = shap_selected_parameters
display(shap_selected[["Model", "Imbalance Method", "F1", "Recall", "ROC-AUC", "Best Hyperparameters"]])''')
md('''## 10.3 Sampel Test untuk Interpretasi Global
Ambil maksimum 2.000 baris test secara acak dengan seed penelitian. Kedua algoritma menggunakan
baris dan urutan yang sama. Tidak ada resampling atau perubahan nilai fitur. Statistik global
mendeskripsikan sampel ini; hasil dapat sedikit berbeda jika seluruh test digunakan.
Gender adalah jenis kelamin: **0 = M (laki-laki), 1 = F (perempuan)**, bukan besaran kontinu.''')
code('''SHAP_SAMPLE_SIZE = min(2000, len(X_test))
X_shap = X_test.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE).loc[:, FEATURES].copy(deep=True)
assert X_shap.index.isin(X_test.index).all()
pd.testing.assert_frame_equal(X_shap, X_test.loc[X_shap.index, FEATURES])
shap_display_names = ["Gender (0=M, 1=F)", "Age (Month)", "Weight", "Height"]
print("Jumlah sampel SHAP:", len(X_shap))
print("Gender: 0=M (laki-laki), 1=F (perempuan).")
display(X_shap.head())''')
md('''## 10.4 Kompatibilitas Output Biner dan Validasi Aditivitas
Helper menerima list per kelas, array 2D (satu raw margin biner), array 3D dengan format
API `(n_sample, n_feature, n_class)`, atau `shap.Explanation` dengan layout tersebut.
Kelas dipilih dari `classes_`, bukan dengan menebak sumbu berukuran dua. Layout lain ditolak
dengan pesan eksplisit agar tidak salah memilih dimensi. Nilai dasar dinormalisasi sesuai
jenis output; tidak ada pembalikan tanda otomatis untuk memaksa hasil cocok.

Validasi: `base + sum(SHAP)` harus cocok dengan raw margin model; sigmoid jumlah tersebut
harus cocok dengan probabilitas Stunted pada **baris yang sama**. Ini penting karena array 2D
sendiri belum membuktikan bahwa kontribusi menjelaskan kelas positif.''')
code('''def shap_positive_explanation(raw_output, expected_value, frame, classes,
                              positive_margin, positive_probability):
    classes = np.asarray(classes)
    assert classes.shape == (2,) and np.array_equal(classes, [0, 1])
    positive_index = int(np.flatnonzero(classes == 1)[0])
    n_samples, n_features = frame.shape
    is_explanation = isinstance(raw_output, shap.Explanation)
    values = raw_output.values if is_explanation else raw_output
    base = raw_output.base_values if is_explanation else expected_value
    multi_output = False
    if isinstance(values, list):
        if len(values) != len(classes):
            raise ValueError("List SHAP harus berisi satu array untuk setiap kelas.")
        if any(np.asarray(v).shape != (n_samples, n_features) for v in values):
            raise ValueError("Shape array pada list SHAP tidak sesuai sampel/fitur.")
        values = np.asarray(values[positive_index], dtype=float)
        multi_output = True
    else:
        values = np.asarray(values, dtype=float)
        if values.ndim == 3:
            if values.shape != (n_samples, n_features, len(classes)):
                raise ValueError(f"Layout 3D SHAP tidak didukung: {values.shape}; "
                                 "dibutuhkan (n_sample, n_feature, n_class).")
            values = values[:, :, positive_index]
            multi_output = True
        elif values.ndim != 2:
            raise ValueError(f"Dimensi SHAP tidak didukung: {values.shape}")
    if values.shape != (n_samples, n_features):
        raise ValueError(f"Shape SHAP tidak cocok: {values.shape}, input {frame.shape}")
    base = np.asarray(base, dtype=float)
    if multi_output:
        if is_explanation and base.shape == (n_samples, len(classes)):
            base = base[:, positive_index]
        elif base.shape == (len(classes),):
            base = np.full(n_samples, base[positive_index])
        else:
            raise ValueError(f"Base value multikelas ambigu/tidak didukung: {base.shape}")
    else:
        if base.ndim == 0 or base.shape == (1,):
            base = np.full(n_samples, float(base.reshape(-1)[0]))
        elif is_explanation and base.shape == (n_samples,):
            base = base.copy()
        elif is_explanation and base.shape == (n_samples, 1):
            base = base[:, 0]
        else:
            raise ValueError(f"Base value single-output ambigu/tidak didukung: {base.shape}")
    assert values.shape[0] == len(frame)
    assert values.shape[1] == len(FEATURES)
    assert np.isfinite(values).all() and np.isfinite(base).all()
    margin = np.asarray(positive_margin, dtype=float)
    probability = np.asarray(positive_probability, dtype=float)
    assert margin.shape == probability.shape == (n_samples,)
    reconstructed = base + values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, margin, rtol=1e-5, atol=1e-4,
                               err_msg="SHAP tidak merekonstruksi raw margin Stunted.")
    np.testing.assert_allclose(expit(reconstructed), probability, rtol=1e-5, atol=1e-6,
                               err_msg="Orientasi kelas/unit SHAP tidak cocok dengan P(Stunted).")
    return shap.Explanation(values=values, base_values=base,
                            data=frame.to_numpy(copy=True), feature_names=FEATURES.copy())

def shap_explain_frame(explainer, final_model, frame):
    assert frame.columns.tolist() == FEATURES
    if hasattr(final_model, "get_booster"):
        margin = final_model.predict(frame, output_margin=True)
    elif hasattr(final_model, "booster_"):
        margin = final_model.predict(frame, raw_score=True)
    else:
        raise TypeError("Estimator SHAP harus XGBoost atau LightGBM.")
    raw_output = explainer.shap_values(frame, check_additivity=True)
    return shap_positive_explanation(
        raw_output, explainer.expected_value, frame, final_model.classes_,
        margin, final_model.predict_proba(frame)[:, 1])

def shap_for_display(explanation):
    # Label presentasi berubah, nama fitur model dan nilai SHAP tetap utuh.
    return shap.Explanation(values=explanation.values.copy(),
                            base_values=np.array(explanation.base_values, copy=True),
                            data=np.array(explanation.data, copy=True),
                            feature_names=shap_display_names.copy())

shap_explainers, shap_global, shap_importances = {}, {}, {}
for shap_key, shap_estimator in shap_models.items():
    shap_explainer = shap.TreeExplainer(
        shap_estimator, model_output="raw", feature_perturbation="tree_path_dependent")
    shap_explanation = shap_explain_frame(shap_explainer, shap_estimator, X_shap)
    shap_explainers[shap_key] = shap_explainer
    shap_global[shap_key] = shap_explanation
    shap_importance = pd.DataFrame({
        "Feature": FEATURES, "Mean Absolute SHAP": np.abs(shap_explanation.values).mean(axis=0)})
    shap_importance = shap_importance.sort_values(
        ["Mean Absolute SHAP", "Feature"], ascending=[False, True]).reset_index(drop=True)
    shap_importance["Rank"] = shap_importance["Mean Absolute SHAP"].rank(
        ascending=False, method="min").astype(int)
    shap_importances[shap_key] = shap_importance
    print("Validasi kelas positif dan aditivitas lulus:", " — ".join(shap_key))''')
md('''## 10.5 Global Feature Importance: Bar Plot dan Beeswarm
Mean absolute SHAP mengukur besar kontribusi rata-rata tanpa arah, bukan koefisien kausal
atau uji signifikansi. Ranking dihitung dari nilai aktual. Pada beeswarm, posisi kanan berarti
kontribusi menuju Stunted dan kiri menjauhi Stunted, relatif terhadap nilai dasar.
Warna menunjukkan nilai fitur rendah/tinggi; pada Gender warna hanya membedakan kategori
M/F, tidak berarti jenis kelamin memiliki tingkatan numerik.''')
code('''def shap_save_current(filename):
    plt.gcf().savefig(shap_output_dir / filename, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

for shap_key, shap_explanation in shap_global.items():
    shap_algorithm, shap_method = shap_key
    shap_slug = shap_algorithm.lower()
    print("Global SHAP:", shap_algorithm, "—", shap_method)
    display(shap_importances[shap_key][["Feature", "Mean Absolute SHAP"]])
    shap_importances[shap_key].to_csv(shap_output_dir / f"shap_importance_{shap_slug}.csv", index=False)
    shap.plots.bar(shap_for_display(shap_explanation), max_display=len(FEATURES), show=False)
    plt.title(f"SHAP Global — {shap_algorithm} ({shap_method})")
    plt.xlabel("Mean |SHAP| (log-odds Stunted)")
    shap_save_current(f"shap_bar_{shap_slug}.png")
    shap.plots.beeswarm(shap_for_display(shap_explanation), max_display=len(FEATURES), show=False)
    plt.title(f"SHAP Summary — {shap_algorithm} ({shap_method})")
    plt.xlabel("SHAP value (log-odds Stunted)")
    shap_save_current(f"shap_beeswarm_{shap_slug}.png")''')
md('''## 10.6 Feature Effect: Fitur Teratas, Height, dan Age (Month)
Plot menampilkan nilai fitur terhadap kontribusi SHAP pada pengamatan test. Fitur duplikat
hanya diplot sekali. Sebaran ini bukan eksperimen mengubah fitur dengan fitur lain tetap;
interaksi dan keterkaitan fitur dapat memengaruhi polanya. Karena itu grafik tidak membuktikan
pengaruh sebab-akibat. Gender, jika berada di ranking teratas, ditampilkan sebagai dua kategori.''')
code('''for shap_key, shap_explanation in shap_global.items():
    shap_algorithm, shap_method = shap_key
    shap_top_feature = shap_importances[shap_key].iloc[0]["Feature"]
    shap_effect_features = list(dict.fromkeys([shap_top_feature, "Height", "Age (Month)"]))
    for shap_feature in shap_effect_features:
        shap.plots.scatter(shap_explanation[:, shap_feature], show=False)
        plt.title(f"SHAP Effect — {shap_algorithm} ({shap_method}) — {shap_feature}")
        plt.ylabel("Kontribusi SHAP (log-odds Stunted)")
        if shap_feature == "Gender":
            plt.xticks([0, 1], ["M (laki-laki)", "F (perempuan)"])
            plt.xlabel("Jenis kelamin")
        shap_feature_slug = shap_feature.lower().replace(" ", "_").replace("(", "").replace(")", "")
        shap_save_current(f"shap_dependence_{shap_algorithm.lower()}_{shap_feature_slug}.png")''')
md('''## 10.7 Interpretasi Lokal: True Positive dan False Negative
Untuk setiap model, ambil satu TP dan satu FN (jika ada) secara reproducible dari **seluruh
test**, berdasarkan pasangan label aktual dan prediksi tersimpan. Baris lokal boleh berada
di luar sampel global; jika demikian, SHAP dihitung langsung untuk baris itu tanpa fit ulang.
Contoh bersifat ilustratif, bukan ringkasan semua TP/FN atau bukti penyebab kesalahan.

Tabel menampilkan empat fitur, kelas aktual/prediksi, dan probabilitas Stunted. Waterfall
menguraikan raw margin dari nilai dasar menuju output individu. Nilai dasar bukan ambang
klasifikasi. Kontribusi positif tidak selalu berarti prediksi akhirnya Stunted: jumlah seluruh
kontribusi bersama nilai dasar yang menentukan output akhir.''')
code('''shap_local_records = []
for shap_key, shap_estimator in shap_models.items():
    shap_algorithm, shap_method = shap_key
    shap_pred_series = pd.Series(np.asarray(predictions[shap_key]), index=y_test.index)
    shap_proba_series = pd.Series(np.asarray(probabilities[shap_key]), index=y_test.index)
    shap_case_masks = {
        "TP": (y_test == 1) & (shap_pred_series == 1),
        "FN": (y_test == 1) & (shap_pred_series == 0)}
    for shap_case, shap_mask in shap_case_masks.items():
        if not shap_mask.any():
            print(f"{shap_algorithm} ({shap_method}): contoh {shap_case} tidak tersedia.")
            continue
        shap_local_frame = X_test.loc[shap_mask, FEATURES].sample(n=1, random_state=RANDOM_STATE).copy()
        shap_local_index = shap_local_frame.index[0]
        shap_sample_table = shap_local_frame.copy()
        shap_sample_table["Gender meaning"] = shap_sample_table["Gender"].map({0: "M (laki-laki)", 1: "F (perempuan)"})
        shap_sample_table["Actual class"] = y_test.loc[shap_local_frame.index].map({0: "Not Stunted", 1: "Stunted"})
        shap_sample_table["Predicted class"] = shap_pred_series.loc[shap_local_frame.index].map({0: "Not Stunted", 1: "Stunted"})
        shap_sample_table["Predicted probability of Stunted"] = shap_proba_series.loc[shap_local_frame.index]
        print(f"{shap_algorithm} ({shap_method}) — {shap_case} — index test {shap_local_index}")
        display(shap_sample_table)
        shap_local_explanation = shap_explain_frame(
            shap_explainers[shap_key], shap_estimator, shap_local_frame)
        shap.plots.waterfall(shap_for_display(shap_local_explanation)[0], max_display=len(FEATURES), show=False)
        plt.title(f"SHAP Local — {shap_algorithm} ({shap_method}) — {shap_case} — index {shap_local_index}")
        shap_save_current(f"shap_local_{shap_algorithm.lower()}_{shap_case.lower()}.png")
        shap_local_values = pd.Series(shap_local_explanation.values[0], index=FEATURES)
        for shap_direction, shap_contributions in [
                ("menuju Stunted", shap_local_values[shap_local_values > 0].sort_values(ascending=False)),
                ("menjauhi Stunted", shap_local_values[shap_local_values < 0].sort_values())]:
            if not shap_contributions.empty:
                print(f"Kontribusi terbesar {shap_direction}: {shap_contributions.index[0]} "
                      f"({shap_contributions.iloc[0]:+.6f} log-odds).")
        shap_local_records.append({
            "Model": shap_algorithm, "Imbalance Method": shap_method, "Case": shap_case,
            "Test index": shap_local_index, **shap_local_frame.iloc[0].to_dict(),
            "Actual class": int(y_test.loc[shap_local_index]),
            "Predicted class": int(shap_pred_series.loc[shap_local_index]),
            "Probability Stunted": float(shap_proba_series.loc[shap_local_index]),
            "Base value": float(shap_local_explanation.base_values[0]),
            **{f"SHAP {f}": float(shap_local_values[f]) for f in FEATURES}})
pd.DataFrame(shap_local_records).to_csv(shap_output_dir / "shap_local_examples.csv", index=False)''')
md('''## 10.8 Ringkasan Otomatis dan Perbandingan XGBoost–LightGBM
Ringkasan dibuat dari SHAP aktual, termasuk kontribusi Weight dan Gender. Nilai kecil tidak
membuktikan bahwa fitur tidak relevan secara biologis: informasi dapat dibagi antarf fitur yang
saling berkaitan. Mean absolute SHAP juga tidak menjelaskan arah efek; arah dilihat pada
beeswarm/dependence dan penjelasan individu.

Kedua model memakai unit log-odds dan sampel identik, tetapi memiliki fungsi prediksi,
nilai dasar, serta statistik lintasan training yang dapat berbeda akibat class weight/resampling.
Perbandingan lintas model terutama menilai **ranking**, bukan menyatakan angka SHAP salah satu
model lebih benar atau besarnya kontribusi kausal lebih kuat.'''.replace('antarf fitur', 'antarfitur'))
code('''shap_summary_lines = []
for shap_key, shap_importance in shap_importances.items():
    shap_algorithm, shap_method = shap_key
    shap_top = shap_importance.iloc[0]
    shap_tied_top = shap_importance.loc[
        shap_importance["Mean Absolute SHAP"] == shap_top["Mean Absolute SHAP"], "Feature"].tolist()
    shap_summary_lines.extend([
        f"Interpretasi SHAP {shap_algorithm} ({shap_method}):",
        "- Fitur dengan kontribusi global terbesar: " + ", ".join(shap_tied_top) + ".",
        f"- Mean |SHAP| tertinggi: {shap_top['Mean Absolute SHAP']:.6f} log-odds.",
        "- Ranking aktual: " + " > ".join(shap_importance["Feature"]) + " (lihat Rank untuk nilai seri).",
        "- SHAP positif meningkatkan output menuju Stunted relatif terhadap nilai dasar; SHAP negatif menurunkannya."])
    for shap_feature in ["Weight", "Gender"]:
        shap_feature_row = shap_importance.set_index("Feature").loc[shap_feature]
        shap_summary_lines.append(
            f"- {shap_feature}: Mean |SHAP|={shap_feature_row['Mean Absolute SHAP']:.6f}, "
            f"rank={int(shap_feature_row['Rank'])}; ini kontribusi terhadap model, bukan bukti kausal/signifikansi.")

shap_comparison_parts = []
for shap_key, shap_importance in shap_importances.items():
    shap_algorithm = shap_key[0]
    shap_comparison_parts.append(shap_importance.set_index("Feature").rename(columns={
        "Mean Absolute SHAP": f"Mean |SHAP| {shap_algorithm}", "Rank": f"Rank {shap_algorithm}"}))
shap_comparison = pd.concat(shap_comparison_parts, axis=1).reindex(FEATURES).reset_index()
display(shap_comparison)
shap_same_ranks = bool((shap_comparison["Rank XGBoost"] == shap_comparison["Rank LightGBM"]).all())
shap_rank_agreement = shap_comparison["Rank XGBoost"].corr(shap_comparison["Rank LightGBM"], method="spearman")
shap_summary_lines.append(
    "Perbandingan: ranking kedua algoritma identik pada sampel ini." if shap_same_ranks else
    "Perbandingan: terdapat perbedaan ranking antaralgoritma pada sampel ini; lihat tabel per fitur.")
shap_summary_lines.append(
    f"Korelasi Spearman ranking empat fitur: {shap_rank_agreement:.6f}. "
    "Ini deskripsi kesesuaian ranking, bukan uji validitas atau signifikansi (NaN berarti ranking konstan).")
shap_summary_lines.append("Tidak ada interpretasi yang dinyatakan lebih benar berdasarkan perbandingan ini.")
print("\\n".join(shap_summary_lines))
shap_comparison.to_csv(shap_output_dir / "shap_comparison.csv", index=False)
(shap_output_dir / "shap_interpretation.txt").write_text("\\n".join(shap_summary_lines), encoding="utf-8")''')
md('''## 10.9 Metadata, Validasi Akhir, dan Keterbatasan
Metadata mencatat model/metode terpilih, hyperparameter, versi library, seed, indeks sampel,
unit output, fingerprint model, dan contoh lokal. Hanya file berawalan `shap_` yang ditulis;
artefak eksperimen lama tidak ditimpa. Pemeriksaan akhir membandingkan results, seluruh
prediksi/probabilitas tersimpan, fingerprint delapan model, dan hash artefak lama.

**Catatan metodologis:** Karena status `Height for Age` secara antropometri dibentuk berdasarkan
usia, jenis kelamin, dan tinggi/panjang badan, dominasi fitur-fitur tersebut dalam SHAP merupakan
pola yang secara konseptual dapat diharapkan. SHAP dalam penelitian ini menjelaskan perilaku
model klasifikasi dan bukan menunjukkan hubungan sebab-akibat maupun faktor risiko longitudinal.
Dominasi aktual tetap harus dibaca dari hasil, bukan diasumsikan sebelumnya.

Fitur antropometri saling berkaitan; atribusi bergantung pada model serta asumsi penanganan
ketergantungan fitur. Tree-path-dependent menggunakan statistik pohon, yang dapat dipengaruhi
pembobotan/resampling training. Class weighting juga dapat memengaruhi kalibrasi probabilitas;
probabilitas yang ditampilkan adalah output model, bukan estimasi risiko klinis tervalidasi.
Sampling 2.000 baris dan contoh TP/FN tunggal tidak menggantikan analisis seluruh populasi atau
validasi eksternal. Pemilihan model dari F1 test dilakukan secara post-hoc untuk interpretasi;
grafik ini tidak boleh dipakai untuk tuning ulang lalu mengklaim test tetap independen.

**Batasan teknis:** output dengan layout SHAP yang tidak terdokumentasi akan ditolak, bukan
ditebak. Jika TreeExplainer tidak kompatibel dengan versi model, periksa versi dan format model;
tidak ada fallback yang melatih model pengganti atau mengabaikan kegagalan aditivitas.''')
code('''assert_original_data()
pd.testing.assert_frame_equal(X_test, shap_X_test_snapshot, check_exact=True)
pd.testing.assert_series_equal(y_test, shap_y_test_snapshot, check_exact=True)
pd.testing.assert_frame_equal(results, shap_results_snapshot, check_exact=True)
assert set(predictions) == set(shap_predictions_snapshot)
assert set(probabilities) == set(shap_probabilities_snapshot)
assert set(searches) == set(shap_model_hashes_before)
for shap_key in searches:
    np.testing.assert_array_equal(predictions[shap_key], shap_predictions_snapshot[shap_key])
    np.testing.assert_array_equal(probabilities[shap_key], shap_probabilities_snapshot[shap_key])
    assert shap_model_fingerprint(shap_final_model(searches[shap_key].best_estimator_)) == shap_model_hashes_before[shap_key]
for shap_filename, shap_digest in shap_existing_artifact_hashes.items():
    assert hashlib.sha256((shap_output_dir / shap_filename).read_bytes()).hexdigest() == shap_digest
shap_metadata = {
    "status": "completed", "selection": "highest test F1 per algorithm, interpretation only",
    "tie_break": ["Recall descending", "ROC-AUC descending", "method name ascending"],
    "models": shap_selected[["Model", "Imbalance Method", "F1", "Recall", "ROC-AUC", "Best Hyperparameters"]].to_dict("records"),
    "features": FEATURES.copy(), "gender_encoding": {"0": "M", "1": "F"},
    "positive_class": 1, "sample_size": len(X_shap), "test_size": len(X_test),
    "random_state": RANDOM_STATE, "sample_test_indices": X_shap.index.tolist(),
    "local_examples": [{"model": r["Model"], "method": r["Imbalance Method"],
                        "case": r["Case"], "test_index": r["Test index"]} for r in shap_local_records],
    "model_output": "raw log-odds of Stunted", "feature_perturbation": "tree_path_dependent",
    "packages": {name: version(name) for name in ["shap", "xgboost", "lightgbm", "numpy", "pandas", "scikit-learn"]},
    "model_sha256": {" — ".join(key): digest for key, digest in shap_model_hashes_before.items()},
    "sample_sha256": hashlib.sha256(pd.util.hash_pandas_object(X_shap, index=True).values.tobytes()).hexdigest(),
    "original_artifact_sha256": shap_existing_artifact_hashes,
    "integrity_checks": "passed: test, results, predictions, probabilities, 8 models, original artifacts",
    "additivity_checks": "passed: raw margin and positive-class probability",
    "training_or_resampling_performed": False}
(shap_output_dir / "shap_metadata.json").write_text(
    json.dumps(shap_metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("Analisis SHAP selesai. Test, hasil 8 eksperimen, model, dan artefak lama tetap identik.")
print("Artefak SHAP:", shap_output_dir)''')

assert nb.cells[:len(original)] == original
nbformat.validate(nb)
nbformat.write(nb, PATH)
saved = nbformat.read(PATH, as_version=4)
assert saved.cells[:len(original)] == original
assert old_artifacts == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in artifact_dir.iterdir() if p.is_file() and not p.name.startswith('shap_')}
print(f'Appended {len(nb.cells)-len(original)} cells; {len(original)} original cells and artifacts unchanged.')
