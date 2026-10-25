"""Revisi master yang ada; pertahankan seluruh kode EDA dan cleaning."""
from pathlib import Path
import copy
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
source = nbf.read(ROOT / 'notebooks/master_notebook.ipynb', as_version=4)
nb = copy.deepcopy(source)
nb.cells = copy.deepcopy(source.cells[:48])
nb.cells[0].source = '''# Master Notebook — Prediksi Stunting Balita dengan XGBoost & LightGBM

Revisi eksperimen skripsi: **Baseline vs Class Weight vs SMOTE-NC vs SMOTE-NC + ENN**,
masing-masing pada XGBoost dan LightGBM (8 kombinasi).

Alur: loading → EDA → cleaning → empat fitur mentah dan encoding → split stratified 80:20
→ GridSearchCV dengan resampling di dalam training fold → evaluasi pada test set asli.
Semua skenario memperoleh grid yang sama per algoritma, 5 fold identik, dan scoring F1
kelas **Stunted = 1**, dengan `RANDOM_STATE = 42`.

**Audit metodologi master:** oversampling sebelum CV menyebabkan kebocoran informasi antar-fold;
baseline tanpa tuning membuat perbandingan tidak setara. Revisi memperbaiki kedua hal tersebut
dan memakai resampler yang mengenali Gender sebagai nominal. Kesimpulan dihasilkan dari eksekusi.
Snapshot membuktikan integritas data, sedangkan penempatan sampler dalam pipeline mencegah
kebocoran resampling antar-fold; snapshot saja tidak cukup membuktikan semua bentuk bebas leakage.

**Batasan yang dipertahankan:** cleaning Weight menggunakan asumsi pemulihan tanggal `day.month`;
validitas asumsi perlu didukung sumber data. Penghapusan baris karena `Weight for Height` kosong
tidak diperlukan untuk empat predictor, tetapi aturan master dipertahankan untuk konsistensi kohort.
EDA seluruh data dipertahankan sebagai deskripsi, tidak digunakan untuk memilih parameter.
Height, Age, dan Gender berkaitan langsung dengan pembentukan label antropometri: hasil ini adalah
klasifikasi status saat pengukuran, bukan bukti prediksi risiko stunting di masa depan.
Split baris tidak membuktikan independensi individu bila ada pengukuran berulang tanpa ID subjek.
'''
nb.cells[2].source = nb.cells[2].source.replace('from imblearn.combine import SMOTEENN', '''from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import ParameterGrid
from joblib import Memory, parallel_config
from threadpoolctl import threadpool_limits
from pathlib import Path
from tempfile import TemporaryDirectory
from importlib.metadata import version
from IPython.display import display
import hashlib
import json
import sys''').replace('import time                     # pengukuran waktu training model (Bagian 5 & 8)', 'import time                     # pengukuran waktu tuning dan refit')
nb.cells[2].source = nb.cells[2].source.replace('DATA_PATH = "../data/raw/Overall Data.xlsx"     # sumber data mentah', '''# Mendukung kernel yang dimulai dari root proyek atau folder notebooks.
PROJECT_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                     if (p / "data/raw/Overall Data.xlsx").is_file()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Jalankan notebook dari folder proyek atau notebooks.")
DATA_PATH = PROJECT_ROOT / "data/raw/Overall Data.xlsx"
RUN_DIR = PROJECT_ROOT / "results/master_v2"
RUN_DIR.mkdir(parents=True, exist_ok=True)
GRID_N_JOBS = 4
MODEL_N_JOBS = 1
print("Python:", sys.version)
dependency_versions = {name: version(name) for name in [
    "pandas", "numpy", "scikit-learn", "imbalanced-learn", "xgboost",
    "lightgbm", "matplotlib", "seaborn", "openpyxl", "joblib"]}
print(dependency_versions)
print("SHA256 dataset:", hashlib.sha256(DATA_PATH.read_bytes()).hexdigest())''')
nb.cells[16].source = '''### Catatan: Keseimbangan Kelas

Distribusi target menunjukkan mild imbalance. Manfaat Class Weight, SMOTE-NC,
dan SMOTE-NC + ENN harus diuji terhadap Baseline dengan prosedur tuning setara;
penanganan imbalance tidak diasumsikan selalu meningkatkan performa.'''

def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))

def code(s):
    nb.cells.append(nbf.v4.new_code_cell(s.strip()))

md('''# 4. Preprocessing & Data Splitting
## 4.1 Feature Selection dan Pencegahan Data Leakage
Gunakan hanya Gender, Age (Month), Weight, dan Height. Target serta seluruh kolom turunan
dan No. tidak dimasukkan sebagai predictor. Tidak ada seleksi fitur berdasarkan skor test.''')
code('''TARGET_COL = "Height for Age"
FEATURES = ["Gender", "Age (Month)", "Weight", "Height"]
FORBIDDEN = {"Weight for Age", "Z-Score W/A", "Z-Score H/A",
             "Weight for Height", "Z-Score W/H", "No.", TARGET_COL}
assert not FORBIDDEN.intersection(" ".join(c.split()) for c in FEATURES)
X = df_clean.loc[:, FEATURES].copy()
assert list(X.columns) == FEATURES
print("Predictor:", X.columns.tolist())''')
md('''## 4.2 Encoding Fitur Kategorikal dan Target
Mapping deterministik: M → 0, F → 1; Not Stunted → 0, Stunted → 1.
Encoding tidak dipelajari dari test. Age (Month), Weight, dan Height tetap numerik kontinu.
Skala asli master dipertahankan. Ini perlu dicatat karena metode berbasis tetangga sensitif
terhadap skala; kode Gender 0/1 tetap ditandai nominal pada SMOTE-NC.''')
code('''X["Gender"] = X["Gender"].map({"M": 0, "F": 1})
y = df_clean[TARGET_COL].map({"Not Stunted": 0, "Stunted": 1})
assert not X.isna().any().any(), "Ada missing/kategori Gender tidak dikenal."
assert not y.isna().any(), "Ada target tidak dikenal."
assert set(X["Gender"].unique()) == {0, 1}
assert set(y.unique()) == {0, 1}
assert all(pd.api.types.is_numeric_dtype(X[c]) for c in FEATURES)
assert np.isfinite(X.to_numpy(dtype=float)).all()
display(X.dtypes)
display(y.value_counts().sort_index())''')
md('''## 4.3 Train-Test Split 80:20 Stratified
Split dilakukan sekali, sebelum resampling. Snapshot training dan test disimpan;
ukuran test diturunkan dari split, tidak di-hard-code.''')
code('''X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
X_train_snapshot = X_train.copy(deep=True)
y_train_snapshot = y_train.copy(deep=True)
X_test_snapshot = X_test.copy(deep=True)
y_test_snapshot = y_test.copy(deep=True)
n_test_original = len(X_test)
categorical_indices = [X_train.columns.get_loc("Gender")]
assert categorical_indices == [0]
assert X_train.index.intersection(X_test.index).empty
assert X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index)
print("Training:", X_train.shape, "Test:", X_test.shape)
print("Categorical indices:", categorical_indices)''')
md('''## 4.4 Distribusi Kelas Training dan Test
Distribusi test tetap berasal dari kohort setelah cleaning, tanpa penyeimbangan kelas.''')
code(source.cells[56].source)
md('''## 4.5 Perhitungan scale_pos_weight
Bobot XGBoost dihitung hanya dari y_train asli, kemudian ditetapkan untuk seluruh kandidat
dan fold sesuai protokol penelitian. Tidak dituning dan tidak dihitung dari y_test.
Nilai > 1 memperbesar bobot loss kelas Stunted; pada mild imbalance nilainya umumnya tidak besar.
LightGBM menggunakan `is_unbalance=True` pada skenario Class Weight.''')
code('''negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
assert negative_count > 0 and positive_count > 0
scale_pos_weight = negative_count / positive_count
print("negative_count (Not Stunted):", negative_count)
print("positive_count (Stunted):", positive_count)
print("scale_pos_weight:", scale_pos_weight)''')
md('''## 4.6 Validasi Anti-Data-Leakage
Helper berikut dipanggil setelah demonstrasi, setelah setiap GridSearchCV, dan sebelum evaluasi.
Training asli juga tidak boleh ditimpa. Pemeriksaan indeks menjamin pemisahan baris.''')
code('''def assert_original_data():
    assert X_test.equals(X_test_snapshot), "X_test berubah!"
    assert y_test.equals(y_test_snapshot), "y_test berubah!"
    assert len(X_test) == len(y_test) == n_test_original
    assert X_train.equals(X_train_snapshot), "X_train asli berubah!"
    assert y_train.equals(y_train_snapshot), "y_train asli berubah!"
    assert list(X_train.columns) == list(X_test.columns) == FEATURES
    assert X_train.index.intersection(X_test.index).empty

assert_original_data()
print("Integritas training dan test tervalidasi.")''')
md('''# 5. Definisi Skenario Eksperimen
## 5.1 Baseline
Tanpa resampling dan class weight: XGBoost `scale_pos_weight=1`, LightGBM
`is_unbalance=False`, `class_weight=None`. Baseline tetap memperoleh tuning penuh.
## 5.2 Built-in Class Weight
Tanpa resampling. XGBoost memakai rasio y_train; LightGBM memakai `is_unbalance=True`.
Tidak memberikan `scale_pos_weight` pada LightGBM sehingga tidak terjadi kombinasi bobot.
## 5.3 SMOTE-NC
SMOTENC dengan Gender kategorikal, k=5, sampling_strategy='auto', random_state=42.
Tanpa class weighting.
## 5.4 SMOTE-NC + ENN
Urutan eksplisit SMOTENC → EditedNearestNeighbours(n_neighbors=3) → classifier.
Tanpa class weighting. ENN memakai default `sampling_strategy='auto'` (kelas selain minoritas)
dan `kind_sel='all'`; ketika kelas sama banyak, pemilihan kelas mengikuti aturan library.
ENN bukan metrik khusus data campuran dan memakai jarak pada Gender 0/1 serta numerik skala asli.
Ini merupakan keterbatasan yang perlu dibedakan dari perlakuan nominal Gender dalam SMOTE-NC.

Rujukan API: [SMOTENC](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html),
[Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html),
[ENN](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.EditedNearestNeighbours.html).
Pipeline melewati resampling saat predict; validation fold dan test tidak di-resample.''')
code('''METHODS = ["Baseline", "Class Weight", "SMOTE-NC", "SMOTE-NC + ENN"]
ALGORITHMS = ["XGBoost", "LightGBM"]

def make_smotenc():
    return SMOTENC(categorical_features=categorical_indices,
                   random_state=RANDOM_STATE, k_neighbors=5, sampling_strategy="auto")

def make_pipeline(algorithm, method, memory=None):
    assert algorithm in ALGORITHMS and method in METHODS
    weighted = method == "Class Weight"
    if algorithm == "XGBoost":
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight if weighted else 1,
            random_state=RANDOM_STATE, n_jobs=MODEL_N_JOBS,
            eval_metric="logloss", tree_method="hist", verbosity=0)
    else:
        model = LGBMClassifier(
            is_unbalance=weighted, class_weight=None,
            random_state=RANDOM_STATE, n_jobs=MODEL_N_JOBS, verbosity=-1)
    steps = []
    if method in ["SMOTE-NC", "SMOTE-NC + ENN"]:
        steps.append(("smotenc", make_smotenc()))
    if method == "SMOTE-NC + ENN":
        steps.append(("enn", EditedNearestNeighbours(n_neighbors=3)))
    steps.append(("model", model))
    return ImbPipeline(steps, memory=memory)

for algorithm in ALGORITHMS:
    for method in METHODS:
        pipe = make_pipeline(algorithm, method)
        print(algorithm, "—", method, ":", list(pipe.named_steps))''')
md('''# 6. Analisis Resampling
## 6.1 Distribusi Training Sebelum Resampling
Salinan berawalan `demo_` hanya untuk analisis. Tidak digunakan sebagai input GridSearchCV.''')
code('''demo_X_original = X_train.copy(deep=True)
demo_y_original = y_train.copy(deep=True)
show_dist("Original", demo_y_original)
print("Shape original:", demo_X_original.shape)''')
md('''## 6.2 SMOTE-NC
Oversampling hanya pada salinan training. Helper mempertahankan nama dan urutan fitur
jika library mengembalikan ndarray, dan memeriksa kategori tanpa pembulatan/coercion.''')
code('''def checked_features(values):
    if isinstance(values, pd.DataFrame):
        assert values.columns.tolist() == FEATURES
        frame = values.copy()
    else:
        assert np.asarray(values).ndim == 2 and np.asarray(values).shape[1] == len(FEATURES)
        frame = pd.DataFrame(values, columns=FEATURES)
    assert set(frame["Gender"].unique()).issubset({0, 1}), "Gender sintetis tidak valid!"
    assert np.isfinite(frame.to_numpy(dtype=float)).all()
    return frame

demo_X_nc_raw, demo_y_nc = make_smotenc().fit_resample(demo_X_original, demo_y_original)
demo_X_nc = checked_features(demo_X_nc_raw)
show_dist("SMOTE-NC", demo_y_nc)
print("Shape SMOTE-NC:", demo_X_nc.shape)''')
md('''## 6.3 SMOTE-NC + ENN
ENN diterapkan setelah SMOTE-NC, sesuai urutan pipeline. ENN menghapus instance yang
tidak konsisten dengan tetangganya; jumlah akhir kelas bisa tidak tepat 50:50 dan itu bukan error.''')
code('''demo_enn = EditedNearestNeighbours(n_neighbors=3)
demo_X_ncenn_raw, demo_y_ncenn = demo_enn.fit_resample(demo_X_nc.copy(), demo_y_nc.copy())
demo_X_ncenn = checked_features(demo_X_ncenn_raw)
show_dist("SMOTE-NC + ENN", demo_y_ncenn)
print("Shape SMOTE-NC + ENN:", demo_X_ncenn.shape)
print("Aturan kelas ENN aktual:", demo_enn.sampling_strategy_)''')
md('''## 6.4 Perbandingan Distribusi Kelas
Class Weight tidak mengubah jumlah instance training; distribusinya sama dengan Original.''')
code('''resampling_distribution = pd.DataFrame({
    name: pd.Series(target).value_counts().reindex([0, 1], fill_value=0)
    for name, target in [("Original", demo_y_original), ("SMOTE-NC", demo_y_nc),
                         ("SMOTE-NC + ENN", demo_y_ncenn)]}).T
resampling_distribution.columns = LABELS
display(resampling_distribution)
ax = resampling_distribution.plot.bar(rot=0, color=["#4477AA", "#EE6677"])
ax.set_ylabel("Jumlah instance training")
ax.set_title("Distribusi kelas sebelum dan setelah resampling")
plt.tight_layout()
plt.show()''')
md('''## 6.5 Validasi Fitur Kategorikal Setelah Synthetic Oversampling
Nilai Gender harus tetap {0, 1}, termasuk baris sintetis. Tidak dilakukan pembulatan nilai
pecahan untuk menyembunyikan kesalahan encoding.''')
code('''for name, frame in [("SMOTE-NC", demo_X_nc), ("SMOTE-NC + ENN", demo_X_ncenn)]:
    checked_features(frame)
    print("Gender unique setelah", name, ":", sorted(frame["Gender"].unique().tolist()))''')
md('''## 6.6 Validasi Test Set Tetap Utuh
Demonstrasi selesai; seluruh input asli harus identik dengan snapshot.''')
code('''assert_original_data()
print("Test set tetap utuh:", n_test_original, "baris.")''')
md('''# 7. Hyperparameter Tuning
## 7.1 StratifiedKFold dan Grid yang Setara
StratifiedKFold(5, shuffle=True, random_state=42) dimaterialisasi sekali agar indeks fold
identik pada delapan eksperimen. GridSearchCV selalu menerima X_train/y_train asli.
Scoring `f1` menggunakan kelas positif 1 pada target biner; refit memilih mean CV F1 tertinggi.
Tidak ada tuning threshold atau early stopping menggunakan test.

Masing-masing grid memiliki 81 kandidat × 5 fold, total 3.240 fit CV + 8 refit.
Untuk efisiensi, cache pipeline disimpan sementara dan dikosongkan setiap eksperimen.
Cache hanya menggunakan hasil fit training fold yang identik; tidak menyatukan fold atau
memakai hasil demo. Empat worker thread grid dan satu thread model diterapkan sama pada semua skenario.
Backend threading menghindari masalah cleanup proses loky pada lingkungan Windows ini.
Waktu tuning adalah wall-clock seluruh search termasuk refit/cache; waktu refit dicatat terpisah.
Waktu tetap dipengaruhi beban mesin dan bukan benchmark statistik.''')
code('''skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_splits = list(skf.split(X_train, y_train))
validation_seen = np.zeros(len(X_train), dtype=int)
for train_idx, valid_idx in cv_splits:
    assert not np.intersect1d(train_idx, valid_idx).size
    assert y_train.iloc[train_idx].value_counts().min() > 5
    validation_seen[valid_idx] += 1
assert np.all(validation_seen == 1)
xgb_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__gamma": [0, 0.1, 0.3],
}
lgbm_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__num_leaves": [15, 31, 63],
}
PARAM_GRIDS = {"XGBoost": xgb_grid, "LightGBM": lgbm_grid}
assert all(len(ParameterGrid(grid)) == 81 for grid in PARAM_GRIDS.values())
searches, tuning_times = {}, {}

def tune_experiment(algorithm, method):
    assert_original_data()
    key = (algorithm, method)
    print("Mulai:", algorithm, "—", method, flush=True)
    with TemporaryDirectory(prefix="cv_cache_", dir=RUN_DIR) as cache_dir:
        pipeline = make_pipeline(algorithm, method, memory=Memory(cache_dir, verbose=0))
        search = GridSearchCV(
            pipeline, PARAM_GRIDS[algorithm], scoring="f1", cv=cv_splits,
            n_jobs=GRID_N_JOBS, pre_dispatch=GRID_N_JOBS, refit=True,
            error_score="raise", return_train_score=False, verbose=1)
        start = time.perf_counter()
        # Worker berbasis thread tidak membuat file memmap/proses loky.
        with threadpool_limits(limits=1), parallel_config(backend="threading"):
            search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        # Prediksi tidak memerlukan cache; lepaskan lokasi sementara dari estimator.
        search.best_estimator_.memory = None
        search.estimator.memory = None
    assert_original_data()
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()
    searches[key] = search
    tuning_times[key] = elapsed
    print("Best CV F1:", search.best_score_)
    print("Best hyperparameters:", search.best_params_)
    print(f"Tuning: {elapsed:.2f} s; refit: {search.refit_time_:.2f} s")
    slug = algorithm.lower() + "_" + str(METHODS.index(method) + 1)
    pd.DataFrame(search.cv_results_).to_csv(RUN_DIR / f"cv_{slug}.csv", index=False)
''')
for i, (algorithm, method) in enumerate(((a, m) for a in ['XGBoost','LightGBM'] for m in ['Baseline','Class Weight','SMOTE-NC','SMOTE-NC + ENN']), 2):
    md(f'''## 7.{i} {algorithm} — {method}
Tuning memakai fold dan grid tetap untuk {algorithm}; hanya strategi imbalance yang berubah.
Input tetap training asli. Evaluasi test dilakukan setelah seluruh tuning selesai.''')
    code(f'tune_experiment({algorithm!r}, {method!r})')
md('''# 8. Model Evaluation & Comparison
## 8.1 Tabel Hasil Semua Model
Semua model terbaik telah di-refit oleh GridSearchCV. Helper tunggal hanya melakukan prediksi,
tanpa fit ulang, pada test asli. Precision, Recall, dan F1 dihitung untuk Stunted (1),
ROC-AUC dari `predict_proba(X_test)[:, 1]`. Prediksi kelas memakai aturan default classifier
yang sama antar-skenario; tidak ada optimasi threshold. Tabel menyorot maksimum F1 dan Recall;
ranking deskriptif tidak dipakai untuk tuning ulang pada test.''')
code('''assert_original_data()
assert len(searches) == 8
predictions, probabilities = {}, {}

def evaluate_on_test(algorithm, method, search):
    model = search.best_estimator_
    assert np.array_equal(model.classes_, [0, 1])
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    assert len(pred) == len(proba) == n_test_original
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    assert tn + fp + fn + tp == n_test_original
    predictions[(algorithm, method)] = pred
    probabilities[(algorithm, method)] = proba
    return {
        "Model": algorithm, "Imbalance Method": method,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, pos_label=POS_LABEL, zero_division=0),
        "Recall": recall_score(y_test, pred, pos_label=POS_LABEL, zero_division=0),
        "F1": f1_score(y_test, pred, pos_label=POS_LABEL, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Best CV F1": search.best_score_, "Tuning Time": tuning_times[(algorithm, method)],
        "Refit Time": search.refit_time_, "Best Hyperparameters": search.best_params_,
    }

results = pd.DataFrame([evaluate_on_test(a, m, searches[(a, m)])
                        for a in ALGORITHMS for m in METHODS])
metric_columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Best CV F1"]
table_columns = ["Model", "Imbalance Method", "Accuracy", "Precision", "Recall", "F1",
                 "ROC-AUC", "TN", "FP", "FN", "TP", "Best CV F1", "Tuning Time"]
display(results[table_columns].style.format({c: "{:.4f}" for c in metric_columns} |
        {"Tuning Time": "{:.2f}"}).highlight_max(subset=["F1", "Recall"], color="#c6e5c4"))
display(results[["Model", "Imbalance Method", "Best Hyperparameters"]])
display(results.sort_values(["F1", "Recall"], ascending=False)[table_columns])
assert_original_data()''')
md('''## 8.2 Confusion Matrix
Baris = aktual, kolom = prediksi, label berurutan Not Stunted lalu Stunted.
Orientasi setiap matriks: TN FP / FN TP. Skala warna sama untuk delapan panel.''')
code('''fig, axes = plt.subplots(2, 4, figsize=(20, 9))
cm_max = max(confusion_matrix(y_test, p, labels=[0, 1]).max() for p in predictions.values())
for ax, (key, pred) in zip(axes.ravel(), predictions.items()):
    sns.heatmap(confusion_matrix(y_test, pred, labels=[0, 1]), annot=True, fmt="d",
                cmap="Blues", vmin=0, vmax=cm_max, cbar=False,
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_title(" — ".join(key))
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
plt.tight_layout()
fig.savefig(RUN_DIR / "confusion_matrices.png", dpi=200, bbox_inches="tight")
plt.show()''')
md('''## 8.3 ROC Curve
Dua panel dengan sumbu dan warna skenario identik agar empat kurva tiap algoritma mudah dibaca.''')
code('''COLORS = dict(zip(METHODS, ["#4477AA", "#EE6677", "#228833", "#AA3377"]))
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
for ax, algorithm in zip(axes, ALGORITHMS):
    for method in METHODS:
        proba = probabilities[(algorithm, method)]
        fpr, tpr, _ = roc_curve(y_test, proba, pos_label=POS_LABEL)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=COLORS[method], label=f"{method} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), title=algorithm,
           xlabel="False Positive Rate", ylabel="True Positive Rate (Recall Stunted)")
    ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
fig.savefig(RUN_DIR / "roc_curves.png", dpi=200, bbox_inches="tight")
plt.show()''')
for subsection, metric, title in [(4, 'Recall', 'Recall Stunted'), (5,'F1','F1-Score Stunted'), (6,'ROC-AUC','ROC-AUC')]:
    md(f'''## 8.{subsection} Perbandingan {title}
Delapan kombinasi ditampilkan dengan sumbu 0–1 agar perbedaan tidak dibesar-besarkan.''')
    if subsection == 4:
        code('''def plot_comparison(metric, ylabel, filename, unit_interval=True):
    values = results.pivot(index="Model", columns="Imbalance Method", values=metric)
    values = values.reindex(index=ALGORITHMS, columns=METHODS)
    ax = values.plot.bar(color=[COLORS[m] for m in METHODS], figsize=(11, 5), rot=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Algoritma")
    ax.set_title("Perbandingan " + ylabel)
    if unit_interval:
        ax.set_ylim(0, 1.08)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f" if unit_interval else "%.1f", fontsize=8, padding=2)
    ax.legend(title="Strategi imbalance", loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4)
    plt.tight_layout()
    ax.figure.savefig(RUN_DIR / filename, dpi=200, bbox_inches="tight")
    plt.show()''')
    code(f'plot_comparison({metric!r}, {title!r}, {metric.lower().replace("-", "_") + ".png"!r})')
md('''## 8.7 Training/Tuning Time
Waktu tuning mencakup search dan refit; refit adalah waktu training final pipeline terbaik
pada training lengkap, termasuk resampling bila ada. Cache fold dapat mempercepat kandidat
berikutnya; seluruh skenario memakai kebijakan cache yang sama dan dimulai dengan cache kosong.''')
code('''display(results[["Model", "Imbalance Method", "Tuning Time", "Refit Time"]])
plot_comparison("Tuning Time", "Waktu tuning (detik)", "tuning_time.png", unit_interval=False)
plot_comparison("Refit Time", "Waktu refit (detik)", "refit_time.png", unit_interval=False)''')
md('''## 8.8 Interpretasi Hasil Otomatis
Prioritas interpretasi: Recall Stunted, F1, ROC-AUC, Precision, lalu Accuracy. False Negative
adalah balita berlabel Stunted yang diprediksi Not Stunted. F1 tetap menjadi kriteria pemilihan
hyperparameter yang ditetapkan sebelum evaluasi. Semua selisih di bawah dihitung dari hasil aktual.
Selisih positif disebut peningkatan numerik, bukan bukti signifikansi statistik atau manfaat klinis.
Satu split tidak cukup membuktikan kebutuhan resampling secara umum; tidak ada uji inferensial
atau validasi eksternal dalam protokol ini. Peringkat test bersifat deskriptif dan tidak mengubah model.''')
code('''interpretation_lines = []
def report(text):
    interpretation_lines.append(text)
    print(text)

for number, metric in [(1, "Recall"), (2, "F1"), (3, "ROC-AUC")]:
    best_value = results[metric].max()
    winners = results.loc[results[metric] == best_value]
    names = "; ".join(winners["Model"] + " — " + winners["Imbalance Method"])
    report(f"{number}. {metric} tertinggi: {names} ({best_value:.6f}).")

indexed = results.set_index(["Model", "Imbalance Method"])
comparison_rows = []
for number, candidate, reference in [
        (4, "Class Weight", "Baseline"), (5, "SMOTE-NC", "Baseline"),
        (6, "SMOTE-NC + ENN", "SMOTE-NC"), (7, "SMOTE-NC", "Class Weight")]:
    report(f"{number}. {candidate} dibanding {reference}:")
    for algorithm in ALGORITHMS:
        delta = indexed.loc[(algorithm, candidate), metric_columns] - indexed.loc[(algorithm, reference), metric_columns]
        recall_status = "naik" if delta["Recall"] > 0 else "turun" if delta["Recall"] < 0 else "tetap"
        f1_status = "naik" if delta["F1"] > 0 else "turun" if delta["F1"] < 0 else "tetap"
        report(f"   {algorithm}: Recall {recall_status} ({delta['Recall']:+.6f}); "
               f"F1 {f1_status} ({delta['F1']:+.6f}); Precision {delta['Precision']:+.6f}.")
        comparison_rows.append({"Model": algorithm, "Candidate": candidate,
                                "Reference": reference, **{f"Delta {k}": float(v) for k,v in delta.items()}})
report("7 (lanjutan). Efektivitas algorithm-level vs data-level dibaca dari selisih SMOTE-NC "
       "terhadap Class Weight di atas: positif mengunggulkan SMOTE-NC pada metrik tersebut, "
       "negatif mengunggulkan Class Weight; bila Recall/F1 berbeda arah, tidak ada pemenang tunggal.")
report("8. Dampak ENN:")
for algorithm in ALGORITHMS:
    plain = indexed.loc[(algorithm, "SMOTE-NC")]
    enn = indexed.loc[(algorithm, "SMOTE-NC + ENN")]
    report(f"   {algorithm}: Delta Recall={enn['Recall']-plain['Recall']:+.6f}, "
           f"Delta F1={enn['F1']-plain['F1']:+.6f}, Delta FN={int(enn['FN']-plain['FN']):+d}.")
report("   Kebermaknaan statistik/praktis belum dapat ditetapkan dari satu split dan tanpa ambang manfaat yang ditentukan sebelumnya.")
report("9. Apakah resampling diperlukan pada dataset ini?")
for algorithm in ALGORITHMS:
    group = results[results["Model"] == algorithm]
    no_sampling = group[group["Imbalance Method"].isin(["Baseline", "Class Weight"])]
    sampling = group[group["Imbalance Method"].isin(["SMOTE-NC", "SMOTE-NC + ENN"])]
    nr = no_sampling.sort_values(["F1", "Recall"], ascending=False).iloc[0]
    sr = sampling.sort_values(["F1", "Recall"], ascending=False).iloc[0]
    verdict = "ada keuntungan F1 numerik dari resampling" if sr["F1"] > nr["F1"] else "resampling tidak diperlukan untuk mencapai F1 terbaik pada split ini"
    report(f"   {algorithm}: {verdict}; {sr['Imbalance Method']} vs {nr['Imbalance Method']}: "
           f"Delta F1={sr['F1']-nr['F1']:+.6f}, Delta Recall={sr['Recall']-nr['Recall']:+.6f}.")
report("10. Trade-off terhadap Baseline:")
for algorithm in ALGORITHMS:
    baseline = indexed.loc[(algorithm, "Baseline")]
    for method in METHODS[1:]:
        row = indexed.loc[(algorithm, method)]
        dr, dp = row["Recall"]-baseline["Recall"], row["Precision"]-baseline["Precision"]
        status = "trade-off Recall–Precision" if dr * dp < 0 else "tidak ada perubahan berlawanan arah Recall–Precision"
        report(f"   {algorithm} — {method}: {status}; Delta Recall={dr:+.6f}, Delta Precision={dp:+.6f}.")
comparison_deltas = pd.DataFrame(comparison_rows)
display(comparison_deltas)
assert_original_data()''')
md('''### Artefak dan Reproducibility
Simpan metrik, hasil CV seluruh kandidat, parameter terbaik, prediksi test, manifest versi,
dan grafik ke `results/master_v2`. Artefak dihitung dari eksekusi saat ini; tidak digunakan
sebagai input training. Master asli tetap utuh. Versi package dan hash data membantu audit,
tetapi reproduksi lintas versi library/mesin dapat menghasilkan perbedaan numerik kecil.''')
code('''results.to_csv(RUN_DIR / "comparison.csv", index=False)
comparison_deltas.to_csv(RUN_DIR / "comparison_deltas.csv", index=False)
resampling_distribution.to_csv(RUN_DIR / "resampling_distribution.csv")
best_parameters = {f"{a} — {m}": searches[(a, m)].best_params_ for a in ALGORITHMS for m in METHODS}
(RUN_DIR / "best_parameters.json").write_text(json.dumps(best_parameters, indent=2), encoding="utf-8")
test_predictions = pd.DataFrame({"row_index": y_test.index, "y_true": y_test.to_numpy()})
for (a, m), pred in predictions.items():
    test_predictions[f"{a} — {m} prediction"] = pred
    test_predictions[f"{a} — {m} probability"] = probabilities[(a, m)]
test_predictions.to_csv(RUN_DIR / "test_predictions.csv", index=False)
manifest = {
    "python": sys.version, "packages": dependency_versions,
    "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
    "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
    "raw_rows": len(df), "clean_rows": len(df_clean), "train_rows": len(X_train),
    "test_rows": len(X_test), "features": FEATURES, "positive_label": POS_LABEL,
    "scale_pos_weight": scale_pos_weight, "scoring": "f1", "cv_folds": 5,
    "grid_n_jobs": GRID_N_JOBS, "model_n_jobs": MODEL_N_JOBS,
    "xgb_grid": xgb_grid, "lgbm_grid": lgbm_grid,
    "train_indices": X_train.index.tolist(), "test_indices": X_test.index.tolist(),
}
(RUN_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
(RUN_DIR / "interpretation.txt").write_text("\\n".join(interpretation_lines), encoding="utf-8")
assert_original_data()
print("Selesai: 8 eksperimen. Test tetap identik:", n_test_original, "baris.")
print("Artefak:", RUN_DIR)''')

for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
        cell.metadata = {}
        compile(cell.source, '<notebook>', 'exec')
assert all('SMOTEENN' not in cell.source and 'SMOTE-ENN' not in cell.source for cell in nb.cells)
for idx in range(3, 48):
    if source.cells[idx].cell_type == 'code':
        assert nb.cells[idx].source == source.cells[idx].source
out = ROOT / 'notebooks/master_notebook_v2_smote_nc_classweight.ipynb'
nbf.validate(nb)
nbf.write(nb, out)
print(out, len(nb.cells), 'cells')
