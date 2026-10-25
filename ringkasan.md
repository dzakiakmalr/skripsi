# Ringkasan Eksperimen — Prediksi Stunting Balita Jeneponto

## 1. Alur Eksperimen

| Notebook | Tahap | Output utama |
|---|---|---|
| `01_eda` | Eksplorasi data | Konfirmasi 40.071 baris × 11 kolom; temukan kerusakan `Weight` |
| `02_cleaning` | Perbaikan & cleaning | `data/processed/cleaned_data.csv` |
| `03_preprocessing` | Feature selection, encoding, split | `X/y`, lalu split 80:20 stratified |
| `04_baseline` | XGBoost & LightGBM default (tanpa resampling) | `baseline_results.csv` |
| `05_smoteenn` | SMOTE-ENN hanya pada train set | `X_train_resampled` & `y_train_resampled` |
| `06_tuned` | GridSearchCV + StratifiedKFold, model final | `smoteenn_results.csv`, `best_params.json` |
| `07_comparison` | Perbandingan 4 kombinasi + plot | conf. matrix, ROC, bar chart waktu |

**Rantai data (angka aktual, bukan estimasi):**
- Raw 40.071 → cleaning **40.064** (7 baris dihapus: 1 outlier `Weight=91`, 2 missing `Age (Month)`, 5 missing `Weight for Height`).
- Split 80:20 stratified (`random_state=42`): **train 32.051 / test 8.013**; test set = Not Stunted 4.773, Stunted 3.240.
- SMOTE-ENN hanya pada train: 32.051 → **34.756** (Not Stunted 17.218 / Stunted 17.538; mendekati seimbang).
- Fitur: hanya 4 mentah (`Gender`, `Age (Month)`, `Weight`, `Height`); kolom turunan Z-score/WHO dibuang (anti-leakage).

### Rincian Baris yang Dihapus / Dikoreksi (Data Quality)

**Konvensi nomor baris:** kolom `No.` di dataset = identitas baris (1-based, `No. == index + 1`). Karena file Excel punya header di baris 1, **baris fisik Excel = `No.` + 1**.

#### a. Outlier `Weight` = 91 (dihapus — 1 baris)

| No. | Baris Excel | Gender | Age (Month) | Weight | Height | Weight for Height |
|---|---|---|---|---|---|---|
| 26337 | 26338 | F | 28.0 | 91 | 84.0 | NaN |

**Alasan:** 91 kg mustahil untuk balita 28 bulan (tinggi 84 cm); nilai aslinya tidak bisa dipastikan. Hanya 1 baris (0,0025%) → di-drop, bukan ditebak. Catatan: baris ini *juga* termasuk salah satu baris yang `Weight for Height`-nya kosong.

#### b. Missing `Age (Month)` (dihapus — 2 baris)

| No. | Baris Excel | Gender | Age (Month) | Weight | Height | Weight for Height |
|---|---|---|---|---|---|---|
| 38602 | 38603 | M | NaN | 5 | 60.0 | Thin |
| 38716 | 38717 | F | NaN | 3.5 (datetime→decimal) | 48.0 | Normal |

**Alasan:** `Age (Month)` fitur kunci, tapi hanya 2 baris (0,005%). Imputasi menyuntik nilai sintetis yang tidak berasal dari pengukuran → drop.

#### c. Missing `Weight for Height` (dihapus — 5 baris)

| No. | Baris Excel | Gender | Age (Month) | Weight | Height | Weight for Height |
|---|---|---|---|---|---|---|
| 619 | 620 | F | 58.0 | 8.9 | 99.0 | NaN |
| 2336 | 2337 | F | 30.0 | 6 | 83.0 | NaN |
| 5959 | 5960 | M | 27.0 | 17 | 79.0 | NaN |
| 14395 | 14396 | F | 58.0 | 13.9 (datetime→decimal) | 57.0 | NaN |
| 26337 | 26338 | F | 28.0 | 91 (outlier) | 84.0 | NaN |

**Alasan:** `Weight for Height` adalah fitur turunan yang sesuai metodologi justru dibuang saat feature selection (anti-leakage), sehingga mengimputasinya tidak memberi manfaat untuk pemodelan → drop.

> Total baris **dihapus = 7** (No. 26337 dihitung sekali): `26337`, `38602`, `38716`, `619`, `2336`, `5959`, `14395` → 40.071 − 7 = **40.064**.

#### d. Koreksi satuan gram → kg (diubah, BUKAN dihapus — 5 baris)

Semua berusia **0 bulan (bayi baru lahir)** — menguatkan bahwa 2800–3100 adalah **gram** (berat lahir normal), jadi dibagi 1000:

| No. | Baris Excel | Gender | Age (Month) | Nilai mentah (gram) | → kg |
|---|---|---|---|---|---|
| 39449 | 39450 | M | 0 | 3085 | 3.085 |
| 39991 | 39992 | F | 0 | 3100 | 3.1 |
| 40013 | 40014 | M | 0 | 2900 | 2.9 |
| 40028 | 40029 | M | 0 | 3000 | 3.0 |
| 40035 | 40036 | M | 0 | 2800 | 2.8 |

> **Catatan perbaikan bug Weight:** rekonstruksi awal memakai `month.day` (salah arah). Setelah koreksi ke `day.month` (Excel membaca nilai asli dengan format DD.MM, umum di locale Indonesia), 20.158 nilai `Weight` berubah sehingga seluruh hasil downstream (notebook 03–07) di-*rerun*. Semua metrik di bawah adalah angka final yang benar.

## 2. Hasil Tiap Skenario

Kelas positif = **Stunted (1)**. Metrik numeric ada di CSV (`results/metrics/`), di sini dibulatkan 4 desimal.

### Skenario 1 — Baseline (default params, tanpa resampling)

| Model | Accuracy | Precision | **Recall (Stunted)** | **F1 (Stunted)** | ROC-AUC | TN / FP / FN / TP |
|---|---|---|---|---|---|---|
| XGBoost | 0.9592 | 0.9582 | 0.9401 | 0.9491 | 0.9929 | 4640 / 133 / 194 / 3046 |
| LightGBM | 0.9609 | 0.9575 | 0.9454 | 0.9514 | 0.9930 | 4637 / 136 / 177 / 3063 |

### Skenario 2 — SMOTE-ENN + Hyperparameter Tuning (GridSearchCV, scoring=F1)

| Model | Accuracy | Precision | **Recall (Stunted)** | **F1 (Stunted)** | ROC-AUC | TN / FP / FN / TP |
|---|---|---|---|---|---|---|
| XGBoost | 0.9561 | 0.9460 | 0.9454 | 0.9457 | 0.9908 | 4598 / 175 / 177 / 3063 |
| LightGBM | 0.9557 | 0.9415 | 0.9494 | 0.9454 | 0.9909 | 4582 / 191 / 164 / 3076 |

**Best params** (disimpan di `results/metrics/best_params.json`):
- XGBoost: `n_estimators=300, max_depth=7, learning_rate=0.2, gamma=0` → **CV F1 mean 0.9965**.
- LightGBM: `n_estimators=300, max_depth=7, learning_rate=0.1, num_leaves=63` → **CV F1 mean 0.9969**.

**Waktu training** (pengukuran satu-kali, mesin lokal; **indikatif & noisy**):
XGBoost baseline ≈1.53s, XGBoost+SMOTE-ENN ≈0.32s, LightGBM baseline ≈1.43s, LightGBM+SMOTE-ENN ≈0.28s.

## 3. Poin Analisis untuk Bab IV

**a. Fokus pada Recall & F1 kelas Stunted (accuracy paradox).**
Accuracy semua model ≈0.955–0.961 terlihat tinggi, tapi tolok ukur utamanya adalah Recall & F1 kelas Stunted. Recall baseline: XGBoost 0.9401, LightGBM 0.9454. Setelah SMOTE-ENN, recall naik menjadi XGBoost 0.9454 dan LightGBM 0.9494 — FN turun dari 194→177 (XGBoost) dan 177→164 (LightGBM), artinya lebih sedikit balita stunting yang terlewat. Nilai recall/F1 inilah yang dibandingkan, bukan accuracy.

**b. Dampak SMOTE-ENN: recall naik tipis, precision/accuracy turun tipis.**
Ini trade-off precision–recall yang wajar: setelah SMOTE-ENN model lebih "berani" memprediksi *Stunted*, sehingga recall naik tetapi false positive bertambah → precision turun (XGBoost 0.958→0.946; LightGBM 0.958→0.942) dan accuracy turun sedikit (0.959→0.956; 0.961→0.956). F1 kelas Stunted cenderung datar (0.949→0.946; 0.951→0.945). Jadi perbaikan recall-nya *modest*, bukan lompatan besar — konsisten dengan karakter data yang hanya *mild imbalance* (40:60).

**c. GAP CV vs Test = sinyal penting.**
`best_score_` CV (≈0.996) jauh di atas recall test (≈0.945–0.949). Ini bukti kuat bahwa **CV pada data sintetis bersifat optimistik** — sampel SMOTE terlalu "mudah" saat divalidasi silang, sehingga skor CV tidak menjamin generalisasi ke populasi nyata.

**c1. Batasan metodologi tuning (SMOTE-ENN diterapkan sekali sebelum CV).**
GridSearchCV dijalankan di atas training set yang sudah di-SMOTE-ENN sekali (notebook 05), bukan menerapkan ulang SMOTE-ENN di dalam tiap fold CV. Konsekuensinya, sampel sintetis dan "induknya" bisa terpecah antar fold training/validasi, sehingga skor F1 CV berpotensi sedikit optimistik. Ini **bukan leakage** ke test set (test tetap murni, dicek `assert`), dan **tidak menyentuh validitas metrik final** yang dilaporkan — hanya memengaruhi ketepatan pemilihan hyperparameter. Dicatat eksplisit sebagai **batasan metodologi** (juga ditambahkan sebagai catatan di `06_tuned_models.ipynb`); alternatif yang lebih ketat adalah membungkus SMOTE-ENN + classifier dalam `imblearn.Pipeline` agar resampling diterapkan ulang per-fold.

**d. XGBoost vs LightGBM.**
Performa sangat dekat; LightGBM sedikit unggul di recall/F1/accuracy pada kedua skenario. Pembedanya ke efisiensi: LightGBM+SMOTE-ENN (≈0.28s) lebih cepat dari XGBoost+SMOTE-ENN (≈0.32s). Namun angka waktu ini **belum reliabel** (pengukuran satu-kali; lihat catatan di bawah).

**e. Yang belum dilakukan — jangan diklaim ada hasilnya:**
- **Feature importance** (XGBoost vs LightGBM) → `[BELUM ADA HASIL]`.
- **Uji signifikansi statistik** antar model (mis. McNemar/paired test) → `[BELUM ADA HASIL]`.
- **Tuning untuk skenario baseline (tanpa SMOTE-ENN)** → baseline masih pakai *default* params, sehingga perbandingannya belum sepenuhnya "apple-to-apple" (SMOTE-ENN diuntungkan karena di-tuning). Ini keterbatasan desain yang harus diakui. `[BELUM ADA HASIL]`
- **Benchmark waktu training yang reliabel** (multi-run + warm-up + median) → `[BELUM ADA HASIL]`.
- **Alternatif resampling lain** (SMOTE murni, RandomOver, dsb.) untuk memperkuat argumen → `[BELUM ADA HASIL]`.

**Saran prioritas untuk melengkapi Bab IV** (diskusikan dulu dgn pembimbing): (1) jalankan Skenario baseline yang juga di-tuning agar perbandingan adil; (2) tambah uji McNemar; (3) ukur waktu training dengan prosedur yang benar jika ingin klaim "LightGBM lebih efisien".