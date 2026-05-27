#  Analisis Faktor Sosial Ekonomi BPS terhadap IPM Menggunakan MLP Regressor
 
> Proyek analisis prediktif berbasis Machine Learning untuk memahami dan memprediksi Indeks Pembangunan Manusia (IPM) kabupaten/kota di Indonesia menggunakan algoritma Multi-Layer Perceptron (MLP) Regressor.
 
---
 
##  Deskripsi Proyek
 
Proyek ini bertujuan untuk menganalisis hubungan antara **faktor-faktor sosial ekonomi** dengan **Indeks Pembangunan Manusia (IPM)** di tingkat kabupaten/kota di Indonesia menggunakan pendekatan *machine learning* berbasis jaringan saraf tiruan.
 
Model yang digunakan adalah **Multi-Layer Perceptron (MLP) Regressor** — salah satu bentuk Artificial Neural Network (ANN) yang mampu mempelajari pola non-linear kompleks antar variabel. Selain membangun model prediktif, proyek ini juga:
 
- Mengevaluasi performa model dalam mengenali pola hubungan non-linear antar indikator sosial ekonomi.
- Mengidentifikasi faktor-faktor yang paling berpengaruh terhadap capaian IPM melalui analisis **Permutation Importance**.
- Membandingkan performa model **Baseline MLP** dengan **MLP hasil Hyperparameter Tuning**.
---
 
##  Dataset
 
| Atribut | Keterangan |
|---|---|
| **Sumber** | Badan Pusat Statistik (BPS) Indonesia |
| **Judul** | Faktor Sosial Ekonomi BPS |
| **Jumlah Observasi** | 514 baris (mewakili 514 kabupaten/kota di Indonesia) |
| **Jumlah Kolom** | 12 kolom |
| **Tipe Data** | Campuran (2 kolom identitas bertipe `object`, 10 kolom numerik kontinu) |
| **Missing Values** | Tidak ada (data lengkap) |
 
###  Deskripsi Variabel
 
| No | Nama Variabel | Tipe | Keterangan |
|---|---|---|---|
| 1 | `Provinsi` | Identitas | Nama provinsi wilayah |
| 2 | `Kab/Kota` | Identitas | Nama kabupaten/kota |
| 3 | `Persentase Penduduk Miskin (P0)` | Fitur | Persentase penduduk di bawah garis kemiskinan (%) |
| 4 | `Rata-rata Lama Sekolah Penduduk 15+` | Fitur | Rata-rata lama sekolah penduduk usia 15 tahun ke atas (tahun) |
| 5 | `Pengeluaran per Kapita Disesuaikan` | Fitur | Pengeluaran per kapita yang disesuaikan (ribu rupiah/orang/tahun) |
| 6 | `Umur Harapan Hidup` | Fitur | Rata-rata usia harapan hidup saat lahir (tahun) |
| 7 | `Akses Sanitasi Layak` | Fitur | Persentase rumah tangga dengan akses sanitasi layak (%) |
| 8 | `Akses Air Minum Layak` | Fitur | Persentase rumah tangga dengan akses air minum layak (%) |
| 9 | `Tingkat Pengangguran Terbuka` | Fitur | Persentase angkatan kerja yang menganggur (%) |
| 10 | `Tingkat Partisipasi Angkatan Kerja` | Fitur | Persentase penduduk usia kerja yang aktif di pasar kerja (%) |
| 11 | `PDRB atas Dasar Harga Konstan` | Fitur | Produk Domestik Regional Bruto menurut pengeluaran (rupiah) |
| 12 | `Indeks Pembangunan Manusia` | **Target (y)** | Skor komposit pembangunan manusia (skala 0–100) |
 
---
 
##  Tools & Library
 
| Kategori | Library / Tools |
|---|---|
| **Manipulasi Data** | `pandas`, `numpy` |
| **Visualisasi** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` |
| **Model Utama** | `MLPRegressor` (sklearn.neural_network) |
| **Preprocessing** | `StandardScaler`, `FunctionTransformer`, `ColumnTransformer`, `Pipeline` |
| **Evaluasi** | `r2_score`, `mean_squared_error`, `mean_absolute_error`, `cross_val_score`, `learning_curve` |
| **Feature Importance** | `permutation_importance` |
| **Hyperparameter Tuning** | `RandomizedSearchCV` |
| **Distribusi Sampling** | `scipy.stats.loguniform` |
| **Environment** | Google Colab |
 
---
 
##  Alur Pengerjaan
 
### 1.  Pembacaan & Eksplorasi Data (EDA)
- Memuat dataset dari Google Drive ke dalam DataFrame.
- Mengecek dimensi, tipe data, dan missing values.
- Menghitung **statistik deskriptif** (mean, std, min, max, skewness, kurtosis).
- Temuan: PDRB dan Pengeluaran per Kapita memiliki standar deviasi sangat besar → indikasi ketimpangan ekonomi antar daerah.
### 2.  Analisis Korelasi
- Membuat **matriks korelasi Pearson** antar seluruh variabel numerik.
- Mengidentifikasi korelasi tertinggi terhadap IPM:
  - **Positif kuat:** Rata-rata Lama Sekolah, Pengeluaran per Kapita, Umur Harapan Hidup, Akses Sanitasi.
  - **Negatif kuat:** Persentase Penduduk Miskin (P0), Tingkat Pengangguran Terbuka.
### 3.  Deteksi Outlier & Multikolinearitas
- Deteksi outlier menggunakan **Z-score** (threshold |z| > 3).
  - Variabel PDRB dan Pengeluaran per Kapita mengandung outlier positif (daerah dengan ekonomi tinggi).
- Penghitungan **VIF (Variance Inflation Factor)** untuk mendeteksi multikolinearitas.
  - Beberapa variabel ekonomi memiliki VIF tinggi, namun tidak menjadi kendala serius untuk MLP.
### 4.  Visualisasi Distribusi & Hubungan
- **Histogram** untuk distribusi tiap variabel utama.
- **Scatter plot** untuk melihat pola hubungan masing-masing fitur terhadap IPM.
### 5.  Preprocessing & Feature Engineering
- Penghapusan duplikat berdasarkan kolom identitas wilayah.
- **Transformasi Logaritmik (`log1p`)** pada variabel PDRB dan Pengeluaran per Kapita (menangani right-skewed distribution dan outlier ekstrim).
- **Standardisasi (`StandardScaler`)** pada seluruh fitur numerik.
- Seluruh tahap preprocessing diintegrasikan ke dalam **sklearn Pipeline**.
### 6.  Pembagian Data
- Split data: **80% training (411 baris)** dan **20% testing (103 baris)**.
- Strategi: `shuffle=True`, `random_state=42` untuk reproduktifitas.
### 7.  Pemodelan MLP
Dibangun dua model:
 
**Model 1 — Baseline MLP:**
  - Arsitektur: 2 hidden layers `(64, 32)`
  - Aktivasi: ReLU
  - Optimizer: Adam
  - Regularisasi L2 (alpha): `1e-4`
  - Early Stopping aktif (15 iterasi tanpa perbaikan)
**Model 2 — Best MLP (Hyperparameter Tuning):**
  - Pencarian hyperparameter menggunakan `RandomizedSearchCV` (20 iterasi, 5-fold CV).
  - Parameter yang dituning: `hidden_layer_sizes`, `activation`, `alpha`, `learning_rate_init`, `batch_size`.
  - Model terbaik dipilih berdasarkan skor R² tertinggi pada cross-validation.
### 8.  Evaluasi Model
- Metrik evaluasi: **R², RMSE, MAE** pada data train dan test.
- **5-Fold Cross-Validation** untuk mengukur kestabilan model.
- **Learning Curve** untuk mendeteksi overfitting/underfitting.
- **Permutation Importance** untuk mengidentifikasi fitur paling berpengaruh.
- **Residual Analysis** untuk memeriksa pola error.
---
 
##  Hasil
 
### Perbandingan Performa Model
 
| Metrik | Baseline MLP | Best MLP (Tuned) |
|---|---|---|
| **R² Train** | ~0.967 | ~0.973 |
| **R² Test** | 0.919 | **0.955** |
| **RMSE Test** | 2.317 | **1.736** |
| **MAE Test** | 1.510 | **1.065** |
| **CV Mean R² (5-Fold)** | 0.870 | **0.961** |
| **CV Std R²** | ~0.040 | **~0.028** |
 
> Model hasil tuning mengungguli baseline secara konsisten: R² test naik +3,6 poin, RMSE turun ≈25%, dan MAE turun ≈30%.
 
###  Fitur Paling Berpengaruh terhadap IPM
Berdasarkan hasil **Permutation Importance**:
 
1.  **Rata-rata Lama Sekolah Penduduk 15+** — faktor dominan tertinggi
2.  **Pengeluaran per Kapita Disesuaikan** — representasi daya beli masyarakat
3.  **Umur Harapan Hidup** — indikator kesehatan populasi
### Temuan Utama EDA
- IPM, Umur Harapan Hidup, dan Rata-rata Lama Sekolah memiliki distribusi relatif normal dan merata antar daerah.
- PDRB dan Pengeluaran per Kapita berdistribusi sangat *right-skewed*, mencerminkan ketimpangan ekonomi antar wilayah yang signifikan.
- Tingkat Kemiskinan berkorelasi **negatif kuat** dengan IPM, menegaskan bahwa kemiskinan menjadi penghambat utama pembangunan manusia.
---
 
##  Kesimpulan
 
1. **Model MLP Regressor terbukti efektif** dalam memprediksi IPM berdasarkan faktor sosial ekonomi daerah, dengan akurasi tinggi (R² ≈ 0.955) dan generalisasi yang baik.
2. **Hyperparameter tuning secara signifikan meningkatkan performa:** R² naik dari 0.919 → 0.955, RMSE turun dari 2.317 → 1.736, dan stabilitas cross-validation meningkat (CV R² 0.87 → 0.96).
3. **Model tidak mengalami overfitting**, dibuktikan oleh gap R² train-test yang kecil (≈0.018) serta learning curve yang konvergen dengan stabil.
4. **Tiga faktor utama penentu IPM:** Rata-rata Lama Sekolah, Pengeluaran per Kapita, dan Umur Harapan Hidup — konsisten dengan teori pembangunan manusia yang menempatkan pendidikan, ekonomi, dan kesehatan sebagai komponen kunci.
5. **Sebaran residual acak** di sekitar nol mengindikasikan tidak adanya bias sistematis pada prediksi model.
---
 
##  Saran Pengembangan
 
- Tambahkan data panel multi-tahun untuk analisis prediksi tren IPM jangka panjang.
- Perluas fitur dengan indikator tambahan (fasilitas kesehatan, akses digital, kemiskinan ekstrem).
- Bandingkan dengan algoritma lain: SVR, Gradient Boosting, atau Gaussian Process Regressor.
- Gunakan Bayesian Optimization atau Keras-Tuner untuk tuning yang lebih efisien.
- Tambahkan metode interpretasi **SHAP** atau **LIME** agar temuan model lebih mudah dikomunikasikan kepada pemangku kebijakan.
---
 
##  Struktur File
 
```
├── Analisis_Faktor_Sosisal_Ekonomi_BPS_terhadap_IPM_menggunakan_MLP.ipynb
├── Faktor_Sosial_Ekonomi_BPS.csv
└── README.md
```
 
---
 
*Dataset bersumber dari Badan Pusat Statistik (BPS) Republik Indonesia.*
