# Car Price Preprocessing and Modeling

Repositori ini memuat implementasi *end-to-end machine learning pipeline* untuk memprediksi harga mobil bekas (*used car price prediction*). Proyek ini berfokus pada tahapan komprehensif penyiapan data (*data cleaning*, penanganan pencilan, standardisasi, dan encoding), analisis eksploratif (EDA & ExDA), perbandingan performa algoritma regresi (LARS, Linear Regression, dan Gradient Boosting Regressor), hingga serialisasi model siap pakai (*model persistence*).

## Ringkasan Proyek
- **Tipe Masalah:** Regresi Nilai Kontinu (*Tabular Data*)
- **Dataset:** Car Price Prediction (~2.500 baris total)
- **Target Variabel:** `Price`
- **Algoritma yang Diuji:**
  - Least Angle Regression (LARS)
  - Linear Regression (OLS)
  - Gradient Boosting Regressor (GBR)
- **Metrik Evaluasi:** Mean Absolute Error (MAE), Mean Squared Error (MSE) / Root Mean Squared Error (RMSE), $R^2$ Score
- **Format Penyimpanan Model:** `.joblib` & `.pkl`

## Pustaka & Lingkungan Komputasi
- **Bahasa:** Python 3.x
- **Lingkungan Kerja:** Jupyter Notebook
- **Pengolahan & Analisis Data:** Pandas, NumPy
- **Visualisasi:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn
- **Serialisasi Objek:** Joblib, Pickle

## Struktur Repositori
```text
car-price-preprocessing-and-modeling/
├── dataset/
│   ├── car_price.csv              # Data mentah
│   ├── train.csv                  # Data latih (2.000 baris, 10 kolom)
│   ├── test.csv                   # Data uji tanpa label Price (500 baris)
│   └── test_ground_truth.csv      # Kunci target Price data uji
├── notebooks/
│   └── prediksihargamobil.ipynb   # Notebook alur kerja lengkap
├── models/
│   ├── gbr_car_model.joblib       # Serialisasi model GBR (Joblib)
│   └── gbr_car_model.pkl          # Serialisasi model GBR (Pickle)
└── README.md
```

## Hasil Evaluasi dan Penjelasan Metrik Validasi

Berdasarkan hasil pengujian model pada *validation set*, diperoleh nilai metrik evaluasi sebagai berikut:

| Metrik | Nilai |
|---|---|
| MAE (Mean Absolute Error) | $25.161,45$ |
| RMSE (Root Mean Squared Error) | $28.828,00$ |
| MSE (Mean Squared Error) — turunan: $\text{RMSE}^2$ | $\approx 831.053.584$ ($\approx 8{,}31 \times 10^{8}$) |
| $R^2$ Score (Koefisien Determinasi) | $-0,0976$ |

> Nilai MSE tidak dilaporkan langsung oleh notebook; angka di atas dihitung dari $\text{RMSE}^2$.

---

## Landasan Teori Metrik Evaluasi

Penjelasan metrik pada bagian ini disandarkan pada lima jurnal rujukan. Keterangan halaman mengikuti nomor halaman cetak jurnal; khusus jurnal [4] yang tidak mencantumkan nomor halaman cetak pada naskahnya, halaman mengacu pada urutan halaman berkas PDF.

### Daftar Rujukan

| Kode | Rujukan |
|---|---|
| [1] | Sholihah, S. M., Aditiya, N. Y., Evani, E. S., & Maghfiroh, S. (2023). Konsep Uji Asumsi Klasik pada Regresi Linier Berganda. *Jurnal Riset Akuntansi Soedirman (JRAS)*, 2(2), 102–110. |
| [2] | Arifuddin, D., Kusrini, & Kusnawi (2025). Comparison of the Performance of Multiple Linear Regression Algorithms and Multi Layer Perceptron Neural Networks in Predicting Drug Sales. *MALCOM: Indonesian Journal of Machine Learning and Computer Science*, 5(2), 722–737. https://doi.org/10.57152/malcom.v5i2.1952 |
| [3] | Agustina, A., Putri, Y. N., Nisah, H., Raihanah, S. D., Najib, A., & Tundjungsari, V. (2026). Analisis Prediksi Harga Berlian Menggunakan Metode Regresi Linier. *JATI (Jurnal Mahasiswa Teknik Informatika)*, 10(1), 1825–1832. |
| [4] | Inabah, S. F., Putri, I. A., & Mutiarachim, A. (2026). Analisis Komparatif Multiple Linear Regression dan Random Forest dalam Prediksi Performa Akademik Siswa. *Digital Business Intelligence – Jurnal Ilmiah UNTAG Semarang*, 2(1). |
| [5] | Putra, A. G. W., Yudistira, N., & Soebroto, A. A. (2026). Analisis Performa Model TimeMixer terhadap Eskalasi Dimensi Data dan Studi Komparatif pada Prediksi Data Time Series. *Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer (J-PTIIK)*, 10(3). |

### Dasar Bersama: Residual sebagai Unsur Pembentuk Semua Metrik

MAE, MSE/RMSE, dan $R^2$ seluruhnya dibangun dari selisih antara nilai aktual dan nilai prediksi, yaitu *residual* $e_i = y_i - \hat{y}_i$.

- Jurnal [1] (hlm. 105, langkah uji Kolmogorov-Smirnov) merumuskan residual secara eksplisit sebagai selisih nilai aktual dan nilai prediksi ($Y - \hat{Y}$).
- Jurnal [2] (hlm. 724, Persamaan 1) menyebut suku galat ($\epsilon$) sebagai selisih antara nilai aktual dan nilai prediksi yang mencerminkan variasi target yang tidak dapat dijelaskan oleh variabel prediktor.
- Jurnal [3] (hlm. 1828) memberikan definisi yang sama dan menambahkan bahwa koefisien regresi diperoleh dengan meminimalkan total galat kuadrat.

**Implikasi bagi proyek ini:** seluruh metrik pada tabel di atas adalah ringkasan statistik dari residual model GBR pada *validation set*, sehingga kualitas penanganan data dan fitur pada tahap *preprocessing* langsung tercermin pada nilainya.

---

### 1. MAE (*Mean Absolute Error*)

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\left|y_i - \hat{y}_i\right|$$

**Definisi.** MAE adalah rata-rata kesalahan absolut antara nilai prediksi dan nilai aktual.
- [3] (hlm. 1827, Subbab 2.4, dan rumus pada Tabel 1, hlm. 1829) mendefinisikan MAE sebagai rata-rata kesalahan absolut.
- [5] (hlm. 4, Persamaan 8) menegaskan bahwa MAE mengukur rata-rata besarnya kesalahan absolut **tanpa mempertimbangkan arah kesalahan** (selisih positif dan negatif diperlakukan sama).

**Argumen penguat.**
1. **Satuan sama dengan target.** [3] (hlm. 1830) menafsirkan MAE 1.200,69 sebagai prediksi yang rata-rata menyimpang sekitar 1.200,69 satuan harga dari nilai sebenarnya. Dengan logika yang sama, MAE **$25.161,45$** pada proyek ini berarti rata-rata estimasi harga mobil meleset sekitar angka tersebut dari harga aslinya. Pada [5] (hlm. 5, Subbab 2.5), evaluasi dilakukan setelah data didenormalisasi agar galat terbaca dalam satuan asli; prinsip ini relevan karena `Price` pada proyek ini juga dibaca dalam satuan aslinya.
2. **Lebih tahan terhadap pencilan.** [5] (hlm. 5, awal Subbab 2.4 lanjutan dan Subbab 2.5) menyatakan MAE bersifat *robust* karena selisih tidak dikuadratkan sehingga kurang sensitif terhadap nilai ekstrem, dan cocok bila distribusi galat diasumsikan mengikuti distribusi Laplace. Hal ini menjadi alasan MAE dijadikan metrik pendamping RMSE pada data harga yang cenderung memiliki ekor panjang.
3. **Kaitan dengan uji statistik klasik.** Uji Glejser dalam [1] (hlm. 107) menggunakan **nilai mutlak residual** sebagai variabel yang diuji. Ini menunjukkan bahwa besaran absolut residual merupakan ukuran standar untuk menilai sebaran galat model, selaras dengan prinsip yang digunakan MAE.
4. **Besarnya MAE harus dibaca terhadap skala target.** [2] (hlm. 732) mengingatkan bahwa besarnya galat bergantung pada skala target. Tanpa mengetahui rentang harga mobil, angka $25.161,45$ belum dapat dinilai "besar" atau "kecil" secara mandiri; pembanding yang tepat adalah galat model *baseline* yang memprediksi rata-rata (lihat Bagian 3).

---

### 2. MSE dan RMSE (*Mean Squared Error* dan *Root Mean Squared Error*)

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2 \qquad\qquad \text{RMSE} = \sqrt{\text{MSE}}$$

**Definisi.** MSE adalah rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual; RMSE adalah akar kuadrat dari MSE sehingga kembali ke satuan target.
- [3] (hlm. 1827 dan Tabel 1 hlm. 1829) mendefinisikan MSE sebagai rata-rata kuadrat selisih prediksi dan menegaskan bahwa metrik ini **lebih sensitif terhadap kesalahan besar**.
- [2] (hlm. 725 dan 731) menyebut MSE sebagai ukuran kesalahan prediksi, mendefinisikannya sebagai rata-rata kuadrat *error*, dan menurunkan RMSE sebagai akar dari MSE untuk dibaca dalam satuan data (misalnya RMSE 150,148 dibaca sebagai deviasi sekitar 150 unit analgesik per hari).
- [5] (hlm. 5, Persamaan 9) mendefinisikan MSE sebagai rata-rata kuadrat selisih nilai observasi dan prediksi.

**Argumen penguat.**
1. **Penalti lebih besar untuk galat ekstrem.** [5] (hlm. 5, Subbab 2.5) menjelaskan bahwa MSE memberi penalti lebih besar pada kesalahan prediksi yang signifikan. [3] (hlm. 1830) juga menyebut kesalahan besar lebih berpengaruh pada MSE sehingga metrik ini sensitif terhadap pencilan. Inilah dasar interpretasi bahwa **RMSE ($28.828,00$) > MAE ($25.161,45$)** menandakan adanya sebagian prediksi dengan galat cukup ekstrem. Selisih keduanya tidak besar (rasio MAE/RMSE ≈ 0,87), yang berarti galat relatif merata, bukan hanya akibat beberapa titik ekstrem, sehingga seluruh prediksi memang kurang akurat.
2. **Dasar teoretis MSE.** [5] (hlm. 5) menyatakan bahwa meminimalkan MSE ekuivalen secara matematis dengan memaksimalkan *likelihood* data bila galat berdistribusi normal (Gaussian), serta dapat diuraikan menjadi komponen *bias* dan *variance*. Dekomposisi ini membantu menelusuri apakah galat berasal dari ketidakmampuan menangkap rata-rata atau variabilitas data. Hal ini mendukung rekomendasi diagnosis lanjutan pada bagian Kesimpulan.
3. **Asumsi normalitas residual.** [1] (hlm. 104–105) menjelaskan bahwa uji normalitas ditujukan untuk menilai apakah residual mengikuti distribusi normal, dan pelanggaran asumsi ini menyebabkan nilai prediksi bias dan tidak konsisten (hlm. 105). Karena kegunaan MSE sebagai kriteria optimal bertumpu pada galat berdistribusi normal [5], pemeriksaan distribusi residual (histogram atau Q-Q plot) pada model OLS/LARS di proyek ini layak dilakukan.
4. **Ketergantungan pada skala.** [2] (hlm. 732) menegaskan besarnya MSE bergantung pada skala target. Pada proyek ini MSE $\approx 8{,}31 \times 10^{8}$ tampak sangat besar karena harga mobil berskala puluhan ribu; karena itu RMSE lebih mudah dibaca daripada MSE.
5. **Contoh pembacaan dari studi lain.** [3] (hlm. 1830) menyajikan MSE 2.582.825,77 pada rentang harga berlian yang mencapai puluhan ribu dan menilainya wajar karena disertai $R^2$ tinggi (0,944). Perbedaan kuncinya: pada proyek ini galat besar **disertai** $R^2$ negatif, sehingga kesimpulannya berbalik.

---

### 3. $R^2$ Score (*Koefisien Determinasi*)

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

**Definisi.** $R^2$ menunjukkan proporsi variasi variabel target yang dapat dijelaskan oleh model. Nilai maksimumnya 1; semakin mendekati 1, semakin baik model merepresentasikan hubungan antara variabel *independent* dan *dependent*.
- [3] (hlm. 1827, Subbab 2.4; rumus pada Tabel 1, hlm. 1829) memberikan definisi dan rumus tersebut; pada hlm. 1830 nilai $R^2$ = 0,944 ditafsirkan sebagai 94,4% variasi harga dapat dijelaskan oleh atribut dalam model.
- [4] (hlm. 11–12, Subbab 3.6 dan 4.1) menggunakan $R^2$ bersama MSE dan MAE, dan menafsirkan $R^2$ = 1,000 sebagai model yang menjelaskan seluruh variasi data; pada Random Forest, $R^2$ = 0,998 ditafsirkan sebagai lebih dari 99% variasi dapat dijelaskan (hlm. 13).

**Argumen penguat untuk nilai negatif ($R^2 = -0{,}0976$).**
1. **Pembanding implisit adalah model rata-rata.** Penyebut pada rumus $R^2$ adalah galat jika setiap prediksi diganti rata-rata target ($\bar{y}$). Jika $R^2 < 0$, jumlah kuadrat galat model **melebihi** galat prediksi rata-rata, sehingga model lebih buruk daripada *baseline* tersebut. Dari angka proyek ini dapat diturunkan RMSE *baseline* ≈ $28.828 / \sqrt{1{,}0976} \approx 27.500$ (estimasi turunan, bukan keluaran notebook), sedangkan RMSE model $28.828$ lebih tinggi.
2. **Studi lain menunjukkan arah sebaliknya.** [4] (hlm. 12–13) menunjukkan model yang menangkap struktur data dengan tepat menghasilkan $R^2$ hampir 1, sedangkan [3] (hlm. 1830) melaporkan $R^2$ 0,944. Kontras dengan $R^2$ negatif pada proyek ini menegaskan bahwa fitur dan pemrosesan data belum menangkap struktur hubungan terhadap `Price`.
3. **Kehati-hatian membaca $R^2$ tinggi.** [4] (hlm. 13–14, Subbab 4.2) menjelaskan bahwa $R^2$ = 1,000 pada studi tersebut terjadi karena variabel target dibentuk langsung sebagai kombinasi linear dari variabel masukan. Artinya, $R^2$ hanya bermakna jika ditafsirkan bersama cara target dibentuk dan karakteristik data. Prinsip ini juga berlaku di proyek ini: pastikan tidak ada fitur yang secara tidak langsung "membocorkan" `Price`, dan pastikan metrik dihitung pada data yang tidak dipakai untuk pelatihan.
4. **Indikasi multikolinearitas lewat $R^2$.** [1] (hlm. 105) menyebut $R^2$ yang tinggi (di atas 0,80) disertai statistik *t* yang kecil atau tidak signifikan sebagai gejala multikolinearitas. Sebagai pemeriksaan tambahan pada model OLS/LARS di proyek ini, VIF (batas umum ≤ 10, [1] hlm. 106) dapat dihitung untuk fitur yang saling berkorelasi.

---

### 4. Membaca Ketiga Metrik Secara Bersamaan

| Pengamatan | Tafsir | Rujukan |
|---|---|---|
| MAE dan RMSE tinggi relatif terhadap sebaran harga | Prediksi sering meleset jauh dari nilai aktual | [2] hlm. 731–732; [3] hlm. 1830 |
| RMSE > MAE | Terdapat galat besar yang diberi penalti lebih berat oleh kuadrat | [3] hlm. 1827, 1830; [5] hlm. 5 |
| $R^2 < 0$ | Model lebih buruk daripada prediksi rata-rata target | [3] hlm. 1827; [4] hlm. 11 |
| Galat tinggi pada semua variasi model | Indikasi *underfitting* atau fitur kurang informatif | [2] hlm. 731, 734–736 |

Satu metrik saja tidak cukup untuk menilai model. [3] (hlm. 1827 dan 1831) menyatakan bahwa ketiga metrik dipilih karena bersama-sama memberi gambaran komprehensif tentang tingkat kesalahan dan kemampuan model menjelaskan variasi data. [5] (hlm. 5) menggunakan MSE dan MAE secara berpasangan karena keduanya menangkap aspek galat yang berbeda (penalti galat besar vs. rata-rata galat yang tahan pencilan).

---

### Kesimpulan

Berdasarkan hasil evaluasi pada *validation set* di atas, dapat disimpulkan bahwa:

1. **Indikasi *Underfitting* / fitur kurang informatif:** Model gagal menangkap pola tren data pada data validasi, ditandai $R^2$ negatif serta MAE dan RMSE yang tinggi.
   - [2] (hlm. 731, 734–736) melaporkan pola serupa: galat tinggi disertai kesimpulan *underfitting*, dan penyebabnya dikaitkan dengan jumlah data yang terbatas, keberadaan pencilan atau *noise*, serta atribut prediktor yang lemah korelasinya terhadap target (hlm. 728 dan 734).
   - *Catatan kehati-hatian:* GBR adalah model yang fleksibel; $R^2$ negatif pada model seperti ini lebih sering menandakan fitur yang kurang informatif atau perbedaan distribusi antara data latih dan validasi daripada kapasitas model yang kurang. Perlu diperiksa apakah galat pada data latih juga tinggi (*underfitting*) atau rendah (*overfitting*).

2. **Rekomendasi Tindak Lanjut:**
   - **Tinjau ulang *data cleaning* dan pencilan.** [2] (hlm. 724 dan 727) menunjukkan bahwa deteksi dan penghapusan pencilan dengan metode IQR (Q1, median, Q3) dilakukan untuk meningkatkan akurasi analisis, sedangkan kemiringan distribusi (*skewness*) dapat ditangani dengan transformasi log, Box-Cox, atau akar kuadrat (hlm. 727). [1] (hlm. 105) juga menyebut transformasi data (log/Ln), penambahan data, dan penghapusan data penyebab ketidaknormalan sebagai penanganan standar.
   - **Seleksi dan rekayasa fitur (*feature engineering*).** [2] (hlm. 728–729, Tabel 1) memperlihatkan bahwa mayoritas atribut berkorelasi lemah terhadap target sehingga pemodelan menjadi berat; atribut yang paling relevan dipilih dengan ambang korelasi. [3] (hlm. 1827 dan 1831) menunjukkan bahwa atribut numerik yang kuat hubungannya dengan harga (misalnya *carat*) menghasilkan $R^2$ tinggi, sedangkan atribut kategorikal yang tidak disertakan (*color*, *clarity*, *cut*) menurunkan presisi. Untuk proyek ini, periksa korelasi fitur terhadap `Price`, tangani fitur kategorikal dengan encoding yang tepat, dan cek multikolinearitas ([1] hlm. 105–106).
   - **Optimasi hiperparameter (*hyperparameter tuning*) dan pembanding model.** [4] (hlm. 13–14) menekankan bahwa pemilihan model harus mempertimbangkan karakteristik data dan bukan sekadar kompleksitas model, sehingga model sederhana (OLS/LARS) tetap perlu dijadikan pembanding. [4] (hlm. 15) juga menyarankan Gradient Boosting dan XGBoost sebagai kandidat pembanding, sedangkan [3] (hlm. 1831) merekomendasikan Gradient Boosting, Polynomial Regression, atau Random Forest ketika terdapat pola non-linear.
   - **Pastikan pra-pemrosesan tidak menimbulkan kebocoran data.** [5] (hlm. 5, Subbab 2.5) menjelaskan bahwa normalisasi dengan *StandardScaler* harus dijaga agar tidak terjadi *data leakage* dari data validasi atau uji ke dalam model. Praktik ini perlu diterapkan pada tahap standardisasi dan encoding di *pipeline* proyek.
   - **Lakukan diagnostik residual untuk model OLS.** Gunakan uji normalitas residual, multikolinearitas, dan heteroskedastisitas sebagaimana dirinci pada [1] (hlm. 104–108). Catatan: [1] membahas asumsi klasik untuk regresi OLS; GBR bukan model OLS sehingga asumsi tersebut tidak wajib untuk GBR, tetapi tetap relevan bagi *baseline* Linear Regression pada proyek ini.

---

### Pemetaan Penggunaan Jurnal

| Jurnal | Dipakai untuk |
|---|---|
| [1] Sholihah dkk. (2023), hlm. 102–110 (https://jos.unsoed.ac.id/index.php/jras/article/view/10792/4870) | Residual, asumsi normalitas, multikolinearitas ($R^2$, VIF), heteroskedastisitas (uji Glejser), penanganan pelanggaran asumsi dan pencilan |
| [2] Arifuddin dkk. (2025), hlm. 722–737 (https://journal.irpi.or.id/index.php/malcom/article/view/1952/960) | Definisi MSE/RMSE dan interpretasi skala, *underfitting*, penanganan pencilan (IQR), seleksi fitur berbasis korelasi, perbandingan model |
| [3] Agustina dkk. (2026), hlm. 1825–1832 (https://www.ejournal.itn.ac.id/jati/article/view/17567/9059) | Definisi dan rumus MAE, MSE, $R^2$; interpretasi nilai; keterbatasan fitur dan rekomendasi model |
| [4] Inabah dkk. (2026), hlm. 11–15 (https://jurnal2.untagsmg.ac.id/index.php/DBIJ/article/view/3504/2835) | Penggunaan MSE/MAE/$R^2$ sebagai tiga metrik bersama, kehati-hatian membaca $R^2$ tinggi, pemilihan model berdasarkan karakteristik data |
| [5] Putra dkk. (2026), hlm. 4–5 (https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/16205/7148) | Definisi MAE dan MSE, ketahanan MAE terhadap pencilan, penalti MSE, dekomposisi bias-*variance*, pencegahan kebocoran data |
