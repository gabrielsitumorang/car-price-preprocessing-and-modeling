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
- **Metrik Evaluasi:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), $R^2$ Score
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

- **MAE (Mean Absolute Error) :** $25.161,45$
- **RMSE (Root Mean Squared Error) :** $28.828,00$
- **$R^2$ Score (Koefisien Determinasi) :** $-0,0976$

---

### Penjelasan Metrik Evaluasi

1. **MAE (*Mean Absolute Error*)**
   * **Definisi:** MAE menghitung rata-rata dari selisih mutlak antara nilai prediksi model dengan nilai aktual (sebenarnya). 
   * **Interpretasi:** Nilai MAE sebesar **$25.161,45$** berarti secara rata-rata, estimasi harga mobil yang dihasilkan oleh model meleset sekitar angka tersebut dari harga aslinya di lapangan.

2. **RMSE (*Root Mean Squared Error*)**
   * **Definisi:** RMSE adalah akar kuadrat dari rata-rata kuadrat galat (*error*). Karena melibatkan proses pengkuadratan, metrik ini memberikan penalti yang jauh lebih besar terhadap kesalahan prediksi yang bernilai tinggi.
   * **Interpretasi:** Nilai RMSE sebesar **$28.828,00$** menunjukkan tingkat sebaran galat. Karena nilainya lebih tinggi daripada MAE, hal ini mengindikasikan adanya beberapa titik data prediksi yang memiliki tingkat kesalahan cukup ekstrem.

3. **$R^2$ Score (*Koefisien Determinasi*)**
   * **Definisi:** $R^2$ Score mengukur seberapa besar proporsi variasi dari variabel target (`Price`) yang dapat dijelaskan oleh model. Nilai maksimumnya adalah $1.0$.
   * **Interpretasi:** Nilai $R^2$ bernilai negatif yaitu **$-0,0976$**. Kondisi ini menunjukkan bahwa model regresi Anda saat ini memiliki performa yang **lebih buruk** daripada sekadar memprediksi menggunakan nilai rata-rata (*mean*) dari data target.

---

### Kesimpulan

Berdasarkan hasil evaluasi pada *validation set* di atas, dapat disimpulkan bahwa:
1. **Indikasi *Underfitting*:** Model gagal menangkap pola tren data yang kompleks pada data validasi, yang ditandai dengan nilai $R^2$ negatif serta angka *error* (MAE dan RMSE) yang tinggi.
2. **Rekomendasi Tindak Lanjut:** 
   * Perlu dilakukan peninjauan ulang pada tahap *data cleaning*, khususnya penanganan data pencilan (*outlier*) harga ekstrem yang mungkin mendistorsi proses pelatihan.
   * Melakukan seleksi ulang atau penambahan fitur (*feature engineering*) yang memiliki korelasi lebih kuat terhadap variabel target `Price`.
   * Melakukan optimasi hiperparameter (*hyperparameter tuning*) pada model agar memiliki kemampuan generalisasi yang lebih baik.

## Cara Penggunaan
1. Kloning repositori ini ke perangkat lokal Anda.
2. Pastikan pustaka yang dibutuhkan telah terpasang (*scikit-learn*, *pandas*, *numpy*, dll).
3. Jalankan berkas notebook di dalam direktori `notebooks/prediksihargamobil.ipynb` untuk melihat keseluruhan tahapan pelatihan hingga evaluasi.