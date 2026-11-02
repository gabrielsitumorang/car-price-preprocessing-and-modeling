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

   Hasil Evaluasi dan Penjelasan Metrik ValidasiBerdasarkan hasil pengujian model pada validation set, diperoleh nilai metrik evaluasi sebagai berikut:MAE (Mean Absolute Error) : $25.161,45$RMSE (Root Mean Squared Error) : $28.828,00$$R^2$ Score (Koefisien Determinasi) : $-0,0976$Penjelasan Metrik Evaluasi & Landasan Teori1. MAE (Mean Absolute Error)Definisi: MAE mengukur rata-rata dari selisih mutlak antara nilai prediksi model dengan nilai aktual di lapangan. Dalam literatur peramalan, MAE sangat disukai karena sifatnya yang robust (tahan) terhadap keberadaan nilai ekstrem (outlier) dikarenakan error tidak dikuadratkan.Argumen & Literatur Penguat:Menurut Agustina et al. (2026) (JATI, Vol. 10 No. 1, hlm. 1828), MAE menghitung rata-rata kesalahan absolut untuk menilai seberapa dekat estimasi model terhadap nilai sebenarnya secara linear.Putra et al. (2026) (J-PTIIK, Vol. 10 No. 3, hlm. 4) menegaskan bahwa MAE optimal digunakan ketika distribusi galat mengikuti distribusi Laplace karena tidak memberikan penalti berlebihan pada deviasi minor.Didukung pula oleh Inabah et al. (2026) (Digital Business Intelligence, Vol. 02 No. 01), yang menggunakan MAE sebagai indikator deviasi prediktif mutlak dalam membandingkan performa model regresi.Interpretasi Nilai: Nilai MAE sebesar $25.161,45$ menunjukkan bahwa secara rata-rata, estimasi harga mobil yang dihasilkan model meleset sekitar angka tersebut dari harga aslinya.2. MSE (Mean Squared Error) & RMSE (Root Mean Squared Error)Definisi: MSE menghitung rata-rata kuadrat dari selisih galat (error), sedangkan RMSE merupakan akar kuadrat dari MSE. Karena melibatkan proses pengkuadratan, metrik ini memberikan penalti yang jauh lebih besar terhadap kesalahan prediksi bernilai tinggi.Argumen & Literatur Penguat:Arifuddin et al. (2025) (MALCOM, Vol. 5 Iss. 2, hlm. 722, 731) menyatakan bahwa nilai MSE yang tinggi mengindikasikan bahwa model sering kali meleset jauh dari target aktual, yang menjadi salah satu tanda utama terjadinya underfitting dalam regresi.Sholihah et al. (2023) (JRAS, Vol. 2 No. 2, hlm. 103, 105) menjelaskan bahwa analisis galat residual berbasis Ordinary Least Squares (OLS) sangat bergantung pada pengukuran kuadratik untuk memastikan estimasi parameter yang tidak bias.Putra et al. (2026) (J-PTIIK, Vol. 10 No. 3, hlm. 5) menambahkan bahwa MSE ideal untuk data berdistribusi normal karena meminimalkan variansi galat serta memfasilitasi dekomposisi antara komponen bias dan varians.Interpretasi Nilai: Nilai RMSE sebesar $28.828,00$ (lebih tinggi daripada MAE) mengonfirmasi adanya beberapa titik data prediksi dengan tingkat kesalahan ekstrem yang mendistorsi kestabilan model.3. $R^2$ Score (Koefisien Determinasi)Definisi: $R^2$ Score mengukur seberapa besar proporsi variasi dari variabel target (Price) yang dapat dijelaskan oleh keseluruhan variabel independen dalam model. Nilai maksimumnya adalah $1.0$.Argumen & Literatur Penguat:Agustina et al. (2026) (JATI, Vol. 10 No. 1, hlm. 1828, 1830) menyatakan bahwa nilai $R^2$ merefleksikan kemampuan model dalam merepresentasikan hubungan proporsional antara atribut input dan target, di mana nilai mendekati 1 menunjukkan kecocokan model yang sangat baik.Inabah et al. (2026) (Digital Business Intelligence, Vol. 02 No. 01) menegaskan bahwa $R^2$ mengukur kualitas goodness-of-fit; jika nilainya sangat rendah atau negatif, hal ini menandakan struktur hubungan linier gagal ditangkap dengan optimal oleh algoritma.Interpretasi Nilai: Nilai $R^2$ bernilai negatif yaitu $-0,0976$, yang secara teoritis membuktikan bahwa model regresi saat ini memiliki performa yang lebih buruk daripada sekadar memprediksi menggunakan nilai rata-rata (mean) data target.Kesimpulan & Rekomendasi Tindak LanjutBerdasarkan tinjauan metrik evaluasi di atas yang didukung oleh literatur terkait:Indikasi Underfitting: Model gagal menangkap pola tren data yang kompleks pada data validasi, tecermin dari nilai $R^2$ negatif serta tingginya angka galat.Rekomendasi Perbaikan:Melakukan peninjauan ulang pada tahap data cleaning dan penanganan outlier ekstrem menggunakan metode seperti Interquartile Range (IQR).Melakukan seleksi ulang atau feature engineering untuk mencari variabel prediktor dengan korelasi yang lebih kuat terhadap Price.Melakukan hyperparameter tuning atau mengeksplorasi pendekatan non-linear / ensemble untuk meningkatkan kemampuan generalisasi model.
