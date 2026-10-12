# 🚗 Car Price Preprocessing and Modeling

Repositori ini memuat implementasi *end-to-end machine learning pipeline* untuk memprediksi harga mobil bekas (*used car price prediction*). Proyek ini berfokus pada tahapan komprehensif penyiapan data (*data cleaning*, penanganan pencilan, standardisasi, dan *encoding*), analisis eksploratif (EDA & ExDA), perbandingan performa algoritma regresi (LARS, Linear Regression, dan Gradient Boosting Regressor), hingga serialisasi model siap pakai (*model persistence*).

---

## 📌 Ringkasan Proyek

* **Tipe Masalah:** Regresi Nilai Kontinu (*Tabular Data*)
* **Dataset:** `Car Price Prediction` (~2.500 baris total)
* **Target Variabel:** `Price`
* **Algoritma yang Diuji:** 
  * Least Angle Regression (LARS)
  * Linear Regression (OLS)
  * Gradient Boosting Regressor (GBR)
* **Metrik Evaluasi:** Mean Absolute Error (MAE), Mean Squared Error (MSE), $R^2$ Score
* **Format Penyimpanan Model:** `.joblib` & `.pkl`

---

## 🛠️ Pustaka & Lingkungan Komputasi

* **Bahasa:** Python 3.x
* **Lingkungan Kerja:** Jupyter Notebook
* **Pengolahan & Analisis Data:** Pandas, NumPy
* **Visualisasi:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn
* **Serialisasi Objek:** Joblib, Pickle

---

## 📂 Struktur Repositori

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
