# 🌍 GeoForward 2D

**Aplikasi web forward modelling geolistrik tahanan jenis 2D berbasis Streamlit**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Tentang Aplikasi

GeoForward 2D adalah aplikasi web yang memungkinkan peneliti geolistrik untuk:

1. **Upload file .INV** hasil inversi RES2DINV
2. **Melihat model true resistivity 2D** secara interaktif
3. **Menjalankan forward modelling otomatis** dengan berbagai variasi
4. **Membandingkan hasil** secara visual dan statistik

### Fitur Utama

| Fitur | Keterangan |
|-------|------------|
| Parser .INV | Membaca semua format output RES2DINV v3.x–10.x |
| Variasi spasi | Simulasi dengan spasi elektroda 1–5 m |
| Variasi konfigurasi | Wenner, Wenner-Schlumberger, Dipole-Dipole |
| Pseudosection interaktif | Plot Plotly yang bisa di-zoom dan di-hover |
| Export data | Download CSV dan .dat format RES2DINV |
| Data demo | Contoh lintasan sintetik 30 m untuk testing |

---

## 🚀 Cara Deploy ke Streamlit Cloud

### Langkah 1 — Fork atau clone repositori ini

```bash
git clone https://github.com/USERNAME/geoforward-2d.git
cd geoforward-2d
```

### Langkah 2 — Push ke GitHub

```bash
git init
git add .
git commit -m "Initial commit GeoForward 2D"
git branch -M main
git remote add origin https://github.com/USERNAME/geoforward-2d.git
git push -u origin main
```

### Langkah 3 — Deploy di Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan akun GitHub
3. Klik **New app**
4. Pilih repositori `geoforward-2d`
5. Set **Main file path**: `app.py`
6. Klik **Deploy!**

Aplikasi akan otomatis tersedia di URL:
```
https://USERNAME-geoforward-2d-app-XXXX.streamlit.app
```

---

## 💻 Menjalankan Secara Lokal

```bash
# Clone repositori
git clone https://github.com/USERNAME/geoforward-2d.git
cd geoforward-2d

# Install dependensi
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## 📁 Struktur File

```
geoforward-2d/
├── app.py              # Aplikasi Streamlit utama
├── inv_parser.py       # Parser file .INV dari RES2DINV
├── forward_model.py    # Kalkulasi forward modelling 2D
├── plotting.py         # Visualisasi penampang dengan Plotly
├── requirements.txt    # Dependensi Python
└── README.md           # Dokumentasi ini
```

---

## 📂 Format File .INV yang Didukung

File .INV harus merupakan output langsung dari RES2DINV. Format yang didukung:

```
Nama_Lintasan
Unit electrode spacing = 2.0 m
Array type: Wenner-Schlumberger (7)
Number of electrodes: 16

Iteration 1   RMS error = 21.4%
   1    2.00    0.600   128.40
   2    4.00    0.600   143.20
   ...

Iteration 5   RMS error = 3.2%
   1    2.00    0.600   125.80
   ...
```

---

## 🧮 Metode Kalkulasi Forward

Aplikasi menggunakan pendekatan **semi-analitik berbasis weighted IDW (Inverse Distance Weighting)**:

- Setiap kombinasi elektroda (C1-P1-P2-C2) dihitung kontribusi resistivitas-nya dari setiap blok model
- Bobot berdasarkan jarak dan posisi blok relatif terhadap spread elektroda
- Faktor geometri K dihitung sesuai formula masing-masing konfigurasi

### Formula faktor geometri K

| Konfigurasi | Formula K |
|-------------|-----------|
| Wenner | 2πa |
| Wenner-Schlumberger | πn(n+1)a |
| Dipole-Dipole | πn(n+1)(n+2)a |

---

## 📊 Panduan Penggunaan

### 1. Upload Data
- Klik "Browse files" di sidebar atau gunakan drag & drop
- Format: `.inv` atau `.INV` (bisa juga `.txt`)
- Atau klik "Gunakan Data Demo" untuk mencoba aplikasi

### 2. Atur Parameter
- Pilih variasi spasi elektroda (bisa lebih dari satu)
- Pilih variasi konfigurasi (bisa lebih dari satu)
- Atur spasi referensi untuk komparasi konfigurasi
- Opsional: tambahkan noise sintetik

### 3. Jalankan Simulasi
- Klik tombol **▶ Jalankan Forward Modelling**
- Semua kombinasi variasi dijalankan otomatis

### 4. Analisis Hasil
- Tab **Data Inversi**: ringkasan file .INV dan konvergensi RMS
- Tab **Model True ρ**: penampang resistivitas 2D hasil inversi
- Tab **Forward Modelling**: pseudosection per simulasi + export
- Tab **Komparasi Spasi**: perbandingan semua variasi spasi
- Tab **Komparasi Konfigurasi**: perbandingan semua konfigurasi

---

## 🔬 Referensi Ilmiah

- Loke, M.H. (2004). *2D and 3D electrical imaging surveys*. Geotomo Software.
- Loke, M.H. (2004). *RES2DMOD ver. 3.01 Manual*. Geotomo Software.
- Telford, W.M., Geldart, L.P., Sheriff, R.E. (1990). *Applied Geophysics*. Cambridge University Press.
- Reynolds, J.M. (1997). *An Introduction to Applied and Environmental Geophysics*. Wiley.

---

## 📝 Lisensi

MIT License — bebas digunakan untuk keperluan akademik dan penelitian.

---

## 👤 Kontribusi

Pull request dan issue sangat disambut. Untuk pertanyaan akademik terkait interpretasi geolistrik, silakan buka Issue di GitHub.
