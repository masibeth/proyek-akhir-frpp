# Laporan Analisis Proyek RFPP Kelompok 5

Tanggal analisis: 2026-11-12

## 1. Ringkasan eksekutif

Proyek ini membangun eksperimen klasifikasi kualitas daging dengan **fusi dua
modalitas**:

- visual: fitur warna/tekstur dari citra;
- sensor: delapan kanal sensor gas e-nose.

Arsitektur proyek relatif terstruktur dan berfokus pada evaluasi yang anti-leakage.
Konfigurasi eksperimen dipusatkan di [`config.yaml`](./config.yaml), pemrosesan
data dan fitur berada di [`src/`](./src/), pengujian berada di [`tests/`](./tests/),
sedangkan eksplorasi dan pembuatan figur berada di
[`notebooks/01_eksplorasi_dan_evaluasi.ipynb`](./notebooks/01_eksplorasi_dan_evaluasi.ipynb).

Hasil tersimpan menunjukkan bahwa:

- model terbaik adalah **SVM-RBF pada fitur fused**, macro-F1
  `0.9983 ± 0.0003`;
- Random Forest fused sangat dekat, macro-F1 `0.9981 ± 0.0007`;
- visual sendiri sudah sangat kuat (`0.9959` macro-F1 dengan Random Forest),
  sedangkan sensor sendiri lebih rendah (`0.9600`);
- kontribusi utama berasal dari modalitas visual, tetapi sensor tetap memberi
  tambahan performa pada semua model utama;
- scaling sangat penting untuk KNN dan SVM-RBF;
- pengujian sintaks berhasil, tetapi test suite belum dapat dieksekusi karena
  dependensi Python belum terpasang pada environment aktif.

## 2. Struktur dan tanggung jawab komponen

| Komponen | Peran |
|---|---|
| [`config.yaml`](./config.yaml) | Sumber parameter, path, label, protokol split, model, metrik, dan importance |
| [`src/load_data.py`](./src/load_data.py) | Audit data mentah, validasi hash, agregasi sensor, timestamp visual, dan manifest |
| [`src/features_visual.py`](./src/features_visual.py) | Ekstraksi 20 fitur visual dan cache Parquet |
| [`src/features_sensor.py`](./src/features_sensor.py) | Validasi kanal sensor dan cache Parquet |
| [`src/protocol.py`](./src/protocol.py) | Split berbasis grup, pairing antar-modalitas, pipeline preprocessing, dan model |
| [`src/train_eval.py`](./src/train_eval.py) | Grid evaluasi, metrik, confusion matrix, dan delta berpasangan |
| [`src/probe_discordant.py`](./src/probe_discordant.py) | Probe pasangan visual-sensor yang labelnya sengaja dibuat diskordan |
| [`src/importance.py`](./src/importance.py) | Block permutation importance dan importance per fitur |
| [`tests/`](./tests/) | Pengujian fitur, protokol, training/evaluasi, probe, dan importance |
| [`results/`](./results/) | Hasil eksperimen CSV |
| [`figures/`](./figures/) | 21 visualisasi EDA dan evaluasi |

Notebook memiliki 44 sel dan mencakup inventaris data, EDA sensor/visual,
protokol pairing, evaluasi komparatif, probe diskordan, dominasi modalitas, dan
batas klaim.

## 3. Data dan preprocessing

### 3.1 Sensor

Menurut [`data/manifest.json`](./data/manifest.json):

- 20.815 baris mentah menjadi 1.735 baris setelah agregasi rata-rata per menit;
- kelas agregat: 625 `fresh` dan 1.110 `spoiled`;
- tersedia 21 grup waktu `fresh` dan 38 grup waktu `spoiled`;
- delapan kanal MQ digunakan: `Mq-2`, `Mq-3`, `Mq-4`, `Mq-5`, `Mq-135`,
  `Mq-136`, `Mq-137`, `Mq-138`;
- kolom waktu, lingkungan, TVC, dan label asli dikeluarkan dari fitur;
- hash unduhan cocok dengan hash penerbit;
- pemetaan kelas mengikuti keputusan proyek: kelas 1--3 menjadi `fresh`,
  kelas 4 menjadi `spoiled`.

Agregasi per menit dan split per blok 30 menit mengurangi pseudo-replikasi dan
mencegah baris yang sangat berdekatan waktunya tersebar ke train dan test.

### 3.2 Visual

- 948 citra per kelas ditemukan;
- timestamp dibaca dari nama file karena EXIF kosong;
- interval waktu dominan adalah dua menit;
- 24 frame di sekitar pergantian label dibuang;
- 1.872 frame digunakan: 939 `fresh` dan 933 `spoiled`;
- terdapat 64 grup waktu per kelas dengan ukuran blok 30 menit.

Fitur visual berjumlah 20:

- 16 histogram HSV: 8 bin hue, 4 bin saturation, 4 bin value;
- 4 fitur GLCM: contrast, homogeneity, energy, correlation.

Urutan preprocessing yang terdokumentasi adalah resize, center crop 80%,
Gaussian smoothing, konversi HSV, histogram, dan GLCM.

### 3.3 Fusi

Fitur fused memiliki 28 dimensi, yaitu 20 visual diikuti 8 sensor. Pairing
dilakukan hanya di dalam fold dan tanpa pengembalian, sehingga baris unimodal
dibandingkan dengan pasangan yang sama seperti fused.

## 4. Protokol evaluasi

Implementasi utama berada di [`src/protocol.py`](./src/protocol.py) dan
[`src/train_eval.py`](./src/train_eval.py):

1. split visual dan sensor secara terpisah;
2. gunakan `StratifiedGroupKFold` bila grup tersedia;
3. lakukan pairing berlabel sama di dalam fold;
4. fit clipping dan scaling hanya pada data latih;
5. latih model dan evaluasi pada data uji;
6. ulangi untuk 5 fold dan 5 pairing seed.

Grid utama mencakup tiga set fitur (`visual`, `sensor`, `fused`) dan empat model:
Random Forest, SVM-RBF, KNN, dan MLP. Untuk fused KNN dan SVM-RBF, varian
standard scaling dan tanpa scaling juga diuji.

Desain ini diperkuat oleh tes sintetis di
[`tests/test_protocol.py`](./tests/test_protocol.py), termasuk pemeriksaan
tidak ada overlap ID/grup, pairing unik, kesamaan baris unimodal-fused, dan
scaler yang hanya fit pada train.

## 5. Hasil eksperimen

### 5.1 Performa utama

Nilai berikut berasal dari [`results/summary.csv`](./results/summary.csv) dan
merupakan mean ± standar deviasi antar lima pairing seed.

| Set fitur | Model | Scaling | Macro-F1 |
|---|---|---:|---:|
| fused | SVM-RBF | standard | 0.9983 ± 0.0003 |
| fused | Random Forest | standard | 0.9981 ± 0.0007 |
| fused | KNN | standard | 0.9964 ± 0.0008 |
| fused | MLP | standard | 0.9841 ± 0.0057 |
| visual | Random Forest | standard | 0.9959 ± 0.0010 |
| visual | KNN | standard | 0.9909 ± 0.0009 |
| visual | SVM-RBF | standard | 0.9902 ± 0.0009 |
| visual | MLP | standard | 0.9788 ± 0.0133 |
| sensor | Random Forest | standard | 0.9600 ± 0.0041 |
| sensor | SVM-RBF | standard | 0.9621 ± 0.0009 |
| sensor | KNN | standard | 0.9668 ± 0.0025 |
| sensor | MLP | standard | 0.9503 ± 0.0167 |

Fused meningkatkan macro-F1 dibanding visual untuk semua model, dengan
peningkatan terbesar pada MLP (`+0.0213` rata-rata) dan terkecil pada Random
Forest (`+0.0021`). Dibanding sensor, peningkatannya jauh lebih besar, sekitar
`+0.0243` sampai `+0.0582`.

### 5.2 Dampak scaling

Pada fused:

- KNN turun dari macro-F1 `0.9964` menjadi `0.9586` tanpa scaling;
- SVM-RBF turun dari `0.9983` menjadi `0.9701` tanpa scaling.

Ini mengonfirmasi bahwa perbandingan model berbasis jarak perlu dipisahkan
antara varian dengan dan tanpa standard scaling. Random Forest tidak memerlukan
scaling secara konseptual, tetapi pipeline utama tetap konsisten menggunakan
konfigurasi standard.

### 5.3 Dominasi modalitas

Menurut [`results/importance_summary.csv`](./results/importance_summary.csv),
permutasi seluruh blok visual menurunkan macro-F1 sekitar `0.36--0.38`,
sedangkan blok sensor sekitar `0.12--0.13`. Visual adalah sumber sinyal
dominan, namun hasil delta berpasangan menunjukkan sensor tetap menambah
informasi di atas visual.

Interpretasi ini sebaiknya tetap disebut sebagai **kepentingan prediktif
dalam protokol eksperimen**, bukan hubungan kausal.

### 5.4 Probe diskordan

[`results/probe_summary.csv`](./results/probe_summary.csv) menguji pasangan
visual dan sensor yang sengaja berbeda label, tanpa memasukkan pasangan tersebut
ke data latih. Probe ini tidak menghitung akurasi karena tidak memiliki
ground truth gabungan.

Temuan penting:

- arah `busuk-segar` cenderung lebih sering diprediksi sebagai busuk;
- arah `segar-busuk` lebih bervariasi antar model;
- confidence tinggi tidak boleh dibandingkan langsung antar model karena jenis
  skornya berbeda (`max_proba` versus `abs_margin`).

## 6. Kekuatan proyek

1. **Konfigurasi terpusat**: parameter eksperimen dapat ditelusuri melalui
   [`config.yaml`](./config.yaml), termasuk seed global dan dimensi fitur.
2. **Anti-leakage eksplisit**: split grup waktu, pairing dalam fold, serta fit
   preprocessing pada train didukung oleh implementasi dan tes.
3. **Reproducibility data**: manifest menyimpan jumlah baris, distribusi kelas,
   hash data sensor, hash konfigurasi, dan keputusan penting.
4. **Evaluasi berpasangan**: perbandingan fused versus unimodal memakai baris
   dan seed pairing yang konsisten.
5. **Analisis interpretabilitas**: block permutation, drop-one, top-10 fitur,
   dan probe diskordan memberi konteks di luar satu angka akurasi.
6. **Artefak hasil lengkap**: CSV metrik, confusion matrix, delta,
   importance, probe, serta 21 figur tersedia di repository.

## 7. Risiko dan area perbaikan

### Prioritas tinggi

1. **Dependensi belum siap di environment aktif.** `pytest` tidak tersedia baik
   sebagai executable maupun modul di `.venv`, sehingga test suite belum bisa
   dijalankan. `requirements.txt` sudah mematok versi, tetapi perlu instalasi
   environment sebelum hasil dapat diverifikasi ulang.
2. **Konfigurasi imputasi belum diwujudkan.** `config.yaml` menetapkan
   `imputation: median`, tetapi pipeline di [`src/protocol.py`](./src/protocol.py)
   tidak memasukkan transformer imputasi. Saat ini data hilang ditolak oleh
   audit/cache, bukan diimputasi. Dokumentasi konfigurasi dan perilaku kode
   perlu diselaraskan: pilih implementasi median fit pada train atau hapus
   klaim imputasi.

### Prioritas menengah

3. **`RobustZClip` diterapkan ke semua modalitas.** `make_pipeline` selalu
   menambahkan clipping berbasis konfigurasi sensor, termasuk ketika `kind`
   adalah visual. Jika clipping visual memang keputusan desain, hal ini perlu
   dinyatakan di konfigurasi/dokumentasi; jika tidak, preprocessing perlu
   dibedakan per modalitas.
4. **Raw data tidak menjadi bagian tree yang terlihat.** Repository menyimpan
   manifest dan hasil, sementara folder raw diabaikan oleh `.gitignore`.
   Reproduksi penuh bergantung pada unduhan sumber eksternal dan kecocokan hash.
5. **Performa sangat tinggi perlu dibaca hati-hati.** Visual merupakan satu
   trayektori kamera dengan label yang berubah berdasarkan waktu. Walaupun
   blok batas dibuang dan split berbasis grup digunakan, sinyal waktu/trajectory
   masih berpotensi menjadi proxy label. Analisis lintas sesi atau validasi
   eksternal akan memperkuat klaim generalisasi.

### Prioritas rendah

6. Belum terlihat konfigurasi formal untuk lint/build selain test suite.
   Menambahkan perintah standar, misalnya `pytest`, pada dokumentasi penggunaan
   akan mengurangi ambiguitas operasional.
7. Threshold dan hasil importance sudah tersedia, tetapi interval ketidakpastian
   atau pengujian statistik formal belum menjadi keluaran utama. Ini bukan
   blocker, tetapi berguna bila laporan akademik menuntut signifikansi.

## 8. Validasi yang dilakukan

- `python -m compileall -q src tests`: **berhasil**.
- `pytest -q`: **tidak dapat dijalankan**, executable `pytest` tidak ditemukan.
- `./.venv/bin/python -m pytest -q`: **tidak dapat dijalankan**, modul `pytest`
  tidak terpasang.
- Status Git saat analisis dimulai bersih; laporan ini adalah perubahan baru.

## 9. Rekomendasi tindak lanjut

1. Instal dependensi dari [`requirements.txt`](./requirements.txt) pada
   environment terisolasi, lalu jalankan seluruh `pytest`.
2. Putuskan kontrak missing value: implementasikan median imputation di dalam
   pipeline train-only atau revisi konfigurasi agar menyatakan data hilang
   ditolak.
3. Dokumentasikan keputusan clipping visual atau ubah pipeline agar clipping
   hanya diterapkan pada modalitas yang memang membutuhkannya.
4. Tambahkan prosedur evaluasi berbasis sesi/trajectory yang benar-benar
   terpisah untuk menguji generalisasi di luar urutan waktu yang sama.
5. Simpan command reproducibility utama untuk audit: pembuatan manifest, cache
   fitur, evaluasi, importance, probe, dan notebook.

## 10. Kesimpulan

Proyek sudah memiliki fondasi eksperimen yang baik dan lebih matang daripada
pipeline klasifikasi biasa: data diaudit, konfigurasi dipusatkan, evaluasi
berpasangan, dan leakage diuji secara eksplisit. Secara empiris, fusi visual
dan sensor memberi performa terbaik, tetapi visual adalah modalitas dominan.

Sebelum proyek dianggap sepenuhnya reproducible, dua hal paling penting adalah
menyiapkan environment dependensi agar test suite dapat dijalankan dan
menyelaraskan deklarasi imputasi serta clipping dengan implementasi aktual.
