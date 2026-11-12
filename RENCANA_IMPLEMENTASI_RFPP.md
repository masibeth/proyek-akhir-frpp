# Rencana Implementasi — RFPP Kelompok 5
## Fusi Fitur Citra RGB + E-Nose untuk Deteksi Kebusukan Daging

Disusun 2026-11-11 sebagai rencana kerja sebelum satu baris kode pun ditulis.
Pendamping dokumen metode: `Detail Metode Fusi - RFPP Kelompok 5.pdf`.

**Tiga keputusan yang sudah dikunci dan menjadi premis rencana ini:**

| Keputusan | Pilihan | Konsekuensi |
|---|---|---|
| Pertanyaan riset | **A + B** | Klaim utama = keunggulan fusi atas unimodal. Tujuan anomali dijawab lewat *probe* diskordan terpisah. **§2 dokumen metode wajib direvisi.** |
| Waktu | **8–14 hari** | Semua butir wajib masuk + multi-seed pairing + permutation importance. Probe B masuk sebagai fase kecil. |
| Dataset | **Belum diunduh** | Fase 1 memuat pencarian, seleksi, pencatatan lisensi, dan pengunduhan. Fase 1 adalah gerbang: gagal di sini, fase lain tidak boleh mulai. |

---

## 1. Sasaran dan batas klaim

Tulis daftar ini di laporan dan jangan melewatinya. Ini melindungi kelompok dari pertanyaan penguji yang tidak bisa dijawab data.

**Boleh diklaim:**
1. Pada pasangan konkordan semi-sintetis, ruang fitur gabungan 28-D memisahkan kelas lebih baik / tidak lebih baik daripada visual-saja dan sensor-saja, diukur pada empat klasifikasi dengan protokol anti-bocor.
2. Blok modalitas mana yang lebih menentukan keputusan model, diukur dengan *block-permutation importance* pada data uji dan dikonfirmasi *drop-one*.
3. Ketika sinyal kedua modalitas bertentangan, model yang dilatih pada pasangan konkordan cenderung mengikuti modalitas tertentu — diukur sebagai tingkat kesepakatan, **bukan akurasi**.
4. Efek penskalaan antar-blok terhadap model berbasis jarak (KNN, SVM-RBF).

**Tidak boleh diklaim:**
1. Akurasi deteksi anomali pada daging nyata — tidak ada objek fisik yang diukur serempak.
2. Korelasi kausal antar-modalitas per sampel — pairing hanya menghubungkan label.
3. Generalisasi ke daging di luar dua dataset ini — ada *domain gap* antar-sumber.
4. Bahwa 28-D adalah representasi optimal — tidak ada pencarian arsitektur.

**Revisi dokumen metode yang wajib (sebelum laporan ditulis):**
- §2 alasan pertama: ganti janji "mendeteksi citra segar tetapi gas busuk" menjadi klaim ruang fitur bersama; tujuan anomali dipindah ke bagian *probe* dengan label eksperimen tambahan.
- §4 *Alignment*: tambah satu kalimat bahwa pasangan diskordan dibuat **hanya untuk uji**, tidak pernah masuk data latih.
- §7: tambah batas klaim *probe* (tidak ada ground truth pasangan diskordan).
- Judul dan penyebutan "early fusion" tetap, karena §7 sudah jujur menyebutnya *feature-level*.

---

## 2. Lingkungan dan struktur repo

**Aturan keras: jangan pasang apa pun ke venv tesis.** Venv itu mengunci 317 tes tesis; menambah paket ke sana berisiko merusak reproduksibilitas. Buat venv terpisah di folder RFPP.

Paket yang dibutuhkan (belum ada di mesin ini): `scikit-image`, `matplotlib`, `pillow`, `pyyaml`. Yang sudah tersedia dan boleh dipakai versinya sebagai acuan: numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0, Python 3.12.4.

```
RFPP/
  Detail Metode Fusi - RFPP Kelompok 5.pdf
  RENCANA_IMPLEMENTASI_RFPP.md      # berkas ini
  config.yaml                       # SATU sumber angka beku
  requirements.txt                  # versi dipatok
  data/raw/visual/                  # tidak masuk git
  data/raw/sensor/
  data/manifest.json                # sumber, versi, lisensi, jumlah sampel, sha256
  src/
    __init__.py
    load_data.py                    # F1: audit + manifest
    features_visual.py              # F2
    features_sensor.py              # F3
    protocol.py                     # F4: split + pairing + pipeline (PEMEGANG INVARIAN)
    train_eval.py                   # F5
    probe_discordant.py             # F5b
    importance.py                   # F5c
    report.py                       # F6
  cache/features/                   # matriks fitur beku (.parquet)
  results/                          # metrics.csv, importance.csv, probe.csv
  figures/
  tests/                            # uji anti-bocor
```

`config.yaml` memuat **semua** angka yang bisa bergerak: seed global, ukuran resize, ukuran ROI, sigma denoise, bin HSV, level GLCM, jarak/sudut GLCM, daftar kanal sensor, jumlah fold, jumlah seed pairing, daftar model dan hyperparameter-nya. Tidak boleh ada angka ajaib di dalam `src/`.

---

## 3. Spesifikasi fitur beku

### 3.1 Visual — tepat 20-D

| Kelompok | Definisi | Dimensi |
|---|---|---|
| Histogram HSV | per-kanal, L1-normalized (jumlah = 1): H **8** bin, S **4** bin, V **4** bin | **16** |
| GLCM | grayscale ROI dikuantisasi ke **64 level**; jarak **1**; sudut **0, 45, 90, 135** lalu dirata-ratakan; `symmetric=True`, `normed=True`; properti: contrast, homogeneity, energy, correlation | **4** |
| | **Total** | **20** |

Pra-pemrosesan beku: resize ke **256×256** → *center crop* **80%** (204×204) sebagai ROI untuk membuang latar/wadah → Gaussian **σ = 1.0** → RGB→HSV.

Alasan sudut dirata-ratakan: ketahanan terhadap rotasi potongan daging. Alasan 64 level: GLCM stabil dan cepat pada citra kecil. Kedua alasan ini ditulis di laporan, bukan hanya di kode.

### 3.2 Sensor — dimensi diputuskan di F1, bukan dipaksakan 6

Dokumen menulis "±6". Itu tidak bisa dikerjakan. Aturan keputusan:

- Jika dataset UCI terpilih punya **tepat 6 kanal** resistansi/tegangan yangusable → pakai 6, dokumen tidak berubah.
- Jika **lebih dari 6** → **pakai semua kanal** dan perbarui angka dimensi di dokumen + diagram. Jangan pangkas jadi 6 tanpa kriteria.
- Jika tetap ingin 6 → satu-satunya cara yang sah: **PCA ke 6 komponen, fit di dalam fold latih**. Bukan memilih kanal berdasarkan separabilitas kelas di seluruh data (itu kebocoran terawasi).

Pra-pemrosesan beku: buang baris rusak/`inf` → outlier ditangani dengan *robust z-score* |z| > 5 **dihitung dari fold latih** → imputasi median **dari fold latih** → Z-score **dari fold latih**.

### 3.3 Penskalaan antar-blok (perbaikan atas dokumen)

Dokumen menormalisasi **hanya cabang sensor**, padahal blok visual berskala jauh lebih besar. Akibatnya KNN dan SVM-RBF didominasi visual karena satuan, bukan informasi.

Aturan beku: **standardisasi seluruh vektor gabungan setelah concatenation, fit di fold latih saja**. Simpan juga varian "tanpa penskalaan gabungan" khusus untuk KNN dan SVM sebagai eksperimen pembanding — ini menjawab tujuan KNN yang ditulis dokumen.

---

## 4. Protokol evaluasi anti-bocor (inti rencana)

Urutan yang **tidak boleh dilanggar**, per fold:

```
1. SPLIT tiap modalitas terpisah  →  2. PAIRING di dalam latih & di dalam uji
→  3. FIT scaler di latih  →  4. FIT model di latih  →  5. EVALUASI di uji
```

Rincian:
1. **Split visual**: `StratifiedGroupKFold` bila ada pengulangan objek/hari (grup = id objek/hari); kalau tidak ada, `StratifiedKFold`. **Split sensor**: stratifikasi kelas, grouped bila ada pengulangan run.
2. **Pairing**: per kelas, 1-ke-1 **tanpa pengembalian**, jumlah pasangan = min(N_visual, N_sensor) di kelas itu. Terjadi **di dalam latih** dan **di dalam uji** secara terpisah. Simpan `pair_id`, `visual_id`, `sensor_id` supaya bisa diaudit.
3. **Scaler**: fit hanya pada matriks latih.
4. **Model**: fit hanya pada latih.
5. **Unimodal memakai baris yang sama persis** dengan fused, supaya perbandingannya berpasangan, bukan kebetulan.

**Unit replikasi = seed pairing, bukan fold.** Untuk tiap seed pairing (R = **5**), jalankan CV **5 fold** → satu rata-rata per seed. Laporkan **mean ± sd antar 5 seed**. Ini jawaban langsung untuk kritik "semi-sintetis": hasilnya bukan artefak satu pairing yang beruntung.

**Matriks eksperimen:** 3 set fitur (visual / sensor / fused) × 4 model (RF, SVM-RBF, KNN, MLP) × 5 seed × 5 fold = **300 fit**. Pada ≤ beberapa ribu baris × 26 dimensi ini hitungan menit, bukan jam — asalkan fitur sudah di-cache.

**Metrik:** accuracy, macro-F1, precision, recall, ROC-AUC, confusion matrix. Macro-F1 sebagai metrik utama bila kelas tidak seimbang. Semua dilaporkan mean ± sd antar seed.

---

## 5. Fase kerja, keluaran, dan gerbang verifikasi

Setiap fase punya **gerbang**: tidak boleh lanjut sebelum gerbangnya hijau. Perintah verifikasi ditulis untuk PowerShell dari folder `RFPP`.

### F0 — Kerangka & lingkungan (½ hari)
**Langkah:** buat venv baru, `requirements.txt` versi dipatok, `config.yaml` lengkap, struktur folder, `git init` + `.gitignore` untuk `data/raw/` dan `cache/`.
**Keluaran:** repo kosong yang bisa diimpor.
**Gerbang:**
```powershell
cd "D:\Documents\S2\Kuliah\261\RFPP"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import skimage, sklearn, matplotlib, yaml, pandas; print('ok')"
# diharapkan: ok
.\.venv\Scripts\python.exe -c "import yaml; c=yaml.safe_load(open('config.yaml',encoding='utf-8')); print(sorted(c))"
# diharapkan: daftar kunci config, tanpa error
```

### F1 — Dataset: seleksi, unduh, audit (1–2 hari) · **GERBANG UTAMA**
**Kriteria seleksi visual (Kaggle):** citra permukaan daging, ada label segar/busuk (atau hari yang bisa dipetakan ke kelas), lisensi jelas, jumlah sampel per kelas tercatat, dan **dicatat apakah ada beberapa foto dari objek/hari yang sama** (menentukan grouped split).
**Kriteria seleksi sensor (UCI):** pembacaan resistansi/tegangan array gas, label normal/busuk atau bisa diturunkan secara terdokumentasi, lisensi jelas, jumlah kanal dan baris tercatat.
**Langkah:** unduh → susun `data/raw/{visual,sensor}/` → tulis `data/manifest.json` (nama dataset, versi/tanggal akses, URL, lisensi, jumlah sampel per kelas, jumlah kanal, sha256 tiap berkas, ada/tidaknya pengulangan objek).
**Keluaran:** manifest + tabel jumlah sampel.
**Gerbang:**
```powershell
.\.venv\Scripts\python.exe -m src.load_data
# diharapkan: tabel per kelas untuk kedua modalitas, jumlah kanal sensor,
# dan baris "pengulangan objek/hari: ada/tidak" — plus manifest tertulis.
# Kode keluar 0. Bila salah satu modalitas tidak punya dua kelas yang sah → BERHENTI,
# ganti dataset, jangan lanjut ke F2.
```
**Risiko terbesar proyek ada di sini.** Jika dataset UCI yang cocok ternyata tidak berlabel busuk, opsi: (a) dataset UCI lain, (b) turunkan kelas dari konsentrasi dengan ambang terdokumentasi. Apa pun pilihannya, tulis di manifest dan laporan.

### F2 — Fitur visual (1 hari)
**Langkah:** implementasi §3.1, cache ke `cache/features/visual.parquet` dengan kolom `visual_id, label, group_id, f00..f19`.
**Keluaran:** matriks N×20 + ringkasan statistik per fitur.
**Gerbang:**
```powershell
.\.venv\Scripts\python.exe -m src.features_visual
# diharapkan: "fitur visual: N baris x 20 kolom", nol NaN, nol kolom konstan,
# dan setiap baris histogram berjumlah 1 (toleransi 1e-6).
.\.venv\Scripts\python.exe -m pytest tests/test_features_visual.py -q
# diharapkan: lulus — termasuk uji bahwa dua citra identik menghasilkan vektor identik
# dan bahwa ROI tidak memuat piksel latar (uji sintetis).
```

### F3 — Fitur sensor (½ hari)
**Langkah:** implementasi §3.2, keputusan dimensi dicatat di `config.yaml` dan manifest, cache ke `cache/features/sensor.parquet`.
**Gerbang:**
```powershell
.\.venv\Scripts\python.exe -m src.features_sensor
# diharapkan: "fitur sensor: M baris x D kolom", D cocok dengan keputusan F1, nol NaN.
# Bila D != 6, dokumen metode dan diagram ikut diperbarui hari ini juga.
```

### F4 — Protokol: split, pairing, pipeline (1–2 hari) · **PEMEGANG INVARIAN**
**Langkah:** implementasi §4 di `src/protocol.py`. Satu fungsi yang mengembalikan (latih, uji) lengkap dengan id sumber, dipakai oleh **semua** eksperimen. Tidak boleh ada jalur lain yang melatih model.
**Keluaran:** modul protokol + uji anti-bocor.
**Gerbang (uji yang wajib ada):**
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_protocol.py -q
# diharapkan: lulus, minimal lima uji:
#  1. tidak ada visual_id yang muncul di latih dan uji sekaligus
#  2. tidak ada sensor_id yang muncul di latih dan uji sekaligus
#  3. tidak ada group_id (objek/hari) yang menyeberang split
#  4. setiap pasangan latih berlabel sama; setiap pasangan uji berlabel sama
#  5. scaler fit di latih: rata-rata fitur latih ~0, rata-rata fitur uji TIDAK dipaksa 0
```
Uji nomor 5 yang menangkap kebocoran scaler — uji yang paling sering dilupakan orang.

### F5 — Pelatihan & evaluasi komparatif (1–2 hari)
**Langkah:** 3 set fitur × 4 model × 5 seed × 5 fold, semua lewat `protocol.py`. Tulis `results/metrics.csv` (satu baris per kombinasi, dengan kolom seed dan fold) dan confusion matrix teragregasi.
**Gerbang:**
```powershell
.\.venv\Scripts\python.exe -m src.train_eval
# diharapkan: 300 baris hasil, kolom lengkap, dan ringkasan 3x4 mean±sd antar seed.
# Sanity: akurasi fused TIDAK boleh lebih rendah dari kedua unimodal secara sistematis
# pada semua model — bila itu terjadi, curigai penskalaan blok, bukan langsung disimpulkan.
```

### F5b — Probe diskordan B (½ hari)
**Desain:** buat pasangan **hanya-uji**: visual Segar ↔ sensor Busuk, dan visual Busuk ↔ sensor Segar. Tidak ada ground truth, jadi **jangan hitung akurasi**.
**Yang diukur:** (a) tingkat kesepakatan prediksi dengan label visual vs dengan label sensor; (b) distribusi kepercayaan model; (c) proporsi pasangan yang ditolak sebagai "Busuk" (sudut pandang keamanan pangan: sinyal gas busuk → tolak).
**Keluaran:** `results/probe.csv` + satu paragraf interpretasi dengan batas klaim eksplisit.
**Gerbang:**
```powershell
.\.venv\Scripts\python.exe -m src.probe_discordant
# diharapkan: dua blok hasil (Segar-Busuk, Busuk-Segar), masing-masing memuat
# agreement_visual, agreement_sensor, dan share_predicted_busuk. Nol pasangan latih terpakai.
```

### F5c — Dominasi modalitas (½ hari)
**Langkah:** *block-permutation importance* pada data uji — permutasi **seluruh 20 kolom visual bersama-sama**, lalu seluruh kolom sensor bersama-sama. Ini metode yang benar untuk pertanyaan "blok mana dominan", karena fitur HSV saling berkorelasi sehingga permutasi per-fitur meremehkan pentingnya blok. Tambahkan per-fitur top-10 sebagai pelengkap, dan konfirmasi dengan selisih *drop-one* dari F5.
**Gerbang:**
```powershell
.\.venv\Scripts\python.exe -m src.importance
# diharapkan: tabel penurunan macro-F1 per blok per model, plus top-10 per fitur.
# Angka blok harus konsisten arah dengan selisih drop-one F5; bila bertentangan, tulis
# kedua angka dan jelaskan, jangan pilih yang menguntungkan.
```

### F6 — Laporan, gambar, revisi dokumen (1–2 hari)
**Keluaran wajib:**
- Tabel 1: 3 set fitur × 4 model, mean ± sd antar seed (macro-F1, accuracy, AUC).
- Tabel 2: confusion matrix model terbaik per set fitur.
- Tabel 3: spesifikasi fitur beku + jumlah sampel per kelas + kardinalitas pairing per seed.
- Gambar 1: diagram alur (perbarui label dimensi bila D ≠ 6).
- Gambar 2: batang perbandingan dengan error bar antar seed.
- Gambar 3: block-permutation importance.
- Gambar 4: hasil probe diskordan.
- Bagian keterbatasan: pairing semi-sintetis, domain gap, korelasi tingkat kelas, batas klaim probe.
- Revisi §2/§4/§7 dokumen metode (daftar di §1 rencana ini).

**Gerbang:**
```powershell
.\.venv\Scripts\python.exe -m src.report
# diharapkan: seluruh tabel dan gambar tertulis di results/ dan figures/,
# dan setiap angka di laporan berasal dari metrics.csv (tidak ada angka diketik tangan).
```

---

## 6. Jadwal relatif (14 hari, dengan buffer)

| Hari | Kerja | Gerbang |
|---|---|---|
| H1 | F0 + mulai pencarian dataset | F0 hijau |
| H2–H3 | F1 seleksi, unduh, audit | **F1 hijau — gerbang utama** |
| H4 | F2 fitur visual | F2 hijau |
| H5 | F3 fitur sensor + keputusan dimensi | F3 hijau |
| H6–H7 | F4 protokol + uji anti-bocor | F4 hijau |
| H8–H9 | F5 matriks 3×4×5×5 | F5 hijau |
| H10 | F5b probe + F5c importance | keduanya hijau |
| H11–H12 | F6 laporan + gambar + revisi dokumen | F6 hijau |
| H13 | Baca ulang seluruh klaim terhadap §1; cek angka | tidak ada klaim melewati batas |
| H14 | Buffer / perbaikan | — |

Jika waktu ternyata hanya 8 hari: H13–H14 hilang, F6 dipadatkan ke H7, dan pemotongan mengikuti §7.

---

## 7. Risiko dan urutan pemotongan

| Risiko | Mitigasi |
|---|---|
| Dataset UCI tidak berlabel busuk | Kriteria seleksi F1 + dua opsi fallback; jangan lanjut sebelum F1 hijau |
| Citra punya pengulangan objek/hari | Grouped split; bila diabaikan, skor naik palsu |
| Kelas sangat tidak seimbang | Stratified split + macro-F1 sebagai metrik utama; laporkan jumlah per kelas |
| Waktu mepet | Urutan pemotongan di bawah |
| Skor fused terlihat terlalu bagus | Curigai kebocoran dulu (uji F4), bukan langsung diklaim |

**Urutan pemotongan bila waktu habis (dari yang pertama dipotong):**
1. Analisis skala KNN (varian tanpa penskalaan gabungan).
2. Jumlah seed pairing 5 → 3.
3. MLP (tiga model tersisa masih mewakili tiga paradigma).
4. Probe diskordan B — **dengan konsekuensi §2 dokumen harus kembali ke klaim A saja**.

**Yang tidak boleh dipotong apa pun alasannya:** baseline unimodal, dan protokol split-sebelum-pairing. Memotong keduanya membuat proyek tidak lagi menjawab pertanyaan fusi.

---

## 8. Pembagian kerja empat orang

| Orang | Tanggung jawab | Fase |
|---|---|---|
| 1 | Cabang visual: pra-pemrosesan, HSV, GLCM, cache | F2 |
| 2 | Cabang sensor: seleksi UCI, pembersihan, keputusan dimensi | F1(sensor), F3 |
| 3 | **Pemegang protokol**: split, pairing, pipeline, uji anti-bocor, train_eval | F4, F5 |
| 4 | Laporan, gambar, revisi dokumen metode, manifest & lisensi | F1(manifest), F6 |

Orang 3 adalah jalur kritis — disarankan dipegang oleh anggota yang paling nyaman dengan rekayasa perangkat lunak, karena satu kesalahan urutan di sana membatalkan seluruh angka. Orang 1 dan 2 **tidak boleh** melatih model sendiri; semua lewat `protocol.py`.

---

## 9. Yang masih perlu keputusan kelompok

1. **Dataset UCI spesifik** yang dipakai (nama + ID + lisensi) — baru bisa diputuskan setelah pencarian F1.
2. **Dimensi sensor akhir** bila kanal ≠ 6 (pakai semua vs PCA-6).
3. **Kriteria kelas busuk** bila dataset sensor tidak punya label biner langsung.
4. Apakah laporan akhir memakai format template kuliah tertentu — belum ada rubrik di folder ini.

---

## 10. Definisi selesai

Proyek dinyatakan selesai bila: seluruh gerbang F0–F6 hijau; `results/metrics.csv` memuat 300 baris (atau jumlah setelah pemotongan yang tercatat); setiap angka di laporan dapat ditelusuri ke CSV; daftar klaim §1 tidak dilanggar; dan dokumen metode sudah direvisi sehingga tidak lagi menjanjikan deteksi anomali dari data konkordan.

---

## 11. Catatan F1 — temuan terverifikasi (2026-11-11)

Bagian ini **menggantikan** asumsi dataset pada §3.2 dan §5-F1. Semua angka di bawah dibaca langsung dari berkas atau halaman sumber, bukan dari ingatan.

### 11.1 Sensor: klaim "UCI" di dokumen metode salah

Pencarian repositori UCI untuk *gas sensor / electronic nose* tidak mengembalikan satu pun dataset daging atau pangan berlabel segar/busuk. Sumber yang sah dan terverifikasi:

- **Mendeley Data `n8mc3nspfn` v1**, terbit 2025-10-03, lisensi **CC BY 4.0**.
- Berkas tunggal `Beef quality 4 classes.csv`, 1.246.972 byte.
- sha256 unduhan **cocok** dengan hash penerbit: `866367252b8498eddff2f488629e3e38341b2014a2e0a941d88e71a7499e7026`.
- Sudah diunduh ke `data/raw/sensor/Beef quality 4 classes.csv`.

Struktur nyata: **20.815 baris × 14 kolom** — `sl no`, `minutes`, `Temperature`, `Humidity`, delapan kanal resistansi MQ (`Mq-2, Mq-3, Mq-4, Mq-5, Mq-135, Mq-136, Mq-137, Mq-138`), `class`, `tvc`. Nol nilai hilang.

Pemetaan kelas **dari penerbit**, berbasis TVC (Total Viable Count, log10):

| class | label | ambang TVC | baris |
|---|---|---|---|
| 1 | Excellent | < 3 | 2.940 |
| 2 | Good | 3 – <4 | 2.268 |
| 3 | Acceptable | 4 – <5 | 2.292 |
| 4 | Spoiled | ≥ 5 | 13.315 |

### 11.2 Tiga konsekuensi yang tidak bisa ditawar

1. **Kelas adalah segmen waktu yang berurutan pada satu rekaman kontinu 29 jam.** `minutes` berjalan 1→1735 tanpa reset: kelas 1 = menit 1–245, kelas 2 = 246–434, kelas 3 = 435–625, kelas 4 = 626–1735. Pembacaan berjarak 5 detik. Maka: (a) split acak per baris adalah **kebocoran parah** — baris latih dan uji bisa berjarak 5 detik pada trayektori yang sama; (b) kelas dapat diprediksi dari waktu saja; (c) **tidak boleh ada klaim generalisasi lintas sampel daging** untuk modalitas sensor. Split wajib berbasis blok waktu, dan laporan harus menyebut sensor sebagai *satu trayektori pembusukan*.
2. **`tvc` dilarang menjadi fitur** — ia adalah sumber label, jadi memakainya = kebocoran langsung. `minutes` dan `sl no` juga dilarang (proksi waktu). `Temperature`/`Humidity` dikeluarkan dari modalitas gas: keduanya kanal lingkungan DHT11, bukan sensor gas, dan berkorelasi dengan waktu.
3. **Pseudo-replication harus dihapus**: 12 pembacaan per menit adalah pengulangan pengukuran, bukan sampel independen. Agregasi ke **rata-rata per menit** (≈1.735 baris) sebelum pairing; kardinalitas pairing dihitung dari baris teragregasi.

### 11.3 Dimensi berubah: 26 → 28

Modalitas gas = **8 kanal MQ**, bukan 6. Maka vektor gabungan = 20 + 8 = **28-D**. Dokumen metode (§3, §4, §5 diagram, dan setiap penyebutan "26") wajib diperbarui, bersama koreksi sumber UCI → Mendeley. Aturan §3.2 "pakai semua kanal bila >6" yang dipakai, bukan PCA.

### 11.4 Visual: dua kandidat terverifikasi

| Kandidat | Kelas | Jumlah | Resolusi | Lisensi | Catatan |
|---|---|---|---|---|---|
| Kaggle `crowww/meat-quality-assessment-based-on-deep-learning` | fresh / spoiled | 1.896 (948 per kelas, seimbang) | 1280×720 | CC BY 4.0 | Paper ASYU 2019 (Ulucan dkk.); pengulangan objek belum diketahui → audit F1 |
| Kaggle `vinayakshanawad/meat-freshness-image-dataset` | Fresh / Half-Fresh / Spoiled | 2.266 | 416×416 (stretch) | CC0 | Perlu aturan pemetaan/pembuangan Half-Fresh; resize stretch mengubah tekstur → risiko untuk GLCM |

Unduhan Kaggle memerlukan akun/token peneliti; bagian itu dijalankan oleh peneliti, bukan oleh agen.

### 11.5 Tambahan pada batas klaim §1

- Baseline sensor mengukur **pemisahan tahap pembusukan pada satu trayektori**, bukan generalisasi ke daging lain.
- Setiap angka sensor yang dilaporkan wajib menyebut agregasi per menit dan split blok waktu.
- Pasangan semi-sintetis kini menggabungkan *foto objek berbeda* (visual) dengan *tahap satu trayektori* (sensor); domain gap-nya lebih besar daripada yang ditulis dokumen, dan harus disebut eksplisit.

### 11.6 Keputusan kelompok (2026-11-11, sudah diambil)

1. **Aturan biner sensor = Opsi A**: segar = {1,2,3} (TVC<5), busuk = {4} (TVC≥5). Tercatat di `config.yaml` sebagai `sensor.class_map` beserta alasannya.
2. **Dataset visual = Kaggle `crowww/meat-quality-assessment-based-on-deep-learning`** (1.896 citra seimbang, CC BY 4.0).

Hasil audit pertama (`python -m src.load_data`, kode keluar 3 karena visual belum diunduh):
sensor 20.815 baris mentah → **1.735 baris teragregasi** (segar 625 / busuk 1.110 menit),
**59 blok waktu 30 menit** (segar 21 / busuk 38), sha256 cocok penerbit, 8 kanal.
Angka-angka ini menjadi kardinalitas pairing dan unit split pada F4/F5.

## 12. Catatan F3 + F4 — hasil terverifikasi (2026-11-11)

### 12.1 F3 lulus (cache fitur sensor)

```powershell
cd "D:\Documents\S2\Kuliah\261\RFPP"
.\.venv\Scripts\python.exe -m src.features_sensor
# keluaran: fitur sensor: 1735 baris x 8 kanal -> cache\features\sensor.parquet
#           per kelas: {'spoiled': 1110, 'fresh': 625}
#           blok waktu per kelas: {'fresh': 21, 'spoiled': 38}
# kode keluar 0
```

Yang disimpan di cache adalah **8 kanal teragregasi apa adanya** (belum di-clip, belum
diskalakan, belum diimputasi). Semua statistik itu dihitung di dalam fold latih oleh
`src/protocol.py`, supaya cache tidak pernah membawa informasi dari data uji.
Dimensi 8 (bukan 6) mengunci revisi dokumen: fused = 20 + 8 = **28**.

### 12.2 F4 lulus (8 uji anti-bocor, data sintetis)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_protocol.py -q
# diharapkan: 8 passed
```

| Uji | Invarian yang dijaga |
| --- | --- |
| `test_tidak_ada_visual_id_menyeberang` | satu citra tidak pernah ada di latih dan uji |
| `test_tidak_ada_sensor_id_menyeberang` | satu menit tidak pernah ada di latih dan uji |
| `test_group_tidak_menyeberang` | blok waktu/objek tidak menyeberang split (kedua modalitas) |
| `test_pasangan_berlabel_sama` | `v_label == s_label == label` di latih maupun uji |
| `test_pairing_tanpa_pengembalian` | `visual_id` dan `sensor_id` unik dalam satu frame |
| `test_kardinalitas_min_per_kelas` | jumlah pasangan = min(n_visual, n_sensor) per kelas, dihitung dari indeks split |
| `test_scaler_fit_hanya_di_latih` | rata-rata fitur latih ≈ 0, statistik scaler ≠ rata-rata gabungan, rata-rata fitur uji tidak dipaksa 0 |
| `test_unimodal_memakai_baris_yang_sama` | matriks visual/sensor adalah irisan kolom persis dari matriks fused |

Satu kegagalan awal adalah **bug pada pernyataan uji, bukan kebocoran protokol**:
`StandardScaler` duduk setelah `RobustZClip`, jadi `scale.mean_` sama dengan rata-rata
data latih *yang sudah di-clip*, bukan rata-rata data latih mentah. Uji diperbaiki agar
membandingkan dengan `clip.transform(x_tr).mean(axis=0)`; `src/protocol.py` tidak diubah.

### 12.3 Split sensor diuji pada data nyata

`StratifiedGroupKFold(5, shuffle, random_state=20261111)` atas 1.735 menit / 59 blok:

| Fold | Latih (segar/busuk) | Uji (segar/busuk) | Irisan grup |
| --- | --- | --- | --- |
| 1 | 1380 (505/875) | 355 (120/235) | 0 |
| 2 | 1375 (480/895) | 360 (145/215) | 0 |
| 3 | 1405 (505/900) | 330 (120/210) | 0 |
| 4 | 1405 (505/900) | 330 (120/210) | 0 |
| 5 | 1375 (505/870) | 360 (120/240) | 0 |

Kelima fold berbeda (12/12/11/11/13 blok uji), tidak ada grup yang menyeberang.
Plafon pairing per fold ≈ min per kelas: dengan visual crowww 1.896 citra seimbang
(±190 uji per kelas), kelas segar dibatasi sensor (±120) dan kelas busuk dibatasi
visual (±190) → **±310 pasangan uji dan ±1.250 pasangan latih per fold**. Angka pasti
dihitung ulang setelah F2, karena bergantung pada jumlah citra yang benar-benar lolos audit.

### 12.4 Dependensi tambahan dipatok

`pyarrow==26.0.1` (cache parquet) dan `pytest==9.2.0` (gerbang uji) ditambahkan ke
`requirements.txt`; `pip install -r requirements.txt` melaporkan seluruhnya sudah terpenuhi.

### 12.5 Yang masih menunggu

- **F1 sisi visual**: `python -m src.load_data` masih kode keluar 3 sampai citra Kaggle
  crowww diekstrak ke `data/raw/visual/fresh` dan `data/raw/visual/spoiled`.
  Unduhan memakai akun peneliti sendiri, bukan agen.
- **F2** (fitur visual 20-D) baru bisa dimulai setelah gerbang F1 kode keluar 0.
- **F5/F5b/F5c** menunggu kedua cache fitur ada; jalur pelatihannya sudah siap dan terkunci
  oleh 8 uji di atas.

## 13. Catatan F1 visual + F2 — temuan yang mengubah asumsi (2026-11-11)

### 13.1 Dataset visual ternyata SATU trayektori, bukan 1.896 objek independen

Terbaca dari nama berkas (`test_YYYYMMDD_HHMMSSD.jpg`) dan diverifikasi terhadap isi berkas:

| Fakta | Nilai terverifikasi |
| --- | --- |
| Rentang rekaman | 2017-10-16 10:43:21 → 2017-10-19 03:09:21 (±64,4 jam) |
| Interval frame | 2 menit (modus), 1 sesi kontinu per kelas, tanpa jeda >10 menit |
| Titik ganti label | frame terakhir `Fresh` 17 Okt 18:59:21 → frame pertama `Spoiled` 17 Okt 19:01:21 (jeda 2 menit) |
| Resolusi | 1280×720 untuk semua citra |
| Duplikat persis | 0 (1.896 sha256 unik) |
| Kedekatan frame berurutan | rata-rata \|selisih\| 4,6–10,1 tingkat keabuan → sangat berkorelasi |
| EXIF datetime | 0/10 sampel per kelas; timestamp hanya ada di nama berkas |
| Lisensi | CC BY 4.0, teks lengkap di `data/raw/visual/license.txt`; README menyebut IP camera + supermarket Izmir |

Konsekuensi: split baris acak akan menempatkan frame kembar di latih dan uji sekaligus.
Visual wajib memakai **blok waktu** sebagai unit split, persis seperti sensor.

### 13.2 Keputusan kelompok (2026-11-11, sudah diambil)

1. **`group_id` visual = blok 30 menit** dari timestamp nama berkas, dihitung di dalam
   segmen kelas masing-masing (tidak ada blok campuran kelas). Selaras dengan
   `sensor.block_minutes: 30` → satu aturan waktu untuk kedua modalitas.
2. **Batas kelas: buang 2 blok yang bersebelahan** (blok terakhir kelas awal + blok
   pertama kelas akhir) → 24 frame dibuang (9 segar + 15 busuk).
   Sisa **1.872 frame (939 segar / 933 busuk), 128 blok (64 / 64)**. Tercatat di
   `config.yaml` sebagai `visual.boundary.rule: drop_adjacent_blocks` beserta alasannya.
3. **Baris tetap per-citra, bukan agregasi per blok.** Agregasi visual ke 128 blok akan
   menyisakan hanya ±26 pasangan uji per fold — terlalu sedikit untuk MLP/SVM yang stabil.
   Korelasi antar-frame dalam satu blok diakui sebagai keterbatasan; unit replikasi tetap
   `pairing_seed` (5 seed), sesuai §4.

### 13.3 Tambahan pada batas klaim §1 (wajib dipakai di laporan)

- **Dilarang**: menyebut visual sebagai "1.896 citra objek berbeda", dan menyebut hasil
  sebagai generalisasi ke daging/objek lain — untuk **kedua** modalitas.
- **Wajib disebut**: studi ini membandingkan fusi vs unimodal pada **dua trayektori
  berbeda** (sapi Izmir 2017 vs sapi Mendeley), sehingga pasangan bersifat semi-sintetis
  dengan domain gap besar (beda hewan, negara, tahun, kamera).
- **Wajib disebut**: label visual adalah konvensi penerbit pada satu titik waktu, bukan
  transisi yang terlihat di citra; 24 frame di sekitar titik itu dibuang.

### 13.4 F2 lulus (cache fitur visual 20-D)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_features_visual.py -q   # 6 passed
.\.venv\Scripts\python.exe -m src.features_visual
# fitur visual: 1872 baris x 20 kolom -> cache\features\visual.parquet
# per kelas: {'fresh': 939, 'spoiled': 933} | blok: {'fresh': 64, 'spoiled': 64}
# kode keluar 0
```

Urutan fitur beku: `f00–f07` histogram H (8 bin), `f08–f11` S (4), `f12–f15` V (4),
`f16–f19` GLCM contrast/homogeneity/energy/correlation. Pra-pemrosesan sesuai §3.1:
resize 256×256 (bicubic) → center crop 80% (204×204) → Gaussian σ=1,0 → HSV;
GLCM atas grayscale ROI yang dikuantisasi ke 64 level, jarak 1, 4 sudut dirata-ratakan.

Enam uji sintetis yang mengunci spesifikasi: dimensi tepat 20 dan berhingga/negatif-nol;
tiap kelompok histogram berjumlah 1 (tol 1e-6); dua citra identik → vektor identik;
**ROI membuang latar** (tepi merah 26 piksel hilang, bin H merah = 0, bin hijau = 1);
**GLCM merespons tekstur** (contrast papan catur > seragam, energy < seragam);
**rotasi 90° tidak mengubah vektor 20-D** (bukti rata-rata 4 sudut + ROI persegi bekerja).

Nilai fitur maksimum 10,76 (GLCM contrast, skala 64 level) sementara histogram ≤ 1 →
menguatkan §3.3: standardisasi seluruh vektor gabungan pasca-concat memang diperlukan.

### 13.5 Protokol F4 diverifikasi pada data nyata (bukan hanya sintetis)

`iter_folds` atas kedua cache, 5 seed × 5 fold = **25 kombinasi**, semua invarian lolos
(id tidak menyeberang, grup tidak menyeberang, pasangan berlabel sama, pairing unik,
fused 28-D, seluruh nilai berhingga):

| Fold | Latih (segar/busuk) | Uji (segar/busuk) |
| --- | --- | --- |
| 1 | 1247 (505/742) | 311 (120/191) |
| 2 | 1222 (480/742) | 336 (145/191) |
| 3 | 1257 (505/752) | 301 (120/181) |
| 4 | 1259 (505/754) | 299 (120/179) |
| 5 | 1247 (505/742) | 311 (120/191) |

Kelas segar selalu dibatasi sensor (±120 menit uji), kelas busuk dibatasi visual
(±180–190 citra uji) — sesuai plafon yang diperkirakan di §12.3.

### 13.6 Status fase

F0 ✓ · F1 ✓ (kode keluar 0, kedua modalitas) · F2 ✓ · F3 ✓ · F4 ✓ (**14 uji lulus**).
Berikutnya **F5** (300 fit komparatif) — menunggu persetujuan, lalu F5b/F5c dan F6.

## 14. Catatan F5 — hasil evaluasi komparatif (2026-11-11)

### 14.1 Gerbang lulus

```powershell
.\.venv\Scripts\python.exe -m pytest -q          # 17 passed (3 uji baru khusus F5)
.\.venv\Scripts\python.exe -m src.train_eval     # kode keluar 0, ±146 s
# metrics.csv: 350 baris (300 inti + 50 varian penskalaan), 0 NaN, 350 fit unik
```

Keluaran di `results/`: `metrics.csv` (satu baris per fit, kolom seed/fold/feature_set/
model/scaling/n_train/n_test/detik + 5 metrik), `summary.csv` (mean±sd **antar seed**),
`confusion.csv` (teragregasi atas 25 kombinasi), `paired_delta.csv` (selisih per seed).
Tiga uji baru mengunci semantik agregasi: grid = 300 inti + 50 varian; fold dirata-ratakan
di dalam seed sebelum sd dihitung (sd antar-seed ≠ sd antar-fit); selisih berpasangan
dihitung per seed. Dua bug ditemukan dan diperbaiki sebelum angka final: gerbang semula
membandingkan *urutan* kolom (bukan himpunan), dan `paired_delta` mengiterasi model dari
config alih-alih dari data. Keduanya tidak mengubah angka hasil.

### 14.2 Macro-F1 mean±sd antar 5 seed pairing (penskalaan standar)

| Set fitur | Random Forest | SVM-RBF | KNN | MLP |
| --- | --- | --- | --- | --- |
| visual (20-D) | 0,9959 ± 0,0010 | 0,9902 ± 0,0009 | 0,9909 ± 0,0009 | 0,9788 ± 0,0133 |
| sensor (8-D) | 0,9600 ± 0,0041 | 0,9621 ± 0,0009 | 0,9668 ± 0,0025 | 0,9503 ± 0,0167 |
| **fused (28-D)** | **0,9981 ± 0,0007** | **0,9983 ± 0,0003** | **0,9964 ± 0,0008** | **0,9841 ± 0,0057** |

### 14.3 Klaim A — fused vs unimodal (selisih berpasangan, baris identik)

- **Terhadap sensor: 20/20 perbandingan positif** (5 seed × 4 model), rentang +0,0229 … +0,0582.
- **Terhadap visual: 19/20 positif**; satu-satunya kekalahan adalah MLP seed 44 (−0,0034).
  Rata-rata selisih: RF +0,0022 · SVM +0,0080 · KNN +0,0055 · MLP +0,0053.
- Uji tanda satu sisi atas 5 seed: RF/SVM/KNN menang 5/5 → p = 1/32 = 0,031; MLP menang
  4/5 → p = 0,188 (tidak signifikan). Dengan n = 5 seed, hanya itu kekuatan statistik yang sah.
- **Cara menulis yang jujur**: fusi memberi perbaikan *kecil tetapi konsisten* atas visual
  dan *lebih besar* atas sensor. Jangan menulis "fusi jauh lebih unggul" — visual sendiri
  sudah ≈0,99 (efek plafon).

### 14.4 Klaim 4 — efek penskalaan gabungan (fused, model berbasis jarak)

| Model | standard | none | selisih |
| --- | --- | --- | --- |
| KNN | 0,9964 ± 0,0008 | 0,9586 ± 0,0020 | **−0,0378** |
| SVM-RBF | 0,9983 ± 0,0003 | 0,9701 ± 0,0013 | **−0,0282** |

Koreksi §3.3 atas dokumen metode terbukti empiris: tanpa standardisasi seluruh vektor
gabungan, model jarak kehilangan 3–4 poin macro-F1. (`none` = tanpa penskalaan
pasca-concat; clip outlier tetap aktif.)

### 14.5 Confusion matrix teragregasi (positif = busuk; 7.790 baris uji per konfigurasi)

| Set | Model | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- |
| fused | random_forest | 3111 | 14 | 0 | 4665 |
| fused | svm_rbf | 3113 | 12 | 2 | 4663 |
| fused | knn | 3098 | 27 | 2 | 4663 |
| fused | mlp | 3096 | 29 | 97 | 4568 |
| visual | knn | 3054 | 71 | 0 | 4665 |
| visual | svm_rbf | 3049 | 76 | 0 | 4665 |
| visual | random_forest | 3100 | 25 | 5 | 4660 |
| sensor | random_forest | 2903 | 222 | 84 | 4581 |
| sensor | svm_rbf | 2968 | 157 | 134 | 4531 |

Sudut keamanan pangan: visual saja tidak pernah melewatkan busuk (FN = 0 pada KNN/SVM)
tetapi menimbulkan 71–76 alarm palsu; **fused RF menekan FP ke 14 dengan FN tetap 0**.
Sensor saja paling lemah di kedua arah (FP 157–222, FN 84–168).

### 14.6 Batas interpretasi WAJIB (jangan dilewatkan di laporan)

1. Skor absolut 0,95–0,998 **tidak boleh** ditulis sebagai "akurasi deteksi kebusukan
   daging". Kedua modalitas adalah trayektori tunggal dengan kelas = segmen waktu.
   Split blok mencegah frame kembar menyeberang, tetapi **tidak mencegah interpolasi
   temporal**: blok uji tetap dikelilingi blok latih dari waktu yang berdekatan.
2. **Konfound waktu**: karena kelas adalah segmen waktu, drift monoton apa pun
   (pencahayaan kamera, baseline sensor) dapat memisahkan kelas. Desain ini tidak
   memisahkan "kualitas daging" dari "waktu rekaman".
3. Efek plafon membuat selisih fused−visual kecil; laporkan selisih berpasangan dan
   jumlah seed yang menang, bukan hanya angka akurasi.
4. Opsional, **belum dikerjakan dan butuh persetujuan**: uji ketahanan dengan *embargo*
   (buang blok latih yang bersebelahan dengan blok uji) untuk mengukur berapa besar
   interpolasi temporal menyumbang skor. Bila hasilnya turun banyak, itu wajib dilaporkan.

### 14.7 Status fase

F0 ✓ · F1 ✓ · F2 ✓ · F3 ✓ · F4 ✓ · **F5 ✓ (17 uji lulus, 350 fit)**.
Berikutnya F5b (probe diskordan), F5c (dominasi modalitas), lalu F6 (laporan + revisi dokumen).

## 15. Catatan F5b — probe pasangan diskordan (2026-11-11)

### 15.1 Gerbang lulus

```powershell
.\.venv\Scripts\python.exe -m src.probe_discordant   # kode keluar 0
# probe.csv: 200 baris | total pasangan uji diskordan: 31.280
# nol pasangan latih terpakai (diperiksa per fold); akurasi tidak dihitung
```

Pasangan dibuat dari **split uji fold yang sama** dengan model yang dilatih pada pasangan
konkordan fold itu. Kolom `label` **sengaja dibuang** dari frame diskordan supaya akurasi
tidak mungkin terhitung. Gerbang tambahan yang diverifikasi: `v_label != s_label` untuk
semua baris, id unik, id hanya dari split uji, dan `agreement_visual + agreement_sensor = 1`.

### 15.2 Hasil (mean±sd antar 5 seed; 4 model fused, penskalaan standar)

**Arah segar-busuk** (visual segar ↔ sensor busuk):

| Model | ikut visual | ikut sensor | prediksi busuk | confidence |
| --- | --- | --- | --- | --- |
| random_forest | 0,8550 ± 0,0553 | 0,1450 | 0,1450 | 0,6418 (proba) |
| svm_rbf | 0,6622 ± 0,0245 | 0,3378 | 0,3378 | 0,4287 (margin) |
| knn | 0,6706 ± 0,0050 | 0,3294 | 0,3294 | 0,9627 (proba) |
| mlp | 0,7132 ± 0,0139 | 0,2868 | 0,2868 | 0,8074 (proba) |

**Arah busuk-segar** (visual busuk ↔ sensor segar):

| Model | ikut visual | ikut sensor | prediksi busuk | confidence |
| --- | --- | --- | --- | --- |
| random_forest | 0,6750 ± 0,0748 | 0,3250 | 0,6750 | 0,6780 (proba) |
| svm_rbf | 0,8131 ± 0,0133 | 0,1869 | 0,8131 | 0,6658 (margin) |
| knn | 0,8314 ± 0,0109 | 0,1686 | 0,8314 | 0,9695 (proba) |
| mlp | 0,7793 ± 0,0207 | 0,2207 | 0,7793 | 0,8112 (proba) |

### 15.3 Interpretasi dan batas klaim

- **Visual dominan di kedua arah** (66–86% keputusan mengikuti label visual), konsisten
  dengan F5 (visual sendiri ≈0,99 vs sensor ≈0,96) dan dengan F5c.
- **Sudut keamanan pangan**: pada arah segar-busuk, model cenderung *menerima*
  (prediksi segar) walau gas menyatakan busuk — `share_predicted_busuk` hanya 0,145–0,338.
  Ini risiko nyata dari fusi yang didominasi visual, dan layak jadi satu paragraf diskusi.
- `agreement_visual` dan `agreement_sensor` **bukan dua bukti independen**: jumlahnya
  selalu 1 pada pasangan diskordan. Laporkan satu saja plus arah, jangan diklaim ganda.
- **Skala confidence tidak seragam**: KNN/RF/MLP memakai `max_proba` (0–1), SVM memakai
  `|decision_function|` (margin, bukan probabilitas). Kolom `confidence_kind` mencatat
  jenisnya. **Dilarang membandingkan angka confidence antar model**; yang sah adalah
  pola dalam model yang sama (mis. KNN sangat percaya diri 0,96–0,97 padahal keputusannya
  mengikuti visual).
- Tidak ada ground truth pasangan diskordan → **dilarang menyebut akurasi probe**.
  Ini ukuran kecenderungan model, bukan kinerja deteksi (batas klaim §1 nomor 3).

## 16. Catatan F5c — dominasi modalitas (2026-11-11)

### 16.1 Gerbang lulus

```powershell
.\.venv\Scripts\python.exe -m src.importance   # kode keluar 0, total 1545,6 s (±26 menit)
.\.venv\Scripts\python.exe -m pytest -q        # 22 passed
```

`results/importance.csv` 200 baris (25 kombinasi × 4 model × 2 blok),
`importance_per_feature.csv` 2.800 baris (× 28 fitur), `importance_summary.csv` 8 baris,
`importance_top10.csv` 40 baris; semuanya 0 NaN. Beban sebenarnya adalah
**60.000 prediksi** (100 fit × 30 target × 20 permutasi), bukan 100 fit — perkiraan waktu
awal meleset karena menghitung fit, bukan prediksi.

### 16.2 Penurunan macro-F1 per blok (mean±sd antar 5 seed, 20 permutasi/uji)

| Model | blok visual (20 kolom) | blok sensor (8 kolom) | rasio |
| --- | --- | --- | --- |
| random_forest | **+0,3777 ± 0,0183** | +0,1159 ± 0,0179 | 3,3× |
| svm_rbf | **+0,3715 ± 0,0073** | +0,1341 ± 0,0072 | 2,8× |
| knn | **+0,3775 ± 0,0035** | +0,1263 ± 0,0023 | 3,0× |
| mlp | **+0,3634 ± 0,0076** | +0,1245 ± 0,0017 | 2,9× |

Baseline macro-F1 uji rata-rata 0,9942. **Klaim 2 terpenuhi**: blok visual lebih menentukan
daripada blok sensor pada keempat model, arah konsisten dengan selisih drop-one F5
(`searah` untuk RF/SVM/KNN/MLP).

### 16.3 Dua jebakan interpretasi yang wajib dijelaskan di laporan

1. **Per-fitur jauh lebih kecil daripada per-blok, dan itu bukan berarti fitur tidak penting.**
   Penurunan maksimum per-fitur 0,0528 vs per-blok 0,4727 (beda satu orde). Penyebabnya
   redundansi internal blok: memindahkan satu bin HSV saja hampir tidak merusak keputusan
   karena bin lain membawa informasi yang sama. Inilah alasan rencana memilih
   *block-permutation* — permutasi per-fitur meremehkan pentingnya blok.
2. **Permutasi (0,37) dan drop-one (0,038) tidak boleh dibandingkan besarnya.**
   Permutasi merusak blok **tanpa melatih ulang**, sehingga model tidak bisa berkompensasi;
   drop-one **melatih ulang** model tanpa modalitas itu, sehingga sisa modalitas menutup
   sebagian besar kehilangan. Keduanya menjawab pertanyaan berbeda; yang dikonfirmasi
   adalah **arah/urutannya**, dan itu searah pada keempat model. Tulis kedua angka.

### 16.4 Fitur teratas (pelengkap, mean penurunan antar seed)

Pemetaan nama: `f00–f07` histogram H, `f08–f11` S, `f12–f15` V, `f16–f19` GLCM
(contrast, homogeneity, energy, correlation).

- **MLP**: f09 +0,0185 · f11 +0,0095 · f19 +0,0054 · f08 +0,0043 → bin **saturasi** dominan.
- **KNN**: f15 +0,0041 · Mq-2 +0,0031 · Mq-135 +0,0020 · f16 +0,0012 → bin V terang + GLCM contrast.
- **SVM-RBF**: Mq-2 +0,0047 · Mq-4 +0,0030 · Mq-135 +0,0030 → justru kanal gas teratas.
- **RF**: f08 +0,0014 · Mq-135 +0,0012 · f10 +0,0006 → semua sangat kecil (redundansi tinggi).

Catatan jujur: urutan per-fitur **tidak stabil antar model** (SVM menempatkan gas di atas,
MLP menempatkan saturasi di atas). Jangan menyimpulkan "fitur X paling penting" secara
lintas model; laporkan per model, dan andalkan tingkat blok untuk klaim dominasi.

### 16.5 Batas tambahan

Importance diukur pada data uji dari trayektori yang sama, sehingga mewarisi seluruh batas
§14.6 (interpolasi temporal, konfound waktu, efek plafon). Angka ini menjawab "blok mana
yang dipakai model", **bukan** "modalitas mana yang secara fisik mendeteksi kebusukan".

### 16.6 Status fase

F0 ✓ · F1 ✓ · F2 ✓ · F3 ✓ · F4 ✓ · F5 ✓ · F5b ✓ · **F5c ✓ (22 uji lulus)**.
Tinggal **F6**: laporan, gambar, dan revisi dokumen metode (26→28, UCI→Mendeley, §2/§4/§7).

## 17. Notebook eksplorasi & evaluasi (2026-11-11)

`notebooks/01_eksplorasi_dan_evaluasi.ipynb` (kernel `Python (RFPP)`, 44 sel, 23 sel kode,
21 gambar tertanam, dieksekusi penuh tanpa galat). Notebook **hanya membaca**
`results/*.csv`, `cache/features/*.parquet`, `data/manifest.json`, dan `config.yaml` —
tidak ada pelatihan ulang, jadi angkanya identik dengan `results/`. Semua gambar juga
ditulis ke `figures/01_…png` s.d. `figures/21_…png` untuk dipakai F6.
Dependensi notebook dipatok di `requirements.txt`: `ipykernel==7.4.0`,
`nbclient==0.11.0`, `nbformat==5.11.1`.

Isi: inventaris data; trajektori 8 kanal e-nose + audit blok batas; boxplot dan korelasi
sensor; trajektori visual + contoh citra mentah dan ROI pra-proses; fitur visual sepanjang
waktu; histogram per kelas; korelasi 20×20 visual; ukuran fold dan komposisi uji;
macro-F1 per set; heatmap 5 metrik; selisih berpasangan per seed + uji tanda; efek
scaling; confusion matrix 3×4; probe diskordan + tabel confidence; importance blok;
permutasi vs drop-one; top-10 per fitur; dan bagian batas klaim wajib.

### 17.1 Temuan baru: asimetri perlakuan blok batas antar modalitas

Blok 30 menit sensor ke-20 (menit 601–630) memuat **dua kelas**: 25 menit segar +
5 menit busuk. Visual membuang dua blok di batas kelas (24 frame), tetapi **sensor
tidak punya aturan `boundary` di `config.yaml`**, sehingga blok campuran itu dipakai utuh.

Yang sudah diverifikasi (sel audit notebook, memakai `fold_splits` yang sama dengan eksperimen):

- **Bukan kebocoran antar-fold.** `StratifiedGroupKFold` memindahkan blok secara utuh;
  irisan blok latih∩uji = 0 pada kelima fold untuk kedua modalitas (selaras dengan
  `tests/test_protocol.py`).
- **Tetapi ada ambiguitas label di dalam blok**: menit transisi pembusukan dipaksakan
  biner oleh `class_map`. Ini derau label, bukan kebocoran.

**Keputusan terbuka (belum dijalankan, menunggu persetujuan kelompok):**
(a) pertahankan cache apa adanya dan tulis asimetri ini sebagai keterbatasan protokol
di laporan; atau (b) terapkan `drop_adjacent_blocks` pada sensor lalu jalankan ulang
F5–F5c (±30 menit) dan perbarui seluruh angka. Semua angka §14–§16 dan notebook saat
ini memakai opsi (a). Jangan mengklaim kedua modalitas diperlakukan simetris di batas
kelas sebelum keputusan ini diambil.