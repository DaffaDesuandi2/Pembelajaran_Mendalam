# Studio 2 — Klasifikasi Sampah

**Kelompok 1 | Deep Learning (IF25-40401)**  
Program Studi Teknik Informatika — Institut Teknologi Sumatera (ITERA)  
Semester Ganjil 2026/2027

## 1. Anggota Kelompok

| No. | Nama Lengkap | NIM |
|---|---|---|
| 1 | Muhammad Daffansyah Desuandi | 123140127 |
| 2 | Nadya Shafwah Yusuf | 123140167 |
| 3 | *Belum diisi* | *Belum diisi* |

## 2. Deskripsi Proyek

Proyek ini bertujuan mengembangkan model *deep learning* untuk mengklasifikasikan gambar sampah ke dalam lima kategori:

1. `botol_plastik`
2. `gelas_plastik`
3. `kertas_kardus`
4. `saset_kemasan`
5. `styrofoam`

Dataset yang digunakan terdiri atas 2.753 gambar pelatihan. Evaluasi internal dilakukan menggunakan 410 gambar yang dipisahkan dari data pelatihan.

Metrik utama yang digunakan adalah **Macro-F1 Score**, yaitu rata-rata F1-score dari seluruh kelas dengan bobot yang sama.

## 3. Arsitektur Model

Model akhir menggunakan **ConvNeXt-Tiny** dengan pendekatan transfer learning dan fine-tuning.

| Komponen | Konfigurasi |
|---|---|
| Arsitektur | ConvNeXt-Tiny |
| Library | PyTorch, torchvision, timm |
| Pretrained weights | ImageNet-21K → ImageNet-1K |
| Model identifier | `convnext_tiny.fb_in22k_ft_in1k` |
| Jumlah kelas | 5 |
| Resolusi input | 224 × 224 |
| Optimizer | AdamW |
| Learning rate scheduler | Cosine Annealing |
| Loss function | CrossEntropyLoss |
| Label smoothing | 0,1 |
| Mixed precision | Digunakan |
| Fine-tuning V4 | 18 epoch |
| Pemilihan checkpoint | Macro-F1 validasi lokal tertinggi |

### Preprocessing

Pada proses evaluasi, gambar melalui transformasi berikut:

- Resize menjadi 256 piksel pada sisi yang lebih pendek.
- Center crop berukuran 224 × 224 piksel.
- Konversi gambar menjadi tensor.
- Normalisasi menggunakan mean dan standard deviation ImageNet.

### Strategi Pelatihan

Model dikembangkan melalui beberapa eksperimen untuk meningkatkan kemampuan generalisasi, terutama pada kategori yang sulit dibedakan seperti `kertas_kardus` dan `styrofoam`.

Pada konfigurasi V4, digunakan pretrained weights ImageNet-21K yang telah di-fine-tune pada ImageNet-1K, gradient accumulation, mixed precision, dan fine-tuning pada backbone ConvNeXt.

Tidak ada penambahan dataset eksternal yang tercatat dalam eksperimen kelompok. ImageNet digunakan sebagai sumber pretrained weights.

## 4. Eksperimen dan Hasil

### 4.1 Perbandingan Eksperimen

| Eksperimen | Pendekatan | Macro-F1 Lokal | Akurasi Lokal |
|---|---|---:|---:|
| Baseline | ResNet-18 pretrained ImageNet | 0,9230 | 92,68% |
| Preview 2 | ConvNeXt-Tiny ImageNet-1K | 0,9418 | 94,63% |
| V3 | ConvNeXt-Tiny + class weight 1,5 pada `kertas_kardus` | 0,9367 | 94,15% |
| **V4 (Final)** | **ConvNeXt-Tiny ImageNet-21K → 1K** | **0,9529** | **95,61%** |

Seluruh nilai pada tabel merupakan hasil validasi lokal menggunakan 410 gambar, bukan hasil validation set tersembunyi milik dosen.

### 4.2 Hasil Evaluasi Model Final

| Kelas | F1-score |
|---|---:|
| Botol plastik | 0,9686 |
| Gelas plastik | 0,9610 |
| Kertas/kardus | 0,9143 |
| Saset kemasan | 0,9751 |
| Styrofoam | 0,9453 |
| **Macro-F1** | **0,9529** |

**Akurasi: 95,61% (392 dari 410 gambar diklasifikasikan dengan benar).**

### 4.3 Confusion Matrix Model Final

Baris merupakan kelas sebenarnya, sedangkan kolom merupakan hasil prediksi.

| Aktual \ Prediksi | Botol | Gelas | Kertas | Saset | Styrofoam |
|---|---:|---:|---:|---:|---:|
| Botol plastik | 77 | 0 | 0 | 1 | 0 |
| Gelas plastik | 1 | 74 | 0 | 0 | 3 |
| Kertas/kardus | 1 | 1 | 48 | 1 | 3 |
| Saset kemasan | 1 | 0 | 1 | 98 | 0 |
| Styrofoam | 1 | 1 | 2 | 1 | 95 |

### 4.4 Analisis Eksperimen

**Baseline ResNet-18**

Model awal menghasilkan Macro-F1 0,9230. Kesalahan klasifikasi paling menonjol ditemukan pada kategori `kertas_kardus`.

**ConvNeXt-Tiny Preview 2**

Penggantian backbone menjadi ConvNeXt-Tiny meningkatkan Macro-F1 lokal menjadi 0,9418. Namun, F1 kategori `kertas_kardus` masih berada pada 0,8800.

**Eksperimen V3 — Class Weighting**

Pemberian bobot loss sebesar 1,5 untuk kategori `kertas_kardus` meningkatkan recall kelas tersebut, tetapi menurunkan precision. Macro-F1 keseluruhan turun menjadi 0,9367.

Eksperimen ini menunjukkan bahwa meningkatkan penalti kesalahan pada kelas tertentu tidak selalu memperbaiki performa keseluruhan.

**Eksperimen V4 — ImageNet-21K**

Penggunaan pretrained weights ImageNet-21K → 1K dan perubahan konfigurasi fine-tuning menghasilkan Macro-F1 lokal terbaik sebesar 0,9529.

F1 kelas `kertas_kardus` meningkat dari 0,8800 pada Preview 2 menjadi 0,9143 pada V4. Peningkatan juga terjadi pada kelas `saset_kemasan` dan `styrofoam`.

Model V4 dipilih sebagai kandidat Final karena memperoleh Macro-F1 tertinggi pada validasi lokal.

## 5. Hasil Kompetisi

| Putaran | Macro-F1 Lokal | Macro-F1 Resmi |
|---|---:|---|
| Preview 1 | 0,9230 (baseline eksperimen awal) | Belum dicantumkan |
| Preview 2 | 0,9418 | Belum dicantumkan |
| Final | **0,9529** | Menunggu evaluasi |

*Catatan: hasil validasi lokal tidak dapat dibandingkan secara langsung dengan leaderboard dosen karena menggunakan data evaluasi berbeda. Nilai resmi diisi setelah diumumkan.*

## 6. Cara Menjalankan

### 6.1 Persiapan Environment

Disarankan menggunakan virtual environment Python.

```bash
python -m venv .venv
```

Aktifkan environment, kemudian instal dependensi:

```bash
pip install -r submission/requirements.txt
```

Pastikan PyTorch dengan dukungan CUDA sudah terpasang jika ingin melakukan training menggunakan GPU NVIDIA.

### 6.2 Training Model

Dataset harus dipersiapkan secara lokal dan tidak diunggah ke repository publik.

Contoh perintah eksperimen V4:

```bash
python train_v4.py --train ./my_train --val ./my_val --batch-size 1 --accumulation 16 --finetune-epochs 18
```

Checkpoint terbaik disimpan dalam direktori eksperimen V4.

### 6.3 Evaluasi Model Final

```bash
python evaluate.py --submission ./submission --data ./my_val --batch-size 2
```

Evaluator akan memuat arsitektur dari `submission/model.py` dan bobot terlatih dari `submission/model.pth`.

Model menggunakan `pretrained=False` pada proses evaluasi karena seluruh bobot diperoleh dari checkpoint submission dan tidak memerlukan pengunduhan pretrained weights.

## 7. Struktur Repository

```text
Pembelajaran_Mendalam/
├── submission/
│   ├── model.py
│   ├── model.pth
│   └── requirements.txt
├── train_baseline.py
├── train_v4.py
├── evaluate.py
├── README.md
└── .gitignore
```

Folder dataset, split validasi lokal, virtual environment, dan checkpoint eksperimen tidak disertakan dalam repository publik.

## 8. Kesimpulan

Eksperimen menunjukkan bahwa penggunaan ConvNeXt-Tiny dengan pretrained ImageNet-21K → 1K memberikan performa validasi lokal yang lebih baik daripada ResNet-18 baseline dan ConvNeXt-Tiny dari eksperimen sebelumnya.

Model akhir berhasil mencapai **Macro-F1 0,9529 dan akurasi 95,61%** pada 410 gambar validasi lokal.

Tantangan utama terdapat pada kemiripan karakteristik visual antarkelas, terutama `kertas_kardus` dan `styrofoam`. Hasil percobaan class weighting memperlihatkan adanya trade-off antara precision dan recall, sedangkan eksperimen V4 menghasilkan perbaikan klasifikasi yang lebih menyeluruh.

Model V4 dipilih untuk submission Final berdasarkan hasil evaluasi lokal terbaik.

---

**Mata Kuliah:** Deep Learning (IF25-40401)  
**Institusi:** Institut Teknologi Sumatera  
**Tahun Akademik:** 2026/2027