# Klasifikasi Berita ClickBait Menggunakan Mamba (Selective State Space Model)

Repositori ini memuat kode program penelitian skripsi yang berjudul "Klasifikasi Berita ClickBait Menggunakan Mamba (Selective State Space Model)". Penelitian ini membangun sebuah model untuk mengklasifikasikan judul berita berbahasa Indonesia ke dalam dua kategori, yaitu *clickbait* dan non-*clickbait*. Model dikembangkan menggunakan arsitektur Mamba (*Selective State Space Model*) dengan tokenisasi IndoBERTweet, dan dilengkapi antarmuka web yang memungkinkan pengujian secara langsung melalui masukan tautan (URL) berita maupun teks judul yang diketik secara manual.

Penulis: Ridhwan Ardiyansyah  535220114
Program Studi Teknik Informatika, Fakultas Teknologi Informasi, Universitas Tarumanagara

## Struktur Direktori

| Berkas | Keterangan |
| --- | --- |
| `Program_Skripsi_ClickBait_Mamba.ipynb` | Berkas notebook yang memuat keseluruhan kode program: pra-pemrosesan data, definisi arsitektur model, pelatihan dengan *K-Fold Cross Validation*, evaluasi, dan antarmuka web. |
| `app.py` | Kode aplikasi web mandiri yang dapat dijalankan di luar lingkungan notebook. |
| `requirements.txt` | Daftar pustaka (*library*) yang diperlukan beserta keterangan pemasangan. |
| `mamba_model_best.pth` | Bobot model hasil pelatihan yang digunakan oleh aplikasi. |
| `README.md` | Dokumentasi dan petunjuk penggunaan program. |

## Kebutuhan Sistem

Pelatihan model maupun pengoperasian aplikasi memerlukan perangkat yang dilengkapi kartu grafis (GPU) NVIDIA dengan dukungan CUDA. Persyaratan ini bersifat wajib karena pustaka `mamba-ssm` dan `causal-conv1d` dikompilasi pada lingkungan CUDA dan tidak dapat dipasang pada perangkat tanpa GPU. Apabila perangkat pengguna tidak memiliki GPU, program dapat dijalankan melalui layanan Google Colaboratory dengan akselerator GPU yang tersedia secara gratis. Versi Python yang digunakan adalah 3.10 atau yang lebih baru.

## Petunjuk Pemasangan

Pemasangan seluruh pustaka yang diperlukan dilakukan dengan menjalankan perintah berikut pada lingkungan yang telah dilengkapi GPU:

```bash
pip install -q transformers pandas scikit-learn matplotlib seaborn
pip install -q newspaper3k lxml_html_clean
pip install -q ninja packaging
pip install -q --no-build-isolation causal-conv1d>=1.2.0
pip install -q --no-build-isolation mamba-ssm
pip install -q flask pyngrok
```

Pustaka `mamba-ssm` harus dipasang menggunakan opsi `--no-build-isolation` dan membutuhkan GPU pada saat proses kompilasi.

## Dataset

Penelitian ini menggunakan dataset CLICK-ID, yaitu kumpulan judul berita berbahasa Indonesia yang telah diberi label *clickbait* dan non-*clickbait*. Program membaca data dari berkas `dataset_final.csv`. Pada lingkungan Google Colaboratory, berkas tersebut ditempatkan pada direktori `/content/drive/MyDrive/Skripsi/dataset_final.csv`. Lokasi berkas dapat disesuaikan dengan mengubah variabel `file_path` pada notebook.

Tautan berkas dataset: https://drive.google.com/file/d/1qLoKZ5LYEGlah3ZFBXdXLO74yfVgRfad/view?usp=sharing

## Model Terlatih

Bobot model hasil pelatihan disimpan dalam berkas `mamba_model_best.pth`. Berkas ini dimuat oleh aplikasi (`app.py` atau sel antarmuka pada notebook) untuk melakukan proses klasifikasi tanpa perlu melatih ulang model. Apabila ukuran berkas melebihi 100 MB, pengunggahan ke repositori dilakukan menggunakan Git LFS (lihat bagian Pengelolaan Berkas Model). Sebagai alternatif, berkas dapat ditempatkan pada layanan penyimpanan eksternal dan tautannya dicantumkan di bawah ini.

Tautan berkas model : https://drive.google.com/file/d/1QqiA0L9VG8BylhOSTllKwosB0W4BdRYa/view?usp=sharing

## Petunjuk Penggunaan

### a. Melalui Google Colaboratory

Cara ini disarankan bagi pengguna yang perangkatnya tidak dilengkapi GPU.

1. Buka berkas notebook melalui Google Colaboratory.
2. Aktifkan akselerator GPU melalui menu Runtime, Change runtime type, kemudian pilih GPU.
3. Jalankan setiap sel secara berurutan, mulai dari pemasangan pustaka, pemuatan dataset dan tokenizer, pendefinisian arsitektur, pelatihan model, hingga sel antarmuka web. Tahap pelatihan dapat dilewati apabila berkas model `mamba_model_best.pth` telah tersedia.
4. Pada sel antarmuka, masukkan token autentikasi ngrok. Setelah sel dijalankan, sistem akan menampilkan sebuah alamat (URL) publik yang dapat dibuka melalui peramban.

### b. Melalui berkas app.py (perangkat ber-GPU)

1. Pasang seluruh pustaka sebagaimana tercantum pada bagian Petunjuk Pemasangan.
2. Tempatkan berkas `mamba_model_best.pth` pada direktori yang sama dengan `app.py`.
3. Jalankan perintah berikut:
   ```bash
   python app.py
   ```
4. Buka alamat `http://localhost:5000` melalui peramban.

Aplikasi juga dapat dipublikasikan ke jaringan internet menggunakan ngrok dengan menetapkan variabel lingkungan `NGROK_AUTHTOKEN` sebelum menjalankan program:

```bash
# Linux atau macOS
export NGROK_AUTHTOKEN="token_ngrok_anda"
# Windows
set NGROK_AUTHTOKEN=token_ngrok_anda

python app.py
```

Variabel lingkungan yang didukung adalah `MODEL_PATH` (nilai bawaan `mamba_model_best.pth`) untuk menentukan lokasi berkas model, dan `NGROK_AUTHTOKEN` untuk mengaktifkan publikasi melalui ngrok.

## Pengoperasian Aplikasi

1. Buka alamat aplikasi pada peramban.
2. Pilih salah satu mode masukan:
   - **Tautan (URL) berita**, yaitu dengan menempelkan alamat berita. Sistem akan mengambil judul dan isi berita secara otomatis melalui proses *web scraping*.
   - **Teks manual**, yaitu dengan mengetikkan judul berita secara langsung.
3. Tekan tombol analisis. Sistem akan menampilkan hasil klasifikasi, yaitu *clickbait* atau non-*clickbait*, beserta nilai probabilitasnya.

## Pengelolaan Berkas Model (Git LFS)

Berkas model berekstensi `.pth` dapat melampaui batas pengunggahan biasa pada GitHub, yaitu 25 MB melalui peramban dan 100 MB per berkas. Untuk berkas berukuran besar digunakan Git LFS dengan langkah berikut:

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add mamba_model_best.pth
git commit -m "Menambahkan berkas model terlatih"
git push
```

Kuota gratis Git LFS pada GitHub adalah 1 GB penyimpanan dan 1 GB lalu lintas data per bulan. Apabila kuota tidak mencukupi, berkas model dapat ditempatkan pada layanan penyimpanan eksternal seperti Google Drive atau Hugging Face, dengan tautannya dicantumkan pada bagian Model Terlatih.

## Keterangan Keamanan

Token autentikasi ngrok maupun kredensial lainnya tidak dicantumkan di dalam kode pada repositori ini. Pengguna memasukkan token tersebut secara mandiri pada saat menjalankan program.
