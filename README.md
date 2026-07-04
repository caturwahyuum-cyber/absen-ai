# 🎓 PresensiMhs — Sistem Presensi Mahasiswa Berbasis Wajah AI

Sistem presensi mahasiswa berbasis kecerdasan buatan (AI) lokal menggunakan pemindaian wajah, deteksi liveness (keaktifan mata), serta validasi wilayah berbasis GPS (geofencing).

## 🚀 Fitur

| Fitur | Teknologi |
|---|---|
| **Deteksi Wajah** | OpenCV Haar Cascade |
| **Pengenalan Wajah** | LBPH Face Recognizer |
| **Liveness Check** | Deteksi mata (Eye Cascade) |
| **Validasi Lokasi** | GPS Haversine Geofencing |
| **Database** | SQLite (lokal, tanpa cloud) |
| **Backend** | Python Flask |
| **Frontend** | HTML + Vanilla JS |

---

## 📦 Instalasi

### 1. Clone / Download Proyek
```bash
cd absen-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Catatan:** Pastikan library `opencv-contrib-python` terinstall (bukan `opencv-python` standar),  
> karena modul recognizer `cv2.face.LBPHFaceRecognizer_create()` ada pada paket contrib.

### 3. Konfigurasi Lokasi Kampus
Edit berkas `app.py` pada bagian konfigurasi koordinat:
```python
CAMPUS_LAT = -6.7126590       # Latitude kampus Anda
CAMPUS_LON = 111.0138809      # Longitude kampus Anda
CAMPUS_RADIUS_M = 200         # Radius toleransi dalam meter
```

### 4. Jalankan Server
```bash
python app.py
```

Server akan aktif pada alamat: **http://localhost:5000**

---

## 📱 Cara Penggunaan

### A. Registrasi Mahasiswa (Admin)
1. Buka halaman admin: **http://localhost:5000/admin**
2. Isi data Nama, NIM, dan Kelas.
3. Klik tombol **"Ambil Foto Wajah"** minimal 5 kali dengan posisi sudut wajah berbeda (depan, kiri, kanan, atas, bawah).
4. Klik tombol **"Daftarkan Mahasiswa"**.

### B. Presensi Kehadiran
1. Buka halaman utama: **http://localhost:5000**
2. Pilih tipe presensi: **Absen Masuk** atau **Absen Pulang**.
3. Izinkan akses kamera dan lokasi (GPS) pada peramban Anda.
4. Posisikan wajah di dalam lingkaran panduan.
5. Pastikan mata terbuka (liveness check).
6. Klik tombol **"Verifikasi & Absen"**.

---

## 🔐 Mekanisme Keamanan

### 1. Face Recognition (LBPH)
- Menggunakan algoritma **Local Binary Patterns Histograms** untuk mencocokkan wajah.
- Model akan otomatis dilatih ulang setiap kali ada mahasiswa baru yang terdaftar.
- Ambang batas kecocokan (confidence score) diatur dengan threshold < 110.

### 2. Liveness Detection
- Mendeteksi mata mahasiswa menggunakan Haar Eye Cascade.
- Berguna meminimalkan kecurangan absen menggunakan cetakan foto atau layar gadget lain.
- Minimal terdeteksi 1 mata untuk lolos validasi keaktifan.

### 3. GPS Geofencing
- Menghitung jarak presensi mahasiswa dari titik koordinat kampus menggunakan rumus **Haversine**.
- Mahasiswa harus berada di dalam batas radius kampus yang ditentukan (misalnya 200 meter).
- Jika di luar radius, presensi masih bisa dilakukan namun status lokasi akan ditandai **Luar Kampus**.

---

## 🗄️ Struktur Database

### Tabel `mahasiswa`
| Kolom | Tipe | Keterangan |
|---|---|---|
| id | INTEGER | Primary key (auto-increment) |
| nama | TEXT | Nama lengkap mahasiswa |
| nim | TEXT | Nomor Induk Mahasiswa (unik) |
| kelas | TEXT | Kelas/jurusan mahasiswa |
| label_id | INTEGER | ID representasi model LBPH |
| terdaftar | TEXT | Timestamp tanggal terdaftar |

### Tabel `absensi`
| Kolom | Tipe | Keterangan |
|---|---|---|
| id | INTEGER | Primary key (auto-increment) |
| mahasiswa_id | INTEGER | Foreign key mengarah ke tabel mahasiswa |
| tipe | TEXT | Tipe absen ('masuk' atau 'pulang') |
| waktu | TEXT | Timestamp pelaksanaan presensi |
| latitude | REAL | Koordinat lintang GPS |
| longitude | REAL | Koordinat bujur GPS |
| jarak_m | REAL | Jarak riil dari kampus (meter) |
| lokasi_valid | INTEGER | 1 = Di dalam kampus, 0 = Luar kampus |
| wajah_conf | REAL | Persentase kecocokan wajah (0-100) |

---

## 📁 Struktur File

```
absen-ai/
├── app.py              # Backend Flask
├── index.html          # Halaman presensi utama
├── admin.html          # Halaman admin & registrasi
├── requirements.txt    # Daftar dependensi Python
├── README.md           # Berkas panduan ini
├── database/
│   ├── absen.db        # Database SQLite (dibuat otomatis)
│   └── faces/          # Kumpulan foto sampel wajah mahasiswa
│       └── A11.2022.12345/
│           ├── 000.jpg
│           └── ...
└── models/
    └── face_model.yml  # File model pengenalan wajah LBPH
```
