import os
import cv2
import json
import base64
import sqlite3
import numpy as np
from datetime import datetime, date
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import math

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'absenai-secret-2024-ganti-ini')
CORS(app)

# Password admin (ubah via env var ADMIN_PASSWORD di Render)
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')

# Konfigurasi Path dan Direktori
BASE_DIR = os.path.dirname(__file__)
if os.environ.get("RENDER"):
    # Di Render cloud: SELALU pakai /tmp (bisa ditulis, gratis)
    DATA_DIR = "/tmp/absen-data"
else:
    # Di lokal: pakai folder database/ di dalam project
    DATA_DIR = os.path.join(BASE_DIR, "database")
DB_PATH    = os.path.join(DATA_DIR, "absen.db")
FACE_DIR   = os.path.join(DATA_DIR, "faces")
MODEL_PATH = os.path.join(DATA_DIR, "face_model.yml")

# Konfigurasi Koordinat Kampus (ganti sesuai lokasi kampus Anda)
CAMPUS_LAT = -6.982605018812796
CAMPUS_LON = 110.40861501968638
CAMPUS_RADIUS_M = 200      # Radius toleransi presensi (meter)

# Pastikan folder data dan model sudah ada
os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def load_cascade(filename):
    """Memuat CascadeClassifier secara aman. Jika file bawaan opencv kosong atau tidak ditemukan,
    maka akan diunduh dari repository GitHub resmi OpenCV dan disimpan di DATA_DIR."""
    # Coba load dari OpenCV default
    default_path = os.path.join(cv2.data.haarcascades, filename)
    cascade = cv2.CascadeClassifier(default_path)
    
    if not cascade.empty():
        print(f"[INFO] Berhasil memuat cascade default: {filename}")
        return cascade

    # Jika kosong, coba load dari local DATA_DIR
    local_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(local_path):
        cascade = cv2.CascadeClassifier(local_path)
        if not cascade.empty():
            print(f"[INFO] Berhasil memuat cascade lokal: {local_path}")
            return cascade

    # Jika belum ada atau masih kosong, unduh dari internet
    import urllib.request
    url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{filename}"
    print(f"[WARN] Cascade default kosong. Mengunduh {filename} dari {url}...")
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(url, local_path)
        cascade = cv2.CascadeClassifier(local_path)
        if not cascade.empty():
            print(f"[INFO] Berhasil mengunduh dan memuat cascade: {local_path}")
            return cascade
    except Exception as e:
        print(f"[ERROR] Gagal mengunduh atau memuat cascade {filename}: {e}")

    return cascade

# Load XML cascade classifier untuk wajah dan mata
face_cascade = load_cascade('haarcascade_frontalface_default.xml')
eye_cascade  = load_cascade('haarcascade_eye.xml')

# Inisialisasi LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_trained = False


# Fungsi-fungsi database
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Membuat tabel database mahasiswa dan absensi jika belum ada."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS mahasiswa (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            nama      TEXT    NOT NULL,
            nim       TEXT    UNIQUE NOT NULL,
            kelas     TEXT,
            label_id  INTEGER UNIQUE,
            terdaftar TEXT    DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS absensi (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            mahasiswa_id INTEGER REFERENCES mahasiswa(id),
            tipe         TEXT CHECK(tipe IN ('masuk','pulang')),
            waktu        TEXT DEFAULT (datetime('now','localtime')),
            latitude     REAL,
            longitude    REAL,
            jarak_m      REAL,
            lokasi_valid INTEGER DEFAULT 0,
            wajah_conf   REAL,
            foto         TEXT,
            catatan      TEXT
        );
    """)
    conn.commit()
    conn.close()
    
init_db()

# Helper fungsi untuk pengolahan gambar dan lokasi
def decode_image(b64_string):
    """Mengubah format gambar base64 dari client menjadi numpy array BGR."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    img_pil = Image.open(BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img_pil)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def haversine(lat1, lon1, lat2, lon2):
    """Menghitung jarak koordinat GPS menggunakan formula Haversine (hasil dalam meter)."""
    R = 6371000  # Radius bumi dalam meter
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def preprocess_gray(img_bgr):
    """Konversi ke grayscale + CLAHE untuk memperbaiki kontras di kondisi gelap."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def detect_face(img_bgr):
    """Mendeteksi wajah dan mengembalikan gambar wajah grayscale ukuran 200x200."""
    gray = preprocess_gray(img_bgr)

    # Percobaan 1: parameter normal
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

    # Percobaan 2 (fallback): parameter lebih longgar jika tidak ada wajah ditemukan
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

    if len(faces) == 0:
        return None, None

    # Ambil wajah dengan ukuran area terbesar
    x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
    face_gray = gray[y:y+h, x:x+w]
    face_gray = cv2.resize(face_gray, (200, 200))
    return face_gray, (x, y, w, h)

def check_liveness(img_bgr, rect):
    """Mendeteksi keberadaan mata untuk memverifikasi keaktifan wajah (liveness check)."""
    if rect is None:
        return False, 0
    x, y, w, h = rect
    face_roi = img_bgr[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Coba deteksi mata normal
    eyes = eye_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=2, minSize=(15, 15))

    # Fallback: pakai CLAHE + parameter lebih longgar
    if len(eyes) == 0:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_roi_eq = clahe.apply(gray_roi)
        eyes = eye_cascade.detectMultiScale(gray_roi_eq, scaleFactor=1.05, minNeighbors=1, minSize=(10, 10))

    eye_count = len(eyes)
    # Minimal terdeteksi 1 mata untuk lolos
    is_live = eye_count >= 1
    return is_live, eye_count

def retrain_model():
    """Melatih ulang model LBPH wajah berdasarkan foto-foto yang terdaftar."""
    global model_trained
    conn = get_db()
    rows = conn.execute("SELECT id, label_id, nim FROM mahasiswa WHERE label_id IS NOT NULL").fetchall()
    conn.close()

    faces, labels = [], []
    for row in rows:
        person_dir = os.path.join(FACE_DIR, row['nim'])
        if not os.path.isdir(person_dir):
            continue
        for fname in os.listdir(person_dir):
            if fname.endswith('.jpg'):
                path = os.path.join(person_dir, fname)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (200, 200))
                    faces.append(img)
                    labels.append(row['label_id'])

    if len(faces) >= 2:
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL_PATH)
        model_trained = True
        return True, len(faces)
    return False, len(faces)

def load_model():
    """Membaca model wajah yang sudah dilatih jika ada, atau latih ulang jika ada foto."""
    global model_trained
    if os.path.exists(MODEL_PATH):
        try:
            recognizer.read(MODEL_PATH)
            model_trained = True
            print(f"[INFO] Model wajah berhasil dimuat dari {MODEL_PATH}")
        except Exception as e:
            print(f"[WARN] Gagal baca model: {e}")
            model_trained = False

    # Jika model tidak ada tapi folder foto ada → retrain otomatis
    if not model_trained and os.path.isdir(FACE_DIR):
        foto_ada = any(
            f.endswith('.jpg')
            for nim_dir in os.listdir(FACE_DIR)
            for f in os.listdir(os.path.join(FACE_DIR, nim_dir))
            if os.path.isdir(os.path.join(FACE_DIR, nim_dir))
        )
        if foto_ada:
            print("[INFO] Model tidak ada tapi foto tersedia, melatih ulang...")
            ok, count = retrain_model()
            if ok:
                print(f"[INFO] Model berhasil dilatih ulang dari {count} foto")

load_model()


# Routes untuk render halaman utama
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/daftar')
def daftar():
    return send_from_directory(BASE_DIR, 'daftar.html')

@app.route('/admin')
def admin():
    if not session.get('admin_logged_in'):
        return redirect('/admin/login')
    return send_from_directory(BASE_DIR, 'admin.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = ''
    if request.method == 'POST':
        pwd = request.form.get('password', '')
        if pwd == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect('/admin')
        error = 'Password salah!'
    return f'''
    <!DOCTYPE html><html lang="id"><head>
    <meta charset="UTF-8"><title>Login Admin</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap" rel="stylesheet">
    <style>
      *{{margin:0;padding:0;box-sizing:border-box}}
      body{{font-family:'Space Grotesk',sans-serif;background:#080c14;color:#e2e8f0;
            min-height:100vh;display:flex;align-items:center;justify-content:center}}
      .card{{background:#111827;border:1px solid #1a2540;border-radius:14px;padding:2.5rem;width:340px}}
      h2{{font-size:1.3rem;margin-bottom:.3rem;color:#00d4ff}}
      p{{font-size:.85rem;color:#64748b;margin-bottom:1.5rem}}
      input{{width:100%;background:#0d1520;border:1px solid #1a2540;color:#e2e8f0;
             padding:.7rem 1rem;border-radius:8px;font-size:.9rem;outline:none;margin-bottom:1rem}}
      input:focus{{border-color:#00d4ff}}
      button{{width:100%;background:linear-gradient(135deg,#00d4ff,#0099cc);color:#000;
              border:none;padding:.8rem;border-radius:8px;font-weight:600;font-size:.9rem;cursor:pointer}}
      .error{{color:#ef4444;font-size:.83rem;margin-bottom:.8rem}}
    </style></head><body>
    <div class="card">
      <h2>🔐 Login Admin</h2>
      <p>Masukkan password untuk akses panel admin</p>
      {"<div class='error'>" + error + "</div>" if error else ""}
      <form method="POST">
        <input type="password" name="password" placeholder="Password admin" autofocus required>
        <button type="submit">Masuk</button>
      </form>
    </div></body></html>
    '''

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect('/admin/login')

# API: Mendaftarkan mahasiswa baru beserta foto sampel wajah
@app.route('/api/mahasiswa/daftar', methods=['POST'])
def daftar_mahasiswa():
    data  = request.get_json()
    nama  = data.get('nama', '').strip()
    nim   = data.get('nim', '').strip()
    kelas = data.get('kelas', '').strip()
    photos = data.get('photos', [])

    if not nama or not nim:
        return jsonify({'ok': False, 'pesan': 'Nama dan NIM wajib diisi'}), 400
    if len(photos) < 3:
        return jsonify({'ok': False, 'pesan': 'Minimal 3 foto wajah diperlukan'}), 400

    conn = get_db()
    try:
        # Cari label_id baru yang unik
        last = conn.execute("SELECT MAX(label_id) as m FROM mahasiswa").fetchone()
        label_id = (last['m'] or 0) + 1

        conn.execute(
            "INSERT INTO mahasiswa (nama, nim, kelas, label_id) VALUES (?,?,?,?)",
            (nama, nim, kelas, label_id)
        )
        conn.commit()

        # Buat direktori penyimpanan foto mahasiswa
        person_dir = os.path.join(FACE_DIR, nim)
        os.makedirs(person_dir, exist_ok=True)

        saved = 0
        for i, b64 in enumerate(photos):
            try:
                img = decode_image(b64)
                face, _ = detect_face(img)
                if face is not None:
                    cv2.imwrite(os.path.join(person_dir, f'{i:03d}.jpg'), face)
                    saved += 1
                else:
                    print(f"[WARN] Wajah tidak terdeteksi pada foto sampel ke-{i}")
            except Exception as e:
                print(f"[ERROR] Gagal memproses foto sampel ke-{i}: {e}")

        # Validasi jika jumlah foto yang tersimpan kurang dari batas minimal
        if saved < 2:
            conn.execute("DELETE FROM mahasiswa WHERE nim=?", (nim,))
            conn.commit()
            conn.close()
            return jsonify({'ok': False, 'pesan': f'Hanya {saved} foto wajah valid. Minimal 2 dibutuhkan.'}), 400

        # Latih ulang model wajah
        retrain_model()
        conn.close()
        return jsonify({'ok': True, 'pesan': f'Mahasiswa {nama} berhasil didaftarkan dengan {saved} foto wajah.'})

    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'ok': False, 'pesan': f'NIM {nim} sudah terdaftar'}), 409
    except Exception as e:
        conn.close()
        return jsonify({'ok': False, 'pesan': str(e)}), 500

# API: Proses presensi kehadiran (masuk / pulang)
@app.route('/api/absen', methods=['POST'])
def absen():
    data = request.get_json()
    b64  = data.get('foto')
    lat  = data.get('latitude')
    lon  = data.get('longitude')
    tipe = data.get('tipe', 'masuk')

    if not b64:
        return jsonify({'ok': False, 'pesan': 'Foto tidak ditemukan'}), 400

    # 1. Dekode dan deteksi wajah
    try:
        img = decode_image(b64)
    except Exception as e:
        return jsonify({'ok': False, 'pesan': f'Gagal decode foto: {e}'}), 400

    face, rect = detect_face(img)
    if face is None:
        return jsonify({'ok': False, 'pesan': 'Wajah tidak terdeteksi. Posisikan wajah dengan benar.'}), 400

    # 2. Cek Liveness (Deteksi Mata)
    is_live, eye_count = check_liveness(img, rect)
    if not is_live:
        return jsonify({'ok': False, 'pesan': 'Gagal verifikasi liveness. Pastikan mata terbuka.'}), 400

    # 3. Pengenalan Wajah via LBPH
    if not model_trained:
        return jsonify({'ok': False, 'pesan': 'Model belum dilatih. Daftarkan mahasiswa terlebih dahulu.'}), 503

    label_id, confidence = recognizer.predict(face)
    # Konversi jarak/confidence score LBPH ke skala 0-100%
    confidence_score = max(0, 100 - confidence)

    if confidence > 110:
        return jsonify({
            'ok': False,
            'pesan': f'Wajah tidak dikenali (skor: {confidence_score:.1f}%). Pastikan NIM Anda sudah terdaftar.',
            'confidence': confidence_score
        }), 401

    # 4. Cari data mahasiswa di DB
    conn = get_db()
    mahasiswa = conn.execute(
        "SELECT * FROM mahasiswa WHERE label_id=?", (label_id,)
    ).fetchone()

    if not mahasiswa:
        conn.close()
        return jsonify({'ok': False, 'pesan': 'Data mahasiswa tidak ditemukan'}), 404

    # 5. Cek Validasi Lokasi (GPS Geofencing)
    jarak = None
    lokasi_valid = False
    if lat is not None and lon is not None:
        jarak = haversine(lat, lon, CAMPUS_LAT, CAMPUS_LON)
        lokasi_valid = jarak <= CAMPUS_RADIUS_M
    
    # 6. Cek apakah sudah absen dengan tipe yang sama hari ini
    today_str = date.today().isoformat()
    existing = conn.execute("""
        SELECT id FROM absensi
        WHERE mahasiswa_id=? AND tipe=? AND DATE(waktu)=?
    """, (mahasiswa['id'], tipe, today_str)).fetchone()

    if existing:
        conn.close()
        label = 'masuk' if tipe == 'masuk' else 'pulang'
        return jsonify({'ok': False, 'pesan': f'Anda sudah absen {label} hari ini.'}), 409

    # 7. Simpan log absensi ke database
    conn.execute("""
        INSERT INTO absensi (mahasiswa_id, tipe, latitude, longitude, jarak_m, lokasi_valid, wajah_conf, foto, catatan)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        mahasiswa['id'], tipe,
        lat, lon, jarak, int(lokasi_valid),
        confidence_score,
        b64[:200], # Simpan sedikit string b64 sebagai representasi thumbnail
        f'Mata terdeteksi: {eye_count}'
    ))
    conn.commit()

    waktu_sekarang = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    conn.close()

    return jsonify({
        'ok': True,
        'pesan': f'Absen {tipe} berhasil!',
        'mahasiswa': mahasiswa['nama'],
        'nim': mahasiswa['nim'],
        'kelas': mahasiswa['kelas'],
        'tipe': tipe,
        'waktu': waktu_sekarang,
        'confidence': round(confidence_score, 1),
        'lokasi': {
            'valid': lokasi_valid,
            'jarak_m': round(jarak, 1) if jarak else None,
            'lat': lat,
            'lon': lon
        }
    })

# API: Mengambil rekap presensi berdasarkan tanggal
@app.route('/api/rekap', methods=['GET'])
def rekap():
    tanggal = request.args.get('tanggal', date.today().isoformat())
    conn = get_db()
    rows = conn.execute("""
        SELECT m.nama, m.nim, m.kelas,
               a.tipe, a.waktu, a.lokasi_valid, a.jarak_m, a.wajah_conf
        FROM absensi a
        JOIN mahasiswa m ON m.id = a.mahasiswa_id
        WHERE DATE(a.waktu) = ?
        ORDER BY a.waktu DESC
    """, (tanggal,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# API: Mengambil semua data mahasiswa terdaftar
@app.route('/api/mahasiswa', methods=['GET'])
def list_mahasiswa():
    conn = get_db()
    rows = conn.execute("SELECT id, nama, nim, kelas, terdaftar FROM mahasiswa ORDER BY nama").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# API: Mengambil statistik presensi hari ini
@app.route('/api/stats', methods=['GET'])
def stats():
    conn  = get_db()
    today = date.today().isoformat()
    total_mahasiswa = conn.execute("SELECT COUNT(*) FROM mahasiswa").fetchone()[0]
    hadir_hari_ini  = conn.execute(
        "SELECT COUNT(DISTINCT mahasiswa_id) FROM absensi WHERE tipe='masuk' AND DATE(waktu)=?", (today,)
    ).fetchone()[0]
    total_absen_bln = conn.execute(
        "SELECT COUNT(*) FROM absensi WHERE strftime('%Y-%m', waktu)=strftime('%Y-%m','now')"
    ).fetchone()[0]
    conn.close()
    return jsonify({
        'total_mahasiswa': total_mahasiswa,
        'hadir_hari_ini': hadir_hari_ini,
        'total_absen_bln': total_absen_bln,
        'campus': {
            'lat': CAMPUS_LAT,
            'lon': CAMPUS_LON,
            'radius_m': CAMPUS_RADIUS_M
        }
    })

# API: Menghapus data mahasiswa berdasarkan ID
@app.route('/api/mahasiswa/<int:mid>', methods=['DELETE'])
def hapus_mahasiswa(mid):
    conn = get_db()
    row = conn.execute("SELECT nim FROM mahasiswa WHERE id=?", (mid,)).fetchone()
    if row:
        import shutil
        person_dir = os.path.join(FACE_DIR, row['nim'])
        if os.path.isdir(person_dir):
            shutil.rmtree(person_dir)
        conn.execute("DELETE FROM absensi WHERE mahasiswa_id=?", (mid,))
        conn.execute("DELETE FROM mahasiswa WHERE id=?", (mid,))
        conn.commit()
        retrain_model()
    conn.close()
    return jsonify({'ok': True})

if __name__ == '__main__':
    print("[SERVER] Sistem Presensi Mahasiswa AI berjalan di http://localhost:5000")
    print(f"[INFO] Koordinat Kampus: {CAMPUS_LAT}, {CAMPUS_LON} (radius {CAMPUS_RADIUS_M}m)")
    app.run(debug=True, host='0.0.0.0', port=5000)
