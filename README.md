# Plaka Takip Sistemi

Şirket içindeki güvenlik bilgisayarında tamamen yerel çalışacak Windows masaüstü plaka takip uygulamasının ilk sürümüdür. Bu sürüm; kullanıcı oturumu, rol tabanlı yetkilendirme, kamera yapılandırması ve plaka kayıt altyapısını sağlar. İnternet veya bulut servisi kullanmaz.

> **Önemli:** Bu aşamada gerçek RTSP bağlantısı, OpenCV görüntü işleme, OCR/ONNX veya plaka tanıma modeli yoktur. Kamera alanları sonraki entegrasyon için hazırlanmıştır.

## Özellikler

- bcrypt ile hashlenen şifreler ve SQLite kullanıcı verisi
- `ADMIN` ve `USER` rolleri
- Login → Dashboard akışı
- Giriş/çıkış kamera kartları ve son hareketler
- Plaka ve yön bazlı kayıt arama
- Son hareketi `ENTRY` olan araçlardan hesaplanan “İçerideki Araçlar” ekranı
- ADMIN için kullanıcı oluşturma ve iki kamerayı yapılandırma
- UI dışında, servis seviyesinde yetki kontrolleri
- Çalışma dizininden bağımsız config ve veritabanı yolu

## Gereksinimler

- Windows 10/11
- Python 3.12
- PySide6
- bcrypt

OpenCV bu sürümde kullanılmadığı için minimum dependency listesine henüz eklenmemiştir.

## Kurulum

PowerShell ile proje klasöründe:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Çalıştırma

```powershell
python main.py
```

İlk çalıştırmada `data/plate_tracker.db` otomatik oluşturulur; tablolar, indexler, iki varsayılan kamera ve kullanıcı yoksa geliştirme admin hesabı eklenir.

## Varsayılan geliştirme hesabı

- Kullanıcı adı: `admin`
- Şifre: `admin123`

**Bu hesap yalnızca ilk geliştirme içindir. Production kullanımı öncesinde varsayılan parola mutlaka değiştirilmelidir.**

## Testler

Servis ve arayüz smoke testlerini çalıştırmak için:

```powershell
python -m unittest discover -v
```

UI testi Qt'yi `offscreen` platformunda açar, login formundaki butona tıklar ve ADMIN dashboard’unun oluştuğunu kontrol eder.

## Klasör yapısı

```text
CamerBound/
├── main.py
├── requirements.txt
├── README.md
├── app/
│   ├── auth.py
│   ├── camera.py
│   ├── config.py
│   ├── database.py
│   └── plate_service.py
├── ui/
│   ├── admin_widget.py
│   ├── dashboard_window.py
│   ├── login_window.py
│   ├── records_widget.py
│   └── styles.py
├── config/
│   └── settings.json
├── data/
│   └── plate_tracker.db       # İlk çalıştırmada oluşur
├── models/                    # Gelecekteki yerel OCR/ONNX modeli
└── tests/
    ├── test_services.py
    └── test_ui_smoke.py
```

## Mimari

- `app/database.py`: SQLite bağlantısı, transaction yönetimi, şema ve başlangıç kamera kayıtları.
- `app/auth.py`: Parola hashleme, login, oturum modeli, kullanıcı oluşturma ve rol kontrolleri.
- `app/camera.py`: Kamera ayarları ve gelecekteki `VideoCapture` yaşam döngüsünün servis sınırı.
- `app/plate_service.py`: Kayıt ekleme, arama, son kayıtlar ve içerideki araç sorgusu.
- `ui/`: Servisleri kullanan PySide6 ekranları; doğrudan SQL çalıştırmaz.
- `app/config.py`: `settings.json` okur ve relative yolları uygulama köküne göre çözer.

## Sonraki geliştirme adımları

1. Varsayılan admin parolasını değiştirme ekranı eklemek.
2. `app/camera.py` içindeki `start_camera` / `stop_camera` sınırına OpenCV `VideoCapture` ve worker thread eklemek.
3. Kamera karelerini Qt sinyalleriyle dashboard kartlarına taşımak.
4. `models/` altına lokal OCR/ONNX modeli ve ayrı tanıma servisi eklemek.
5. `app/plate_service.py` içindeki TODO noktasında kamera/plaka debounce uygulamak.
6. RTSP credential verisini şifreli saklamak.
7. PyInstaller yapılandırması ve temiz bir Windows makinede paket testi yapmak.

## Ayarlar

`config/settings.json`:

```json
{
  "database_path": "data/plate_tracker.db",
  "plate_detection": {
    "duplicate_cooldown_seconds": 10
  }
}
```

Relative veritabanı yolu mevcut terminal klasörüne değil, uygulamanın bulunduğu kök klasöre göre çözülür.
