# Plaka Takip Sistemi

Şirket içindeki güvenlik bilgisayarında tamamen yerel çalışacak Windows masaüstü plaka takip uygulamasının ilk sürümüdür. Bu sürüm; kullanıcı oturumu, rol tabanlı yetkilendirme, canlı kamera önizlemesi ve plaka kayıt altyapısını sağlar. İnternet veya bulut servisi kullanmaz.

> **Önemli:** Bu aşamada RTSP/local video/webcam görüntüsü yalnızca Dashboard önizlemesi için kullanılır. OCR, YOLO, ONNX veya plaka tanıma modeli henüz yoktur.

## Özellikler

- bcrypt ile hashlenen şifreler ve SQLite kullanıcı verisi
- `ADMIN` ve `USER` rolleri
- Login → Dashboard akışı
- Giriş/çıkış kamera kartlarında OpenCV tabanlı canlı önizleme ve bağlantı durumu
- Kamera başına ayrı worker thread, sınırlı önizleme FPS'i ve otomatik yeniden bağlanma
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
- OpenCV (`opencv-python`)

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

## Kamera Testi

RTSP kamera ile test etmek için:

1. Uygulamaya `ADMIN` hesabıyla giriş yapın.
2. Sol menüden **Ayarlar** sayfasını açın.
3. Giriş ve/veya çıkış kamerasının URL alanına kameranın RTSP adresini girin. Örnek biçim: `rtsp://username:password@camera-ip:554/path`
4. Kamerayı **Aktif** olarak işaretleyip kaydedin.
5. Kamera ayarının çalışan akışa uygulanması için çıkış yapıp yeniden giriş yapın.
6. Dashboard'a dönerek doğru yön kartında canlı görüntüyü ve bağlantı durumunu kontrol edin.

Gerçek kamera olmadan aynı pipeline'ı yerel bir video ile test edebilirsiniz. Büyük video dosyasını repository'ye commit etmeden örneğin `test-data/car-video.mp4` konumuna kopyalayın ve Ayarlar ekranındaki URL alanına bu relative yolu girin. Absolute bir video yolu da kullanılabilir. Video bittiğinde worker kısa bir beklemeden sonra kaynağı yeniden açar.

Webcam testi için URL alanına cihaz indeksini (`0`, gerekirse `1`) yazabilirsiniz.

RTSP kullanıcı adı ve parolasını kaynak koda veya repository'ye eklemeyin. Mevcut sürüm `stream_url` değerini yerel SQLite veritabanında düz metin saklar; production öncesinde credential şifreleme eklenmelidir.

## Klasör yapısı

```text
CamerBound/
├── main.py
├── requirements.txt
├── README.md
├── app/
│   ├── auth.py
│   ├── camera.py
│   ├── camera_worker.py
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
    ├── test_camera_service.py
    ├── test_services.py
    └── test_ui_smoke.py
```

## Mimari

- `app/database.py`: SQLite bağlantısı, transaction yönetimi, şema ve başlangıç kamera kayıtları.
- `app/auth.py`: Parola hashleme, login, oturum modeli, kullanıcı oluşturma ve rol kontrolleri.
- `app/camera.py`: Kamera ayarları, worker/thread referansları ve capture yaşam döngüsünün servis sınırı.
- `app/camera_worker.py`: OpenCV kaynağını UI thread'i dışında okur, önizleme FPS'ini sınırlar ve kesintide yeniden bağlanır.
- `app/plate_service.py`: Kayıt ekleme, arama, son kayıtlar ve içerideki araç sorgusu.
- `ui/`: Servisleri kullanan PySide6 ekranları; doğrudan SQL çalıştırmaz.
- `app/config.py`: `settings.json` okur ve relative yolları uygulama köküne göre çözer.

## Sonraki geliştirme adımları

1. Varsayılan admin parolasını değiştirme ekranı eklemek.
2. `models/` altına lokal OCR/ONNX modeli ve ayrı tanıma servisi eklemek.
3. `app/plate_service.py` içindeki TODO noktasında kamera/plaka debounce uygulamak.
4. RTSP credential verisini şifreli saklamak.
5. PyInstaller yapılandırması ve temiz bir Windows makinede paket testi yapmak.

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
