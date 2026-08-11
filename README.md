# Plaka Takip Sistemi

Şirket içindeki güvenlik bilgisayarında tamamen yerel çalışacak Windows masaüstü plaka takip uygulamasının ilk sürümüdür. Bu sürüm; kullanıcı oturumu, rol tabanlı yetkilendirme, canlı kamera önizlemesi ve plaka kayıt altyapısını sağlar. İnternet veya bulut servisi kullanmaz.

> **Önemli:** Bu sürüm sabit giriş/çıkış kameraları için ROI tabanlı lokal OCR MVP'sidir. YOLO veya başka bir object detection modeli kullanılmaz.

## Özellikler

- bcrypt ile hashlenen şifreler ve SQLite kullanıcı verisi
- `ADMIN` ve `USER` rolleri
- Login → Dashboard akışı
- Giriş/çıkış kamera kartlarında OpenCV tabanlı canlı önizleme ve bağlantı durumu
- Kamera başına ayrı worker thread, sınırlı önizleme FPS'i ve otomatik yeniden bağlanma
- PaddleOCR ile ONNX Runtime veya native Paddle CPU üzerinde tamamen lokal plaka tanıma
- Kamera yönüne özel ROI, multi-frame confirmation ve duplicate cooldown
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
- OpenCV (`opencv-contrib-python`, PaddleOCR ile ortak tek `cv2` dağıtımı)
- PaddleOCR 3.7.0
- ONNX Runtime 1.27.0 (CPU)
- Native fallback için PaddlePaddle 3.x (CPU)

## Kurulum

PowerShell ile proje klasöründe:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Native Paddle fallback kullanılacaksa ayrıca `python -m pip install -r requirements-paddle.txt` çalıştırılır.

## Çalıştırma

Geliştirme ortamında Windows:
`run.bat` dosyasına çift tıklayın.

Manuel alternatif:

```powershell
.\.venv\Scripts\python.exe main.py
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

## Plaka Tanıma

Pipeline şu sırayla çalışır:

```text
CameraService frame
→ ENTRY/EXIT ROI crop
→ original + contrast preprocessing
→ lokal PaddleOCR / ONNX Runtime
→ normalization ve Türk plaka doğrulama
→ multi-frame confirmation
→ duplicate cooldown
→ PlateService
→ SQLite ve Dashboard güncellemesi
```

OCR inference ayrı bir `QThread` üzerinde çalışır. Kamera başına yalnızca en güncel frame tutulur; OCR yavaşladığında biriken, sınırsız bir frame kuyruğu oluşmaz. Model bulunmazsa login, Dashboard ve kamera preview çalışmaya devam eder; kartta `OCR: Kullanılamıyor` gösterilir.

### OCR model hazırlama altyapısı

Repository model binary dosyalarını içermez. Development hazırlık scripti [PaddleOCR resmî model listesindeki](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html) `PP-OCRv5_mobile_det` ve `en_PP-OCRv5_mobile_rec` modellerini Paddle'ın resmî model artefact sunucusundan indirir ve `build/ocr-model-downloads/` altında cache'ler. Modeller native Paddle olarak doğrudan kurulabilir veya ayrıca ONNX'e dönüştürülebilir. Yalnızca hazırlama/indirme adımı internet bağlantısı gerektirir.

Normal uygulama çalışması hiçbir zaman model indirmez ve internet gerektirmez. Model yoksa Dashboard'daki `OCR: Kullanılamıyor` davranışı korunur.

Hazırlık seçenekleri:

```powershell
python -m pip install -r requirements-paddle.txt
python scripts/prepare_default_ocr_models.py --backend paddle
python scripts/verify_ocr_models.py
```

Windows geliştirme ortamında önerilen yol `--backend paddle` seçeneğidir; bu yol Paddle2ONNX çalıştırmaz. `--backend auto` Windows'ta Paddle'ı, diğer platformlarda ONNX'i hazırlar. `--dry-run` dosya değiştirmeden yolları gösterir. `--force` indirme cache'ini ve seçilen backend çıktısını baştan hazırlar; doğrulama tamamlanana kadar mevcut çalışan backend klasörünü korur. Resmî artefactlar için dokümante edilmiş checksum yayımlanmadığından sahte checksum kullanılmaz; indirme uzunluğu, güvenli archive extraction ve model yapısı doğrulanır.

Runtime backend seçimi `auto` olduğunda geçerli ONNX modelleri önce kullanılır; ONNX hazır değilse native Paddle CPU'ya geçilir. Uygulama runtime sırasında model indirmez.

#### Advanced / Troubleshooting

ONNX dönüşümünü yeniden denemek için `python scripts/prepare_default_ocr_models.py --backend onnx` kullanılabilir. Manuel dönüşüm veya hazırlanmış ONNX klasörlerini kurmak için `scripts/convert_paddle_models.py` ve `scripts/setup_ocr_models.py` ayrı ayrı kullanılabilir.

ONNX modelleri `models/ocr/onnx/`, native modeller `models/ocr/paddle/` altında tutulur. Eski `models/ocr/detection|recognition` ONNX yapısı geriye uyumluluk için hâlâ tanınır. Her backend klasöründeki `model-info.json` kullanılan resmî model adlarını kaydeder. Binary dosyalar `.gitignore` kapsamındadır.

### ROI ve tanıma ayarları

`config/settings.json` içindeki `plate_detection` alanı kullanılır:

- `recognition_interval_ms`: Her kamera için OCR denemeleri arasındaki minimum süre. Varsayılan `500` ms.
- `min_confidence`: Database kaydı için minimum OCR güveni. Varsayılan `0.65`.
- `confirmations_required`: Plakanın kaydedilmeden önce kaç frame'de görülmesi gerektiği.
- `confirmation_window_seconds`: Confirmation oylarının geçerli olduğu süre.
- `duplicate_cooldown_seconds`: Aynı kamera ve plakanın tekrar kaydedilmesini engelleyen süre.
- `ocr_backend`: `auto`, `onnx` veya `paddle`. Varsayılan `auto`.
- `roi.ENTRY` / `roi.EXIT`: `x`, `y`, `width`, `height` şeklinde normalize `0-1` koordinatları.

Geçersiz ROI uygulamayı kapatmaz; varsayılan ROI kullanılır ve OCR durum mesajında uyarı gösterilir. Dashboard preview üzerindeki yeşil çerçeve OCR'ın taradığı alanı belirtir.

Admin kullanıcı **Ayarlar → Plaka Alanını Kalibre Et** ile son kamera frame'i üzerinde ROI çizebilir. Ayar atomik kaydedilir ve kamera preview kapanmadan OCR'a uygulanır. Aynı sayfadaki **Modelleri Kontrol Et** işlemi seçilen backend'i ve yerel modelleri arka planda doğrular.

### Tek görsel OCR testi

Model kalitesini gerçek kamera olmadan kontrol etmek için:

```powershell
python scripts/test_plate_ocr.py "C:\test-data\plate.jpg" --direction ENTRY --save-debug "data\ocr-debug"
```

Script ham OCR metnini, normalize/düzeltilmiş plakayı, validation sonucunu, kutuları, confidence ve inference süresini terminale yazar. `--save-debug` ROI, preprocessing varyantları ve çizilmiş OCR kutularını kaydeder.

Video veya görsel klasöründe geçici SQLite ile uçtan uca, headless pipeline testi:

```powershell
python scripts/test_pipeline.py "C:\test-data\plates" --direction ENTRY --sample-every 2
```

Çıktı; frame/OCR denemesi, aday, onaylanan kayıt, duplicate ve ortalama inference sürelerini içerir. Runtime logları dönen dosyalar halinde `data/logs/app.log` konumuna yazılır; plaka metni loglanmaz.

## Şu anda OCR'ı çalıştırmak için

OCR model dosyaları repository içinde bulunmaz; geliştirme ortamında ayrıca hazırlanmalıdır.

Önerilen development akışı:

```powershell
python -m pip install -r requirements-model-tools.txt
python scripts/prepare_default_ocr_models.py
python scripts/verify_ocr_models.py
```

Model indirme ve dönüştürme adımı internet bağlantısı gerektirir. Modeller hazırlandıktan sonra **CamerBound normal kullanım sırasında internet gerektirmez**; runtime hiçbir zaman otomatik model indirmez.

Hazırlık tamamlandıktan sonra:

1. Bir plaka görseliyle lokal OCR'ı deneyin:

   ```powershell
   python scripts/test_plate_ocr.py "C:\test-data\plate.jpg" --direction ENTRY
   ```

2. Uygulamayı başlatın:

   ```powershell
   python main.py
   ```

3. Dashboard kamera kartlarında `OCR: Aktif` durumunu kontrol edin.

`verify_ocr_models.py` model yoksa exit code `1` ve `OCR status: NOT READY`; modeller ve runtime hazırsa exit code `0` ve `OCR status: READY` döndürür. Production runtime internetten model indirmez.

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
│   ├── plate_recognition.py
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
├── models/ocr/                # Lokal detection/recognition modelleri (binary'ler ignore edilir)
├── scripts/
│   ├── setup_ocr_models.py
│   ├── prepare_default_ocr_models.py
│   ├── verify_ocr_models.py
│   └── test_plate_ocr.py
└── tests/
    ├── test_camera_service.py
    ├── test_config.py
    ├── test_plate_recognition.py
    ├── test_services.py
    └── test_ui_smoke.py
```

## Mimari

- `app/database.py`: SQLite bağlantısı, transaction yönetimi, şema ve başlangıç kamera kayıtları.
- `app/auth.py`: Parola hashleme, login, oturum modeli, kullanıcı oluşturma ve rol kontrolleri.
- `app/camera.py`: Kamera ayarları, worker/thread referansları ve capture yaşam döngüsünün servis sınırı.
- `app/camera_worker.py`: OpenCV kaynağını UI thread'i dışında okur, önizleme FPS'ini sınırlar ve kesintide yeniden bağlanır.
- `app/plate_recognition.py`: ROI, preprocessing, PaddleOCR adapter, plaka doğrulama, voting ve OCR worker yaşam döngüsü.
- `app/plate_service.py`: Kayıt ekleme, arama, son kayıtlar ve içerideki araç sorgusu.
- `ui/`: Servisleri kullanan PySide6 ekranları; doğrudan SQL çalıştırmaz.
- `app/config.py`: `settings.json` okur ve relative yolları uygulama köküne göre çözer.

## Sonraki geliştirme adımları

1. Varsayılan admin parolasını değiştirme ekranı eklemek.
2. Gerçek kamera açılarıyla ENTRY/EXIT ROI değerlerini kalibre etmek.
3. Kuruma ait plaka örnekleriyle OCR doğruluğunu ölçmek ve gerekirse modeli fine-tune etmek.
4. RTSP credential verisini şifreli saklamak.
5. OCR model klasörlerini içeren PyInstaller yapılandırması ve temiz Windows paket testi yapmak.

## Ayarlar

`config/settings.json`:

```json
{
  "database_path": "data/plate_tracker.db",
  "plate_detection": {
    "recognition_interval_ms": 500,
    "min_confidence": 0.65,
    "confirmations_required": 2,
    "confirmation_window_seconds": 3,
    "duplicate_cooldown_seconds": 10,
    "roi": {
      "ENTRY": {"x": 0.1, "y": 0.35, "width": 0.8, "height": 0.55},
      "EXIT": {"x": 0.1, "y": 0.35, "width": 0.8, "height": 0.55}
    }
  }
}
```

Relative veritabanı yolu mevcut terminal klasörüne değil, uygulamanın bulunduğu kök klasöre göre çözülür.
