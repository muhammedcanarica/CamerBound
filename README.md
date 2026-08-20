# CamerBound

<p align="center">
  <img src="assets/camerbound-plate-recognition.png" alt="CamerBound plaka okuma görseli" width="640">
</p>

CamerBound, ENTRY ve EXIT kameralarından araç plakalarını yerel olarak okuyup kayıt altına alan Windows masaüstü uygulamasıdır. Kamera görüntüsü, OCR, kullanıcılar, kayıt geçmişi ve araç fotoğrafları uygulamanın çalıştığı bilgisayarda tutulur.

## Windows için İndir

[⬇️ CamerBound v1.0.0 — Windows Installer](https://github.com/muhammedcanarica/CamerBound/releases/latest/download/CamerBound_Setup.exe)

Windows 10/11 x64 içindir. Python kurulumu gerektirmez.

## Features

- RTSP/IP kamera, webcam ve yerel video kaynağı desteği
- ENTRY ve EXIT için ayrı canlı görüntü ve kalibre edilebilir plaka ROI alanı
- OpenVINO plaka detector'ı ile PaddleOCR/ONNX Runtime tabanlı yerel OCR
- Türk plaka normalizasyonu, format doğrulaması ve track tabanlı confirmation
- Presence ve kamera bazlı duplicate cooldown korumaları
- Giriş/çıkış kayıtları, günlük arşiv, arama ve fotoğraflı kayıt detayı
- ADMIN/USER rolleri, bcrypt parola hashleme ve kamera parolaları için Windows DPAPI
- ADMIN ekranında kamera, kullanıcı, saklama, saat ve OCR tanılama yönetimi
- Yerel SQLite veritabanı, audit günlüğü ve kayıt saklama temizliği

## Requirements

- Windows 10/11
- Python 3.12
- Kamera kaynağı veya test için video/görsel
- OCR ve detector model hazırlığı için internet erişimi; normal uygulama çalışması offline'dır

Ana Python bağımlılıkları PySide6, OpenCV, PaddleOCR, PaddlePaddle, ONNX Runtime, OpenVINO Runtime ve bcrypt'tir. Model dönüştürme araçları ana runtime ortamından ayrı requirements dosyalarında tutulur.

## Setup

PowerShell'de proje kökünden sanal ortamı oluşturun ve runtime bağımlılıklarını kurun:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-paddle.txt
```

Varsayılan PaddleOCR modellerini indirip yerel model dizinine hazırlayın, ardından doğrulayın:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_default_ocr_models.py --backend paddle
.\.venv\Scripts\python.exe scripts\verify_ocr_models.py --backend paddle
```

Open Model Zoo plaka detector modelini izole converter ortamında hazırlayın ve kontrol edin:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_plate_detector.py
.\.venv\Scripts\python.exe scripts\prepare_plate_detector.py --check-only
```

Hazır detector dosyaları şu konumda olmalıdır:

```text
models/plate_detector/vehicle-license-plate-detection-barrier-0123/model.xml
models/plate_detector/vehicle-license-plate-detection-barrier-0123/model.bin
```

`requirements-model-tools.txt` Paddle-to-ONNX araçlarını, `requirements-model-converter.txt` ise detector dönüştürme ortamını destekler. Bunlar ana uygulama runtime'ının yinelenen requirements dosyaları değildir.

## Run

Windows'ta:

```powershell
.\run.bat
```

Manuel çalıştırma:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Camera Configuration

ADMIN hesabıyla **Ayarlar** ekranından ENTRY ve EXIT için RTSP URL'si, webcam indeksi veya yerel video kaynağı tanımlayın. Kamera kullanıcı adı ve parolası yalnızca ilgili yönetim ekranından girilmelidir; kaynak koda veya `config/settings.json` dosyasına credential yazmayın.

Her kamera kartındaki **Plaka Alanını Kalibre Et** işlemiyle ROI'yi plakanın geçeceği bölgeye göre ayrı ayrı ayarlayın. ROI değerleri normalize koordinatlar olarak saklanır. Kalibrasyondan sonra gerçek kamera açısı, araç mesafesi ve ışık koşullarıyla ENTRY ve EXIT yönlerini ayrı ayrı doğrulayın.

## Recognition Pipeline

`CameraWorker` recognition için immutable analiz frame'leri üretir. UI preview akışı bu hattan ayrıdır. Detector, yapılandırılmış ROI üzerinde plaka adayları bulur; detector crop'ları ayrı OCR worker'ına gider. Sonuçlar normalize edilir, Türk plaka formatı ve minimum kalite kurallarından geçirilir, ardından kamera ve plate track kapsamında değerlendirilir.

OCR işi bellekte ve kamera başına bounded tutulur. Aynı track için en fazla iki farklı detector frame'i bekleyebilir: aynı `frame_id` ikinci kez confirmation sayılmaz, farklı ikinci frame korunur, üçüncü farklı frame ise yalnız daha zayıf bekleyen işi kaliteye göre değiştirir. Bu yapı iki gerçek zamansal observation'ı korurken tek aracın kuyruğu doldurmasını engeller.

Normal kayıt yolu farklı gerçek frame'lerden track bazlı confirmation ve stabilizasyon gerektirir. Pending OCR işleri track kaybolduktan sonra tamamlanabilir ve daha önce kuyruğa alınmış farklı gerçek frame'ler normal confirmation'a katılabilir. Buna karşılık historical/buffered **track-end rescue devre dışıdır**: `finalize_track` buffer'dan yeni OCR evidence üretmez ve tek observation track sonunda kayıt oluşturmaz. Historical replay, yalnız track sona ermeden normal bounded detector/OCR akışında işlendiğinde bağımsız gerçek frame kanıtı olabilir.

Detector önce ucuz raw pass çalıştırır. Enhanced/tiled pahalı recovery her frame'de koşmaz; motion event başına veya live akışta throttled aralıkla policy tarafından izin verilir ve yeni live frame bekliyorsa recovery yarıda kesilebilir. False detector crop sonrası OCR araması ve zero-detection/static fallback yolları da bounded ve düşük önceliklidir.

Kayıt öncesinde active presence kontrolü uygulanır. Kayıttan sonra aynı kamera ve aynı normalize plaka için SQLite tabanlı duplicate cooldown ikinci DB satırını ve JPEG'i engeller; bu koruma uygulama yeniden başlatıldığında da sürer. Başarılı kayıtlar mevcut normal `PlateService` yolu üzerinden seçilen temsilci full frame ile kaydedilir.

Varsayılan temel değerler `config/settings.json` içindedir: iki confirmation, 3 saniyelik confirmation penceresi, 2 saniyelik stabilizasyon penceresi, kamera başına en fazla 5 OCR işi, en fazla 2 aktif plate track ve 120 saniyelik same-camera duplicate cooldown. ROI ve runtime ayarlarını değiştirmeden önce saha ölçümü yapın.

## Diagnostics

Geçici DEBUG logları için uygulamayı aynı PowerShell oturumunda başlatın:

```powershell
$env:CAMERBOUND_LOG_LEVEL = "DEBUG"
.\run.bat
```

Loglar `data/logs/` altında tutulur. ADMIN **OCR Tanılama** ekranı frame ingest, detector hit/miss, raw/enhanced/tiled detector çağrıları ve süreleri, OCR queue/drop/stale durumu, confirmation/finalization sonuçları, kayıt sayıları ve mean/p95 gecikme metriklerini gösterir. Yön bazlı tek-frame tanılama yakalama düğmeleri `debug/recognition-frames/` altında gitignored çıktılar üretir.

Yararlı yerel araçlar:

```powershell
.\.venv\Scripts\python.exe scripts\test_plate_ocr.py frame.jpg --direction ENTRY --mode compare --save-debug debug\plate-check
.\.venv\Scripts\python.exe scripts\test_pipeline.py sample.mp4 --direction ENTRY
.\.venv\Scripts\python.exe scripts\benchmark_field_alpr.py debug\field_dataset\manifest.json --output debug\field_dataset\report.json
.\.venv\Scripts\python.exe scripts\verify_ocr_models.py --backend paddle
```

`scripts/prepare_default_ocr_models.py`, `prepare_plate_detector.py`, `setup_ocr_models.py` ve `convert_paddle_models.py` model hazırlama araçlarıdır; `test_plate_ocr.py`, `test_pipeline.py` ve `benchmark_field_alpr.py` saha/development tanısı içindir.

## Testing

Önce kaynakların derlenebilirliğini kontrol edin:

```powershell
.\.venv\Scripts\python.exe -m compileall app ui scripts tests
```

Temel hedefli testler:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config -v
.\.venv\Scripts\python.exe -m unittest tests.test_plate_detector -v
.\.venv\Scripts\python.exe -m unittest tests.test_plate_tracking -v
.\.venv\Scripts\python.exe -m unittest tests.test_plate_recognition -v
```

Tüm unittest modüllerini discovery ile çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Known Limitations

- CamerBound saha testleri yapılmış bir prototiptir; production ANPR sistemi değildir.
- Başarı; kamera açısı, ışık, araç hızı, motion blur ve plakanın frame içindeki piksel boyutuna bağlıdır.
- Kullanılan Open Model Zoo detector'ı Türk plakaları için özel eğitilmemiştir.
- EXIT kameraları, hızlı araçlar, sert açı, düşük ışık ve yansıma zorlayıcı kalabilir.
- Gerçek dünya başarı oranının `%100` olduğu iddia edilmez; her kurulum kendi saha verisiyle ölçülmelidir.

## Data / Privacy

SQLite veritabanları, kamera credential veritabanı, araç JPEG'leri, loglar, debug çıktıları ve model binary dosyaları yereldir ve Git tarafından yok sayılır. Gerçek plaka görsellerini, RTSP credential'larını veya saha veritabanlarını repoya eklemeyin. Uygulama runtime'ı model indirmez ve normal kullanım için bulut servisine ihtiyaç duymaz.

## License / Third Party

Kullanılan üçüncü taraf kütüphane ve modellerin lisans/attribution bilgileri [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) dosyasındadır.
