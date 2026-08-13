# CamerBound

CamerBound, giriş ve çıkış kameralarından araç plakalarını okuyup kayıt altına alan Windows masaüstü uygulamasıdır. Kamera görüntüsü, OCR işleme, araç fotoğrafları, kullanıcı yönetimi ve kayıt geçmişi yerel olarak yönetilir; normal çalışma sırasında bulut servisine ihtiyaç duymaz.

## Özellikler

- ENTRY ve EXIT kameraları için canlı görüntü
- RTSP/IP kamera, webcam ve yerel video dosyası desteği
- Her kamera için ayrı ve kullanıcı tarafından kalibre edilebilir plaka ROI alanı
- PaddleOCR ve ONNX Runtime ile yerel OCR işleme
- Türk plaka normalizasyonu ve format doğrulaması
- Geçerli plakalar için iki okumalı doğrulama
- Aynı aktif aracın tekrar tekrar DB kaydı ve JPEG oluşturmasını engelleyen plate presence kontrolü
- Kamera başına duplicate cooldown koruması
- Yalnızca en güncel frame'i işleyen, backlog oluşturmayan OCR akışı
- Dashboard'da canlı OCR adayı ve kayıt durumu
- Giriş/çıkış kayıtları ve içeride bulunan araç listesi
- Yerel tarihe göre günlük kayıt arşivi
- Plaka arama ve giriş/çıkış filtresi
- Kayıt detayında araç fotoğrafı görüntüleme
- ADMIN ve USER rolleri
- ADMIN kullanıcılar için kullanıcı, kamera, saklama ve tanılama ayarları
- Audit/güvenlik günlüğü
- Otomatik kayıt saklama ve temizleme
- Windows saat senkronizasyonu tanılama
- SQLite yerel veritabanı

## Teknolojiler

- Python 3.12
- PySide6
- OpenCV
- PaddleOCR
- ONNX Runtime
- OpenVINO Runtime
- SQLite
- bcrypt
- Windows DPAPI

## Çalıştırma

Proje kök dizinindeki aşağıdaki dosyayı çalıştırın:

```text
run.bat
```

Manuel çalıştırma:

```powershell
.\.venv\Scripts\python.exe main.py
```

`run.bat`, proje içindeki `.venv` sanal ortamını kullanır. Beklenen Python çalıştırıcısı:

```text
.venv\Scripts\python.exe
```

Production log seviyesi varsayılan olarak `INFO` kalır. Kaynak kodu değiştirmeden
geçici saha tanılaması açmak için aynı PowerShell oturumunda:

```powershell
$env:CAMERBOUND_LOG_LEVEL="DEBUG"
.\run.bat
```

İzin verilen seviyeler `DEBUG`, `INFO`, `WARNING` ve `ERROR` değerleridir. Geçersiz
bir değer güvenli biçimde `INFO` seviyesine döner. Mevcut rotating log sınırları ve
kamera credential/URL sanitization davranışı DEBUG modunda da korunur.

## Kamera Yapılandırması

Kamera kaynakları ADMIN hesabıyla **Ayarlar** ekranından yapılandırılır.

Desteklenen kaynak türleri:

- RTSP/IP kamera adresi
- Webcam indeksi (`0`, `1` gibi)
- Yerel video dosyası

ENTRY ve EXIT kameralarının plaka alanları, canlı görüntü üzerinden **Plaka Alanını Kalibre Et** özelliğiyle ayrı ayrı ayarlanabilir. ROI koordinatları normalize edilmiş biçimde saklanır.

### Kamera erişim bilgileri

Kamera kullanıcı adı ve bağlantı şifresi yalnızca ADMIN kullanıcı tarafından **Kamera Erişim Bilgileri** penceresinden ayarlanabilir.

- Kamera şifresi arayüzde düz metin olarak gösterilmez.
- Kayıtlı şifre Windows DPAPI ile mevcut Windows kullanıcısına bağlı olarak korunur.
- Erişim bilgileri güncellenirken şifre alanı boş bırakılırsa kayıtlı protected password değiştirilmez.
- Kamera URL'leri, erişim bilgileri ve hassas query parametreleri audit/log kayıtlarında temizlenir.
- Kamera kullanıcı adı veya şifresi kaynak koda eklenmemelidir.

## Plaka Tanıma Akışı

```text
Kamera frame
→ Kamera başına 2 saniyelik bounded RAM rolling ring
→ ROI üzerinde lightweight motion event
→ Zamansal dağıtılmış historical replay frame seçimi
→ Detector worker: live latest frame veya replay frame üzerinde OpenVINO detector
→ En fazla iki padded plaka crop'u veya throttled ROI fallback job'u
→ Kamera başına bounded RAM OCR job buffer
→ OCR worker: preprocessing variantları ve PaddleOCR
→ Plaka normalizasyonu
→ Türk plaka doğrulaması
→ Minimum OCR kalite filtresi ve confirmation kontrolü
→ Plate presence kontrolü
→ Duplicate cooldown kontrolü
→ SQLite kayıt
→ Araç JPEG'i
```

Varsayılan OCR değerleri:

```text
İşleme aralığı: 250 ms / kamera
Minimum güven: 0.65
Normal doğrulama: 2 okuma
Doğrulama penceresi: 3 saniye
Presence release süresi: 15 saniye
Same-camera same-plate duplicate cooldown: 120 saniye
```

### Dedicated plate detector (Phase 1)

Detector açıkken OpenVINO modeli yalnızca yapılandırılmış ENTRY/EXIT ROI üzerinde çalışır. Yalnızca modelin `plate` sınıfı OCR'a gönderilir; `vehicle` sınıfı OCR'a gönderilmez. Detection confidence ve bbox alanına göre en iyi iki plaka crop'u seçilir. Detector exception verirse `fallback_to_roi_ocr=true` ile mevcut ROI OCR akışı korunur. Detector başarıyla çalışıp hiç kullanılabilir plate crop üretmezse, kamera başına ayrı tutulan 750 ms throttle süresi dolduğunda safety fallback olarak ROI OCR bir kez denenir; aradaki recognition frame'lerinde pahalı OCR çağrısı yapılmaz.

Detector ve PaddleOCR ayrı worker'larda çalışır; OCR inference sürerken detector round-robin olarak her kameranın en yeni frame'ini işlemeye devam eder. Üretilen OCR job'ları kamera başına en fazla 3 adet RAM'de tutulur ve OCR kuyruğunda 2500 ms'den uzun bekleyen job işlenmeden bırakılır. Job türleri açıkça `DETECTOR_CROP`, `DETECTOR_ERROR_FALLBACK` ve `ZERO_DETECTION_FALLBACK` olarak sınıflandırılır. Tüketici önce detector crop'larını, sonra detector-error fallback'lerini, son olarak zero-detection fallback'lerini seçer; her öncelik seviyesinde ENTRY/EXIT round-robin adaleti korunur. Buffer dolduğunda düşük öncelikli job önce evict edilir; yalnız aynı öncelikte detector confidence, crop alanı ve Laplacian sharpness kullanan ucuz kalite skoru karşılaştırılır. Kamera başına her fallback türünden en fazla bir pending job tutulur, böylece tekrar eden fallback'ler coalesce edilir ve gerçek detector crop'larını dolduramaz. OCR job, crop ile birlikte aynı full camera frame'i ve frame zamanlarını taşıdığı için confirmation gözlem zamanı ile confirmed-record JPEG'i doğru frame'e bağlı kalır. Video veya sürekli frame kaydı yapılmaz.

Detector öncesindeki rolling ring her kamera için son 2000 ms ve en fazla 20 full-resolution frame ile sınırlıdır. ROI, yaklaşık 160 piksel genişliğe küçültülüp grayscale/blur/absdiff ile ucuz bir değişen-piksel oranı hesaplanır. Motion event 500 ms pre-roll, 700 ms post-roll ve 400 ms quiet hysteresis kullanır; event en fazla 4000 ms sürer. Event frame'leri zamansal bin'lere bölünür ve her bin içindeki en keskin ROI seçilerek en fazla 8 historical frame replay edilir. Detector 2 live frame / 1 replay frame oranıyla iki kaynağı dengeler; replay kuyruğu kamera başına 2 event ve 8000 ms scheduling yaşı ile bounded'dır.

Ring ve motion event frame'leri yalnızca RAM referanslarıdır; video yazılmaz ve uygulama kapanınca tamamı kaybolur. Aynı immutable snapshot ring, event ve replay tarafından paylaşılır. Historical replay `captured_at`/`observed_at` ve full frame'i değiştirmez; confirmation ve JPEG doğru original frame'e bağlı kalır. OCR queue staleness frame yaşından değil `queued_at` sonrasındaki gerçek queue bekleme süresinden hesaplanır. Aynı `camera_id + frame_id` live ve replay tarafından görülürse ikinci kez confirmation sayılmaz. Historical zero-detection replay frame'lerinde ROI fallback açılmaz; 750 ms throttled ROI safety fallback yalnızca live detector yolunda korunur. Replay bbox'ları canlı preview overlay'ine gönderilmez.

Yaklaşık raw-frame maliyeti `genişlik × yükseklik × 3 bayt × tutulan benzersiz frame` hesabıdır. Örneğin 1920×1080 BGR frame yaklaşık 5,9 MiB'dir; 20-frame ring kamera başına yaklaşık 119 MiB üst sınırına sahiptir. Aktif/replay event'ler aynı ndarray referanslarını paylaşır ve replay kuyruğunda event başına yalnız seçilen en fazla 8 snapshot tutulur; yine de ring dışına taşan pinned event ve mevcut OCR job full-frame referansları nedeniyle gerçek toplam ring hesabından yüksek olabilir.

Varsayılan detector ayarları:

```json
{
  "enabled": true,
  "backend": "openvino",
  "min_confidence": 0.15,
  "crop_padding_ratio": 0.15,
  "max_plate_candidates_per_frame": 2,
  "fallback_to_roi_ocr": true,
  "zero_detection_roi_fallback_enabled": true,
  "zero_detection_roi_fallback_interval_ms": 750,
  "debug_overlay": false,
  "debug_detection_overlay_ttl_ms": 500
}
```

`plate_detection` seviyesindeki temel buffer ayarları `max_pending_ocr_jobs_per_camera=3`, `ocr_job_max_age_ms=2500`, `pre_detection_buffer_duration_ms=2000`, `pre_detection_buffer_max_frames_per_camera=20`, `max_replay_frames_per_event=8`, `max_pending_replay_events_per_camera=2` ve `replay_event_max_age_ms=8000` değerleridir. Debug detector kutusu son live detection güncellemesinden 500 ms sonra preview üzerinde çizilmez; `debug_overlay=false` production davranışı değişmez.

Beklenen offline model dizini:

```text
models/plate_detector/vehicle-license-plate-detection-barrier-0123/
  model.xml
  model.bin
```

Runtime model indirmez ve internete bağlanmaz. Model bulunamazsa açık diagnostic üretilir; fallback açıksa uygulama mevcut ROI OCR hattıyla çalışmaya devam eder. `debug_overlay=true` olduğunda yalnızca ADMIN dashboard preview kopyasında `PLATE 87%` benzeri kutular çizilir. Kaydedilen araç JPEG'i orijinal full frame olmaya devam eder.

Bu Open Model Zoo modeli MobileNetV2 + SSD tabanlı generic/pretrained bir araç ve plaka detector'ıdır; Türk plakaları için özel eğitilmiş değildir. Resmî model açıklamasında doğrulama alanı Çin plakaları/önden görünen araçlar olarak belirtilir. Türk plaka performansı M15/M16 saha görüntülerinde ayrıca ölçülmelidir.

Throttle edilmiş DEBUG diagnostic örneği:

```text
OCR diagnostics camera_id=2 direction=EXIT roi=1280x460 detector_ms=12.0 plates=0 det_conf=none plate_crops=1280x460 ocr_ms=75.0 total_recognition_ms=87.0 fallback=roi fallback_reason=zero-detection candidate=yes
```

Aynı saha frame'inde detector açık/kapalı karşılaştırması:

```powershell
.\.venv\Scripts\python.exe scripts\test_plate_ocr.py frame.jpg --direction EXIT --detector-mode on --save-debug debug\detector-on
.\.venv\Scripts\python.exe scripts\test_plate_ocr.py frame.jpg --direction EXIT --detector-mode off --save-debug debug\detector-off
```

### Detector modelini geliştirme ortamında hazırlama

Model hazırlama scriptini proje ana `.venv` Python'u ile tek komutta çalıştırın:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_plate_detector.py
```

Script ilk çalıştırmada ignored `.tools/openvino_model_converter/` ortamını oluşturur ve yalnızca bu development ortamına `openvino-dev==2024.6.0`, `tensorflow==2.18.0` ve converter doğrulama bağımlılıklarını kurar. CamerBound production `.venv` ortamına TensorFlow veya legacy Model Optimizer kurulmaz. OpenVINO 2024.6 Model Optimizer ve TensorFlow aynı converter Python ile çalıştırılır.

Model download ve conversion cache'i `models/plate_detector/.omz-work/` altında tutulur. Hazır cache tekrar indirilmez; final FP32 IR dosyaları otomatik olarak aşağıdaki konuma kopyalanır:

```text
models/plate_detector/vehicle-license-plate-detection-barrier-0123/model.xml
models/plate_detector/vehicle-license-plate-detection-barrier-0123/model.bin
```

Model zaten hazırsa aynı komut pip, download veya conversion çalıştırmadan tamamlanır. Zorunlu yeniden hazırlama için `--force`, yalnızca lokal dosya kontrolü için aşağıdaki komut kullanılabilir:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_plate_detector.py --check-only
```

Yerel bir Open Model Zoo checkout'u kullanmak isteyen geliştiriciler için `--omz-tools-dir C:\path\to\open_model_zoo\tools\model_tools` desteği korunur. Bütün pip/download/conversion işlemleri yalnızca bu hazırlama scripti açıkça çalıştırıldığında yapılır; `main.py` ve `run.bat` tamamen offline kalır. Model binary dosyaları Git'e eklenmez. Model kaynağı ve lisans notu [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) dosyasındadır.

Minimum OCR kalite filtresini geçen tüm geçerli ve normalize edilmiş plakalar, confidence değeri ne olursa olsun iki tutarlı observation ile doğrulanır. OCR confidence değeri yalnızca dahili filtreleme, sıralama ve DEBUG diagnostics için kullanılır; kullanıcı arayüzünde doğruluk yüzdesi olarak gösterilmez.

Plate presence kontrolüne ek olarak, aynı kamera ve aynı normalize plaka için son DB kaydı transaction içinde kontrol edilir. İlk kayıttan sonraki 120 saniye boyunca ikinci DB satırı veya JPEG oluşturulmaz. ENTRY ve EXIT kameraları birbirinden bağımsızdır ve kontrol SQLite kayıtlarından yapıldığı için uygulama yeniden başlatıldığında da devam eder.

ENTRY ve EXIT için yalnızca en güncel frame tutulur. Eski frame'ler kuyruğa eklenmez; ortak OCR worker uygun kameraları adil sırayla işler.

## Dashboard OCR Durumları

Kamera kartındaki OCR alanı canlı candidate ve karar durumunu gösterir:

- `Plaka aranıyor`
- `Doğrulanıyor (1/2)`
- `Okuma yeterli değil, kaydedilmedi`
- `Kaydedildi`
- `Zaten kaydedildi`

**Son okunan plaka** alanı canlı candidate'ı değil, DB'ye başarıyla kaydedilmiş son record'u gösterir.

## Araç Fotoğrafları

Başarılı plaka kayıtlarında mevcut kamera frame'inden JPEG araç fotoğrafı oluşturulur.

Varsayılan değerler:

```text
Maksimum genişlik: 960 px
JPEG kalitesi: 60
```

Fotoğraflar aşağıdaki dizinde saklanır:

```text
data/captures/
```

Fotoğraf yalnızca kullanıcı kayıt detayını açtığında yüklenir. Günlük arşiv ve kayıt tablosu JPEG dosyalarını önceden yüklemez.

## Plaka Kayıtları

Plaka kayıtlarının varsayılan görünümü günlük arşivdir. Günler Windows'un yerel saatine göre gruplandırılır ve en yeni gün üstte gösterilir.

Bir gün açıldığında yalnızca o yerel güne ait kayıtlar listelenir. Gün detayında şu bilgiler ve işlemler bulunur:

- Plaka
- Giriş/çıkış yönü
- Kamera
- Yerel tarih ve saat
- Fotoğraf açma
- Kayıt detay penceresi

Timestamp değerleri SQLite'ta UTC olarak saklanmaya devam eder. Gün gruplaması gösterim sırasında UTC'den yerel saate dönüştürülerek yapılır.

## Kayıt Saklama

Varsayılan kayıt saklama süresi `30 gün`dür. ADMIN kullanıcı saklama süresini aşağıdaki seçeneklerden biri olarak belirleyebilir:

- 30 gün
- 90 gün
- 180 gün
- Süresiz

Eski bir plaka kaydı temizlendiğinde ilişkili araç fotoğrafı da silinir. Tüm kayıtların temizlenmesi gibi yıkıcı işlemler ADMIN yetkisi ve açık kullanıcı onayı gerektirir.

## Saat Yönetimi

CamerBound zamanı Windows sisteminden alır ve kayıt timestamp'lerini UTC olarak saklar. Kullanıcı arayüzündeki tarih ve saatler Windows'un yerel saatine dönüştürülür.

Uygulama doğrudan bir NTP sunucusuna bağlanmaz. ADMIN kullanıcısı **Ayarlar > Saat Durumu** bölümünden Windows Time Service durumunu ve mevcut zaman kaynağını görüntüleyebilir.

## Güvenlik

- Kullanıcı şifreleri bcrypt ile hashlenir.
- Kamera bağlantı şifreleri Windows DPAPI ile korunur.
- Kamera şifreleri arayüzde, loglarda veya audit detaylarında düz metin olarak gösterilmez.
- Kamera erişim bilgilerini yalnızca ADMIN kullanıcı değiştirebilir.
- Kritik yönetim işlemleri audit günlüğüne kaydedilir.
- Günlük kullanım işlemleri USER rolüyle sınırlandırılabilir.
- Plaka kayıtları ve araç fotoğrafları yerel olarak saklanır.

## Testler

Tüm test paketini çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

OCR odaklı testler:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_plate_recognition
```

Detector odaklı testler:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_plate_detector
```

UI smoke testleri:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_ui_smoke
```
