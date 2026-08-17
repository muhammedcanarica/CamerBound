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
→ CameraWorker'ın yaklaşık 12 FPS worker-throttled immutable analysis frame'i
→ UI event-loop'undan bağımsız kamera başına 2 saniyelik bounded RAM rolling ring
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

Detector açıkken OpenVINO modeli yalnızca yapılandırılmış ENTRY/EXIT ROI üzerinde çalışır. Yalnızca modelin `plate` sınıfı OCR'a gönderilir; `vehicle` sınıfı OCR'a gönderilmez. Detection confidence ve bbox alanına göre en iyi iki plaka crop'u seçilir. Detector exception verirse `fallback_to_roi_ocr=true` ile mevcut ROI OCR akışı korunur. Detector başarıyla çalışıp hiç kullanılabilir plate crop üretmezse safety fallback yalnız meaningful motion event içinde, 750 ms throttle'a da uyarak event başına en fazla bir kez denenir; statik yolda pahalı OCR çağrısı yapılmaz.

Detector ve PaddleOCR ayrı worker'larda çalışır; OCR inference sürerken detector round-robin olarak her kameranın en yeni frame'ini işlemeye devam eder. Üretilen OCR job'ları kamera başına en fazla 3 adet RAM'de tutulur. ROI fallback job'ları OCR kuyruğunda 2500 ms, gerçek detector crop'ları ise uzun bir non-preemptive fallback sırasında kaybolmamaları için bounded olarak 12000 ms bekleyebilir. Job türleri açıkça `DETECTOR_CROP`, `DETECTOR_ERROR_FALLBACK`, `ZERO_DETECTION_FALLBACK` ve en düşük öncelikli `STATIC_ZERO_DETECTION_RESCUE` olarak sınıflandırılır. Tüketici yüksek öncelikli detector crop'larından başlayarak işler; her öncelik seviyesinde ENTRY/EXIT round-robin adaleti korunur. Buffer dolduğunda düşük öncelikli job önce evict edilir; yalnız aynı öncelikte detector confidence, crop alanı ve Laplacian sharpness kullanan ucuz kalite skoru karşılaştırılır. Kamera başına her fallback türü bounded tutulur, böylece tekrar eden fallback'ler gerçek detector crop'larını dolduramaz. OCR job, crop ile birlikte aynı immutable full camera frame referansını ve frame zamanlarını taşıdığı için confirmation gözlem zamanı ile confirmed-record JPEG'i doğru frame'e bağlı kalır. Video veya sürekli frame kaydı yapılmaz.

UI preview ve recognition frame yolları ayrıdır. `CameraWorker` tarafından worker seviyesinde throttle edilip bir kez kopyalanan frame önce `analysis_frame_ready` üzerinden doğrudan thread-safe recognition ingest yoluna bırakılır. Aynı immutable ndarray daha sonra UI için latest-frame coalescing'e girer; yavaş UI event loop analysis ring'e ulaşan frameleri düşürmez. Recognition ingest callback'i UI/SQLite/inference çalıştırmaz ve kamera yönünü startup'ta hazırlanan runtime cache'den okur.

Zero-detection ROI fallback yalnız meaningful motion event sırasında veya event kapanış frame'inde üretilebilir. İki denemelik toplam bütçenin yalnız ilki live yolda harcanabilir; ikinci deneme olay sonundaki daha geç ve daha kaliteli frame için ayrılır. Böylece araç uzaktayken yapılan erken deneme, yakın/okunaklı son kareyi kör bırakmaz. Historical replay frame'lerinde zero detection için full ROI OCR çalıştırılmaz. Motion üretmeyen fakat detector'ın sürekli kaçırdığı sabit araç için 2500 ms warm-up/cooldown'lu, tek pending job'lı ve en düşük öncelikli static rescue korunur. Detector-error fallback bu motion kuralından bağımsız kalır.

Geniş ROI doğrudan 256×256 model girişine dönüştürüldüğünde yatay ve dikey ölçekler DEBUG logunda ayrıca raporlanır. Birincil detector ve kontrollü shadow pass sonuç vermezse en fazla üç örtüşen yatay tile üzerinde bounded recovery çalışır; tile koordinatları tekrar original ROI koordinatlarına çevrilir ve çakışan sonuçlar birleştirilir. Tile recovery crop'ları küçük plaka çevresindeki bağlamı kaybetmemek için ayrı `tiled_recovery_crop_padding_ratio=0.5` ayarını kullanır; normal detector crop padding'i `0.15` kalır. Başarılı birincil detector çağrısında tile maliyeti yoktur.

Detector crop için mean brightness tek başına kullanılmaz. Her crop'ta mean/median/p10/p90, percentile dynamic range, grayscale standard deviation, local contrast, Laplacian sharpness ve siyah/beyaz saturation oranları ucuz deterministic istatistiklerle ölçülür; crop dahili olarak `NORMAL`, `LOW_LIGHT`, `SHADOW_LOW_CONTRAST` veya `OVEREXPOSED` profiline ayrılır. Mevcut 3 variant ve karanlık crop'taki mevcut dördüncü low-light variant ilk hızlı deneme olarak korunur. `NORMAL` ve `OVEREXPOSED` crop'larda ek OCR çağrısı yapılmaz. İlk deneme geçerli aday üretmezse, minimum confidence altında kalırsa veya zor ışık crop'ında ayrı variant kümeleri iki geçerli plaka arasında eşit desteğe sahipse yalnız saha A/B ölçümünde seçilen shape-preserving gamma-gray variantı ikinci ve son batch/tie-breaker olarak denenir. Validator, correction cost, confirmation, queue ve job priority kuralları değişmez.

Full ROI fallback maksimum 960 px genişlikte iki hafif variant kullanır: önce compact color variant tek başına denenir; geçerli ve minimum kaliteyi geçen plaka bulunamazsa CLAHE/low-light enhanced ikinci variant ayrı çağrıda denenir. Böylece büyük ENTRY ROI için 2x upscale dahil 3–4 variantı tek Paddle çağrısına verme kaldırılmıştır. PaddleOCR CPU kullanımı desteklenen `cpu_threads=4` ayarıyla sınırlandırılarak OpenVINO detector'ın CPU starvation riski azaltılır.

Detector öncesindeki rolling ring her kamera için son 2000 ms ve en fazla 20 full-resolution frame ile sınırlıdır. ROI, yaklaşık 160 piksel genişliğe küçültülüp grayscale/blur/absdiff ile ucuz bir değişen-piksel oranı hesaplanır. Motion event 500 ms pre-roll, 700 ms post-roll ve 400 ms quiet hysteresis kullanır; event en fazla 4000 ms sürer. Event frame'leri zamansal bin'lere bölünür ve her bin içindeki en keskin ROI seçilerek en fazla 8 historical frame replay edilir. Detector 2 live frame / 1 replay frame oranıyla iki kaynağı dengeler; replay kuyruğu kamera başına 2 event ve 8000 ms scheduling yaşı ile bounded'dır.

Ring ve motion event frame'leri yalnızca RAM referanslarıdır; video yazılmaz ve uygulama kapanınca tamamı kaybolur. Aynı immutable snapshot ring, event ve replay tarafından paylaşılır. Historical replay `captured_at`/`observed_at` ve full frame'i değiştirmez; confirmation ve JPEG doğru original frame'e bağlı kalır. OCR queue staleness frame yaşından değil `queued_at` sonrasındaki gerçek queue bekleme süresinden hesaplanır. Aynı `camera_id + frame_id` live ve replay tarafından görülürse ikinci kez confirmation sayılmaz. Historical zero-detection replay frame'lerinde ROI fallback açılmaz; 750 ms throttled ROI safety fallback yalnızca live detector yolunda korunur. Replay bbox'ları canlı preview overlay'ine gönderilmez.

Yaklaşık raw-frame maliyeti `genişlik × yükseklik × 3 bayt × tutulan benzersiz frame` hesabıdır. Örneğin 960×540 BGR frame yaklaşık 1,48 MiB, 1920×1080 frame yaklaşık 5,93 MiB'dir. 40-frame ring bu çözünürlüklerde kamera başına yaklaşık 59 MiB veya 237 MiB; iki kamera için yaklaşık 119 MiB veya 475 MiB üst sınırına sahiptir. Active motion event densest-temporal removal ile ayrıca en fazla 16 frame tutar; replay event başına seçim 10, pending replay event kamera başına 2 ve OCR queue kamera başına 3 ile sınırlıdır. Event/replay aynı immutable ndarray referanslarını mümkün olduğunda paylaşır; ring dışına taşan pinned event ve OCR job referansları nedeniyle gerçek toplam ring hesabından yüksek olabilir ama her katman bounded kalır.

Varsayılan detector ayarları:

```json
{
  "enabled": true,
  "backend": "openvino",
  "min_confidence": 0.15,
  "crop_padding_ratio": 0.15,
  "tiled_recovery_crop_padding_ratio": 0.5,
  "max_plate_candidates_per_frame": 2,
  "fallback_to_roi_ocr": true,
  "zero_detection_roi_fallback_enabled": true,
  "zero_detection_roi_fallback_interval_ms": 750,
  "debug_overlay": false,
  "debug_detection_overlay_ttl_ms": 500
}
```

`plate_detection` seviyesindeki temel buffer/CPU ayarları `max_pending_ocr_jobs_per_camera=3`, `ocr_job_max_age_ms=2500`, `detector_crop_ocr_job_max_age_ms=12000`, `ocr_cpu_threads=4`, `pre_detection_buffer_duration_ms=5000`, `pre_detection_buffer_max_frames_per_camera=40`, `max_replay_frames_per_event=10`, `max_pending_replay_events_per_camera=2` ve `replay_event_max_age_ms=8000` değerleridir. 12 FPS analysis ingest'te 40 frame yaklaşık 3,25 saniyelik timestamp span sağlar; daha düşük ingest hızında duration limiti 5 saniyede devreye girer. Debug detector kutusu son live detection güncellemesinden 500 ms sonra preview üzerinde çizilmez; `debug_overlay=false` production davranışı değişmez.

Beklenen offline model dizini:

```text
models/plate_detector/vehicle-license-plate-detection-barrier-0123/
  model.xml
  model.bin
```

Runtime model indirmez ve internete bağlanmaz. Model bulunamazsa açık diagnostic üretilir; fallback açıksa uygulama mevcut ROI OCR hattıyla çalışmaya devam eder. `debug_overlay=true` olduğunda yalnızca ADMIN dashboard preview kopyasında `PLATE 87%` benzeri kutular çizilir. Kaydedilen araç JPEG'i orijinal full frame olmaya devam eder.

Bu Open Model Zoo modeli MobileNetV2 + SSD tabanlı generic/pretrained bir araç ve plaka detector'ıdır; Türk plakaları için özel eğitilmiş değildir. Resmî model açıklamasında doğrulama alanı Çin plakaları/önden görünen araçlar ve minimum 96 piksel plaka genişliği olarak belirtilir. Türk plaka performansı saha görüntülerinde ayrıca ölçülmelidir. Model değiştirilmemiştir; production offline model ve Apache-2.0 bildirim sınırı korunur.

Throttle edilmiş DEBUG diagnostic örneği:

```text
OCR worker diagnostics camera_id=1 direction=ENTRY profiles=LOW_LIGHT crop_quality=crop0=131x48,mean=42.2,median=31.0,p10=22.0,p90=91.0,range=69.0,stddev=26.1,local=6.1,sharpness=178.2 current_variants=4 shadow_variants=0 inference_calls=1 job_type=DETECTOR_CROP frame_id=123 candidate=01KAC53 recognition_state=AWAITING_CONFIRMATION
```

DEBUG kaydı variant adı, Paddle text-box sayısı, raw/normalized/corrected text, correction cost, OCR confidence, bbox, validator sonucu ve rejection reason alanlarını bounded olarak içerir. Böylece `text_detection_boxes=0` ile text box üretildiği halde geçerli Türk plakası çıkmaması birbirinden ayrılır. Log camera başına throttle edilir ve INFO seviyesini doldurmaz.

Admin OCR Tanılama ekranı model dosyaları ve servis durumuna ek olarak çalışma süresindeki frame ingest, detector hit/miss, queued/processed OCR işi, inference hatası, kayıt, son OCR aktivitesi ve bounded queue drop/stale sayaçlarını gösterir. Kamera başına ring depth/cap, effective history, ingest FPS, tahmini ring RAM ve active-event frame sayısı; son 128 detector/OCR/queue-wait/end-to-end işlemin mean/p95 süreleri de raporlanır. Böylece `ACTIVE` yalnız initialization başarısı olarak kalırken gerçek frame/inference aktivitesi ayrıca görülebilir.

Aynı saha frame'inde gerçek OpenVINO/PaddleOCR aşamalarını karşılaştırma:

```powershell
.\.venv\Scripts\python.exe scripts\test_plate_ocr.py frame.jpg --direction EXIT --mode detector-only --save-debug debug\detector-only
.\.venv\Scripts\python.exe scripts\test_plate_ocr.py frame.jpg --direction EXIT --mode detector-ocr --save-debug debug\detector-ocr
.\.venv\Scripts\python.exe scripts\test_plate_ocr.py frame.jpg --direction EXIT --mode roi-ocr --save-debug debug\roi-ocr
.\.venv\Scripts\python.exe scripts\test_plate_ocr.py frame.jpg --direction EXIT --mode compare --save-debug debug\compare
```

`compare`, doğrudan resize, aspect-preserving letterbox ve bounded tile detector sonuçlarını; ardından current detector-crop, ayrı shadow-color baseline, production için seçilmiş shadow grayscale ve full-ROI OCR ham segmentlerini raporlar. `production` modu detector crop yoksa live motion fallback'in offline proxy'sinde production ile ortak bounded spatial-search helper'ını çalıştırır; üç tile da miss olursa full-ROI safety fallback'e geçer. Crop metrikleri, profile, preprocessing/inference süresi, inference çağrı sayısı, text-box sayısı ve en iyi aday ayrı satırlardadır. `--save-debug` source, ROI, detector overlay, original detector crop, adlandırılmış current/shadow variantları ve OCR result kutularını yazar. Normal UI'yi veya production ayarlarını değiştirmez.

Birden çok gerçek saha frame'i için developer-local manifest `debug/field_dataset/manifest.json` altında tutulabilir. `debug/` Git tarafından yok sayıldığı için gerçek plaka görüntüleri ve benchmark raporları repository'ye eklenmez. OpenVINO ve PaddleOCR modellerini tek runtime içinde tekrar kullanan benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_field_alpr.py debug\field_dataset\manifest.json --label before --recognition-only-ab --output debug\field_dataset\before.json
.\.venv\Scripts\python.exe scripts\benchmark_field_alpr.py debug\field_dataset\manifest.json --label after --baseline debug\field_dataset\before.json --output debug\field_dataset\after.json
```

Rapor primary/tiled detector recall, fallback kullanımı, spatial rescue tile sayısı, exact ve character accuracy, confusion çiftleri, false/no-read oranı, detector/OCR/end-to-end mean-p50-p95 ve CPU inference çağrılarını içerir. Detector-miss event sonunda cheap blur/contrast/exposure proxy ile seçilmiş historical frame'in geniş ROI'si en fazla üç örtüşen yatay Paddle text-search tile'ına ayrılır. İlk geçerli Türk plakası early-exit yapar; üç tile da miss olursa mevcut iki aşamalı compact full-ROI fallback güvenlik ağı kalır. Detector-crop job'ları bu düşük öncelikli bounded rescue'dan önce işlenir. Tek JPEG benchmark'ı multi-frame `SAVED` kararı üretmediği için save precision veya ambiguous-discard metriğini uydurmaz; bunları candidate metriğinden açıkça ayırır. `--recognition-only-ab` yalnız development A/B ölçümüdür ve production davranışını değiştirmez.

Aynı kamera/açıdan gölge ve güneş frame'lerini tek komutta karşılaştırmak için:

```powershell
.\.venv\Scripts\python.exe scripts\test_plate_ocr.py shadow.jpg --compare-image sun.jpg --direction ENTRY --mode compare --save-debug debug\shadow-sun
```

Canlı recognition hattına giren bir sonraki immutable raw frame'i development tanısı için tek sefer kaydetmek gerekirse uygulamayı aynı PowerShell oturumunda şu şekilde başlatın:

```powershell
$env:CAMERBOUND_CAPTURE_NEXT_RECOGNITION_FRAME="1"
.\run.bat
```

Yalnız ilk detector frame'inin eşleşen `-full.jpg` ve exact configured ROI `-roi.jpg` çifti `debug/recognition-frames/` altına yazılır; klasör Git tarafından yok sayılır ve özellik varsayılan olarak kapalıdır. DEBUG OCR diagnostics, worker aşaması, job türü, fallback nedeni, raw OCR segmenti, normalize/valid aday sonucu ve nihai rejection state'ini birlikte raporlar.

Uygulama zaten çalışıyorsa ADMIN `OCR Tanılama` bölümündeki yön-hedefli `Sonraki ENTRY/EXIT Frame'ini Kaydet` düğmeleri aynı tek-shot kaydı runtime'da armar; diğer kameranın frame'i tetikleyiciyi tüketmez ve kamera/OCR worker yeniden başlatılmaz. Düğmeye hedef araç configured ROI içindeyken basın; dosya adı ve INFO capture satırındaki `frame_id`, aynı frame'in detector/queue/OCR satırlarıyla kesin eşleşir. Capture edilen tek frame için bu iki ayrıntılı satır INFO seviyesinde de üretilir ve diagnostic throttle bypass edilir; normal frameler yalnız throttled DEBUG olarak kalır.

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

Minimum OCR kalite filtresini geçen tüm geçerli ve normalize edilmiş plakalar, confidence değeri ne olursa olsun iki tutarlı observation ile doğrulanır. Provisional doğrulamadan sonra yapılandırılmış stabilizasyon penceresi bütünüyle beklenir; aynı frame'deki preprocessing variantları tek bağımsız observation sayılır. Normal crop variantları iki yakın geçerli aday arasında yalnız tek-vote margin ile çatışır ve tekil rakip daha güçlü OCR confidence taşırsa mevcut tek shadow-gray variantı bounded tie-breaker olarak bir kez çalışır; temiz normal crop ek inference almaz. Pencere içinde kalan tek karakterlik çatışma bağımsız frame oylarıyla çözülür, yeterli üstünlük yoksa kayıt oluşturulmaz. Detector bbox'ları belirgin biçimde farklıysa iki aracın near-conflict oyları aynı spatial session'a karıştırılmaz; bbox'sız fallback evidence konservatif kalır. D/O gibi karakterler plaka değerine özel bir kuralla birbirine dönüştürülmez. OCR confidence değeri yalnızca dahili filtreleme, sıralama ve DEBUG diagnostics için kullanılır; kullanıcı arayüzünde doğruluk yüzdesi olarak gösterilmez.

Plate presence kontrolüne ek olarak, aynı kamera ve aynı normalize plaka için son DB kaydı transaction içinde kontrol edilir. İlk kayıttan sonraki 120 saniye boyunca ikinci DB satırı veya JPEG oluşturulmaz. ENTRY ve EXIT kameraları birbirinden bağımsızdır ve kontrol SQLite kayıtlarından yapıldığı için uygulama yeniden başlatıldığında da devam eder.

ENTRY ve EXIT için detector'ın live latest-frame slotunda yalnızca en güncel frame tutulur. Bundan bağımsız pre-detection analysis ring son 5 saniyenin en fazla 40 worker-throttled frame'ini tutar; ortak OCR worker uygun kameraları öncelik ve kamera adaletiyle işler.

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
