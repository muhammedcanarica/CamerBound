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
Kamera latest frame
→ Kamera ROI alanı
→ OpenVINO generic/pretrained plaka detector
→ En fazla iki padded plaka crop'u
→ OCR preprocessing variantları
→ OCR
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

Detector açıkken OpenVINO modeli yalnızca yapılandırılmış ENTRY/EXIT ROI üzerinde çalışır. Yalnızca modelin `plate` sınıfı OCR'a gönderilir; `vehicle` sınıfı OCR'a gönderilmez. Detection confidence ve bbox alanına göre en iyi iki plaka crop'u seçilir. Detector kapalıysa veya initialization başarısız olup `fallback_to_roi_ocr=true` ise mevcut ROI OCR akışı korunur.

Varsayılan detector ayarları:

```json
{
  "enabled": true,
  "backend": "openvino",
  "min_confidence": 0.5,
  "crop_padding_ratio": 0.15,
  "max_plate_candidates_per_frame": 2,
  "fallback_to_roi_ocr": true,
  "debug_overlay": false
}
```

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
OCR diagnostics camera_id=2 direction=EXIT roi=1280x460 detector_ms=12.0 plates=1 det_conf=0.880 plate_crops=180x52 ocr_ms=75.0 total_recognition_ms=87.0 fallback=no candidate=yes
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
