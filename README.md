# CamerBound

CamerBound, giriş ve çıkış kameralarından araç plakalarını otomatik olarak okuyup yerel olarak kaydeden Windows masaüstü uygulamasıdır.

Proje şirket içi kullanım için geliştirilmiştir ve normal çalışma sırasında bulut servisine ihtiyaç duymaz.

## Özellikler

- Giriş ve çıkış kameralarından canlı görüntü
- PaddleOCR ile lokal plaka tanıma
- Türk plaka doğrulama
- Tekrarlı kayıtları engelleme
- Plaka giriş/çıkış geçmişi
- İçeride bulunan araçların listelenmesi
- Confirmed plaka kayıtlarında araç fotoğrafı
- Kayıt detayında araç fotoğrafı görüntüleme
- ADMIN ve USER rolleri
- Kullanıcı yönetimi
- Audit / güvenlik günlüğü
- Kamera bağlantı durumu
- Windows saat senkronizasyonu tanılama
- Otomatik kayıt saklama ve temizleme
- SQLite yerel veritabanı

## Teknolojiler

- Python 3.12
- PySide6
- OpenCV
- PaddleOCR
- PaddlePaddle
- SQLite
- bcrypt

## Çalıştırma

Windows geliştirme ortamında:

```text
run.bat
```

dosyasına çift tıklayın.

Manuel olarak:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Kamera

Kamera kaynakları ADMIN hesabıyla **Ayarlar** ekranından yapılandırılır.

Desteklenen kaynaklar:

- RTSP / IP kamera
- Webcam (`0`, `1` vb.)
- Yerel video dosyası

Production ortamında giriş kamerası olarak MOBOTIX M16, çıkış kamerası olarak MOBOTIX M15 kullanılacaktır.

Kamera kullanıcı adı ve parolaları kaynak koda eklenmemelidir.

## Plaka Tanıma

Temel akış:

```text
Kamera
→ ROI
→ OCR
→ Plaka doğrulama
→ Multi-frame confirmation
→ Duplicate kontrolü
→ SQLite kayıt
→ Araç fotoğrafı
```

OCR yalnızca belirlenen ROI alanında çalışır.

## Araç Fotoğrafları

Confirmed plaka kayıtlarında mevcut kamera frame'inden küçük bir JPEG araç fotoğrafı oluşturulur.
Kayıtlar ekranında fotoğrafı bulunan satırlarda `Aç` butonu gösterilir; bu buton mevcut Kayıt Detayı penceresini açar. Fotoğrafı olmayan satırlarda `-` gösterilir.

Varsayılan ayarlar:

```text
Maksimum genişlik: 960 px
JPEG kalite: 60
```

Fotoğraflar:

```text
data/captures/
```

altında saklanır.

## Kayıt Saklama

Varsayılan kayıt saklama süresi:

```text
30 gün
```

ADMIN kullanıcı bu süreyi 30, 90, 180 gün veya Süresiz olarak değiştirebilir.

Eski kayıt silindiğinde ilişkili araç fotoğrafı da temizlenir.

## Saat

CamerBound saati Windows sisteminden alır ve kayıtları UTC olarak saklar.

Uygulama doğrudan NTP sunucusuna bağlanmaz.

Production bilgisayarında Windows Time Service'in güvenilir bir internet zaman kaynağıyla senkronize edilmesi gerekir.

ADMIN kullanıcısı **Ayarlar > Saat Durumu** bölümünden mevcut zaman kaynağını kontrol edebilir.

## Güvenlik

- Şifreler bcrypt ile hashlenir.
- Günlük kullanımda güvenlik görevlileri USER hesabı kullanmalıdır.
- Kritik işlemler audit log'a kaydedilir.
- Kamera credential bilgileri loglarda maskelenir.
- Plaka kayıtları ve araç fotoğrafları yerel olarak saklanır.

## Mevcut Durum

Uygulamanın lokal kamera, OCR, plaka kaydı, araç fotoğrafı, kayıt arama, retention, audit ve saat tanılama özellikleri çalışmaktadır.

Sıradaki aşama:

```text
MOBOTIX M16 / M15 bağlantısı
→ gerçek araç testi
→ OCR ayarı
→ Windows deployment
```
