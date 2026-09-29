# BIST Shock Quant v2: Adaptive Meta-Engine

Sistem tamamen ücretsizdir: GitHub Actions, Yahoo Finance (yfinance) ve TradingView'in açık tarayıcısı kullanılır. Ücretli API ya da anahtar gerekmez, çalıştırmak için bilgisayar da gerekmez.

## v1'e göre düzeltilen mantık hataları

| # | v1 sorunu | v2 çözümü |
|---|---|---|
| 1 | 19 günlük veri vardı; istatistiksel güç yoktu | `bist_history.py` ilk çalıştırmada **3 yıllık** günlük OHLCV panelini (≈450 hisse) ve makro serileri indirir. Panel aylık gzip parçalar halinde saklanır, repo şişmez. |
| 2 | `z_*` değişkenleri sabit katsayılı dönüşümlerdi | **Gerçek zaman-serisi z-skoru**: her hisse kendi son 60 gününe göre normalize edilir. Taban dünü de içerir, bugünü içermez; ileriye bakma yok. |
| 3 | "Flow" adı yanıltıcıydı (tek mumun CLV'si) | Akış ailesi **Chaikin Money Flow (20g) + mum baskısı**. Bu bir OHLCV birikim vekilidir, emir defteri verisi değildir; rapordaki adı da "Birikim"dir. |
| 4 | Öğrenme T+3'e, doğrulama T+5'e bakıyordu; eşik öğrenildiği veride seçiliyordu | Her yerde **tek etiket** kullanılır: T+1 açılış → T+5 kapanış, net getiri. Eşik yalnızca eğitim diliminde seçilir. Walk-forward **embargo'lu** (6 gün). |
| 5 | Getiri etiketi kayıyordu, giriş fiyatı sinyal kapanışıydı | Etiketler panelden **kesin işlem günleriyle** hesaplanır; tarama kaçsa da T+5 kaymaz. Giriş fiyatı ertesi günün **açılışıdır** (gap dahil). |
| 6 | Rejim yalnızca kesitseldi | `regime.py` makro katmanı: USDTRY şoku/volatilitesi, XU100 (USD) trendi, XBANK/XU100, VIX, TUR−EEM (ülke primi vekili). TL şoku ve risk-off durumunda rejim yükselir, eşiğe prim eklenir, maruziyet azalır. |
| 7 | Maliyet ve pozisyon boyutu modeli yoktu | Likiditeye bağlı **gidiş-dönüş maliyet** (komisyon + kayma) etiketten düşülür. `portfolio.py`: **volatilite hedefli** ağırlık, **korelasyon filtresi** (ρ>0.75 ikinci hisse elenir), maks. 8 pozisyon, brüt limit ve rejim/otonomi çarpanı. |

Ek olarak:
- Canlı skor, backtest ve öğrenme **aynı** `score_frame` fonksiyonunu kullanır; eğitim ile canlı arasında formül farkı yoktur.
- IC, günlük kesitsel Spearman olarak hesaplanır. Anlamlılık, örtüşme düzeltmeli t-istatistiği ile raporlanır.
- Terfi yalnızca OOS kanıtla olur: Wilson LCB, net ortalama ve PF korumaları, en az 150 OOS işlem. Canlı defter bozulursa önceki stabil profile rollback yapılır.
- Win-rate optimizer, embargo'lu OOS havuzda global eşik ofsetini ayarlar.
- `autonomy_guard.py` (drift / safe-mode durum makinesi) değişmedi; artık gerçek veri kalitesi skoru ile besleniyor.

## Günlük akış

- **18:45 TSİ, hafta içi** (`daily_scan.yml`): `main.py` taramayı çalıştırır, sonra `shock_auditor.py` öğrenme denetimini yapar ve Telegram'a iki mesaj gelir.
- Sinyal **kapanış** verisiyle üretilir. **Ertesi gün açılışta** alınır, **5. işlem gününün kapanışında** satılır. ATR felaket stopu (2.5×ATR) yalnızca kuyruk riskine karşıdır.
- **Cumartesi** (`daily_audit.yml`): 3 yıllık veri baştan indirilir (bölünme/bedelsiz temizliği) ve model tam veriyle yeniden denetlenir.

## Kurulum (yalnızca Android telefonla)

1. GitHub uygulamasında ya da tarayıcıda repoyu açın. **Add file → Upload files** ile bu paketteki dosyaları **aynı klasör yapısıyla** yükleyin. Workflow dosyaları `.github/workflows/` altına gider (mobil tarayıcıda "masaüstü sürümü" açıkken klasör yolunu dosya adına `.github/workflows/daily_scan.yml` şeklinde yazabilirsiniz).
2. Repodan **`apply_guard_patch.py`** dosyasını silin; v1'e aitti ve artık gereksiz. `VERIFY_RESULTS.txt` de silinebilir.
3. Eski veri dosyalarına dokunmayın: `backtest_ledger.csv`, `shock_signals_log.csv`, `gecmis_veri.csv`, `shock_ai_state.json`. v1 profilleri otomatik olarak `archive_v1` altına taşınır ve v2 şablonla başlar.
4. **Actions** sekmesinde önce **"BIST v2 Haftalik Tam Veri Yenileme" → Run workflow** çalıştırın. Bu ilk seferde 3 yıllık veriyi indirir (≈5–15 dk).
5. Ardından **"BIST v2 Kapanis Taramasi + Ogrenme" → Run workflow** çalıştırın. Sonrası otomatiktir.
6. `TELEGRAM_TOKEN` ve `CHAT_ID` secret'ları v1'deki gibi kalır.

## Dosyalar

| Dosya | Görev |
|---|---|
| `config.py` | Tüm sabitler: ufuk, maliyet, filtreler, walk-forward, portföy |
| `bist_history.py` | Panel (backfill, artımlı güncelleme, bölünme tespiti) + makro |
| `features.py` | Özellikler, aile skorları, net etiketler, tarih bazlı rejim |
| `sector_flow.py` | Point-in-time korelasyon kümeleri, grup akış özellikleri, sektör panosu |
| `regime.py` | Kesitsel + makro rejim |
| `shock_engine.py` | Tek skor fonksiyonu (`score_frame`) |
| `portfolio.py` | Vol-hedefli boyut, korelasyon filtresi, stop seviyeleri |
| `shock_learner.py` | Profil öğrenme, walk-forward, terfi/rollback, kesin tarihli etiketleme |
| `win_rate_optimizer.py` | Embargo'lu eşik ofseti optimizasyonu |
| `main.py` | Günlük kapanış taraması |
| `shock_auditor.py` | Öğrenme denetimi + rapor (`data/backtest_report.json`) |
| `app.py` | Streamlit paneli (walk-forward raporu ve OOS özsermaye eğrisi dahil) |
| `autonomy_guard.py`, `shock_fetcher.py` | Değişmedi |

## Bilinen sınırlar

- **Survivorship bias:** Evren bugünkü hisselerle başlar ve sonra yalnızca genişler. Borsadan çıkmış hisselerin geçmişi eksiktir; bu nedenle backtest bir miktar iyimser olabilir.
- Yahoo'nun BIST fiyatları bölünmeye göre düzeltilmiştir, **temettüye göre düzeltilmemiştir**. Temettü günlerinde küçük bir etiket gürültüsü oluşur. %25'i aşan günlük hareketler (düzeltilmemiş kurumsal işlem) geçersiz bar sayılır.
- CDS ücretsiz ve güvenilir biçimde alınamadığı için vekil kullanılır (TUR−EEM + USDTRY volatilitesi).
- Maliyet modeli muhafazakârdır (tek yön 8 bps komisyon + likiditeye bağlı 3–60 bps kayma). Aracı kurumunuzun gerçek oranına göre `config.py` içinden ayarlanabilir.


## v2.1: Sektör / Grup Akış Katmanı ve İşaretli Öğrenme

**Amaç:** Kurumsal para tek hisseye değil, birlikte hareket eden sepetlere (tema, faktör, endeks ağırlığı) girer. Sistem artık tek hisse şokunu, o hissenin grubunda birikim olup olmadığıyla birlikte değerlendiriyor.

**Nasıl çalışır (`sector_flow.py`):**
- Her ayın ilk işlem gününde, önceki 250 günün **piyasadan arındırılmış getiri korelasyonlarıyla** hisseler en fazla 16 gruba kümelenir. Kümeleme yalnızca o tarihten önceki veriyi kullanır (ileriye bakma yok) ve takvime sabittir; böylece canlı ile backtest aynı grupları üretir.
- Grup özellikleri: grup CMF'si (birikim), birikim genişliği, 20 günlük göreli güç, lider–takipçi farkı, hissenin gruptan tek başına sapması. Bunlar birleşerek 6. aile olan `sector_score`'u oluşturur.
- Resmi sektör etiketi yerine istatistiksel küme kullanılmasının nedeni gerçek veride görüldü: S&P'de aynı birikim sinyali kümelerle IC +0.043, resmi sektörle +0.007 verdi.

**İşaretli öğrenme:** Öğrenici artık bir ailenin işaretini çevirebilir. Bunun için eğitim verisinde t ≤ −2 gerekir. Gerçek BIST verisinde "olay" ve "aktivite" (gürültülü hacim şokları) T+5'te **geri dönüş** öngörüyor (t≈−5). Yani büyük oyuncunun izi yüksek sesli hacim patlaması değil, **sessiz birikim**.

**Gerçek veriyle walk-forward (OOS, Eki 2024 – Eyl 2026, embargo'lu, net):**

| | Önce | Sonra |
|---|---|---|
| BIST (öğrenen aday, terfi etti) | −%0.64 / PF 0.84 / WR %44.3 | **+%0.40 / PF 1.14 / WR %48.5** (8 dilimin 7'sinde daha iyi; t=1.35, henüz %5 anlamlılıkta değil) |
| S&P (sektörlü şablon) | +%0.25 / PF 1.13 / t=1.0 | **+%0.34 / PF 1.18 / t=3.6** |

**Risk:** Sektör akışı adayları aynı gruba yığabileceği için portföyde **grup başına en fazla 3 pozisyon** sınırı vardır (S&P'de ayrıca resmi sektör başına 3).

**Raporlar:** Telegram'da günlük "Sektör / Grup Akış Panosu" yer alır: birikim, görece güçlü ve dağıtım grupları, 🕵️ sessiz birikim işareti. Her önerinin grubu ve sektör akış skoru da gösterilir. Panelde yeni "🧭 Sektör Akışı" sekmesi var.

**Not:** Sektör ailesinin başlangıç şablon ağırlığı (%15) ve iç formülü, iki projenin tüm verisine bakılarak seçildi. Bu hafif bir veri gözetleme (data snooping) riski taşır. Bu yüzden ağırlığın gerçek değerini, embargo'lu walk-forward ve terfi kuralları üzerinden sistemin kendisi belirler.
