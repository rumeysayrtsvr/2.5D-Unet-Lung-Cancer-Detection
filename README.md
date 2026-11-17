# 🫁 Lung AI Systems

Akciğer kanseri için iki derin öğrenme modelini bir Flask web arayüzünde birleştiren proje:

1. **CT Tümör Segmentasyonu**: 2.5D DenseUNet (PyTorch). Ardışık 3 CT slice'ını (önceki, orta, sonraki) 3 kanallı girdi olarak alır ve ortadaki slice için tümör maskesi üretir. NLSTseg veri seti üzerinde eğitildi.
2. **Histopatoloji Sınıflandırması**: ResNet50 (PyTorch). Akciğer doku görüntüsünü *Adenocarcinoma*, *Squamous Cell Carcinoma* veya *Normal* olarak sınıflandırır. LC25000 veri seti üzerinde eğitildi.

> ⚠️ Bu proje eğitim ve araştırma amaçlıdır. Klinik tanı için kullanılamaz.

## Kurulum

Gereksinimler: Python 3.9+ ve büyük model dosyaları için [Git LFS](https://git-lfs.com).

```bash
git lfs install
git clone <bu deponun adresi>
cd 2.5D-Unet
pip install -r requirements.txt
```

## Çalıştırma

```bash
python3 app.py
```

Tarayıcıda **http://127.0.0.1:5001** adresini açın.

> macOS'ta 5000 portu AirPlay Alıcısı tarafından kullanıldığı için uygulama 5001 portunda çalışır.

## Kullanım

- **CT Segmentasyonu:** Aynı hastaya ait ardışık **3 adet `.npy` slice** seçin, örneğin `input/101834_slice_089/090/091_tumor.npy`. Sonuç ekranında ortadaki slice ve tahmin edilen tümör maskesi gösterilir.
- **Histopatoloji:** Tek bir `.jpeg` görüntü yükleyin, örneğin `input/lungaca3918.jpeg`.

## Proje Yapısı

| Dosya | Açıklama |
|---|---|
| `app.py` | Flask web uygulaması |
| `networks.py`, `model.py` | 2.5D DenseUNet mimarisi ve yükleme |
| `classifier.py` | ResNet50 sınıflandırıcı |
| `preprocess.py` | 2.5D girdi oluşturma, HU normalizasyonu, son işleme |
| `nifti_to_slices.py` | NIfTI CT hacimlerini 2D `.npy` slice'lara dönüştürme |
| `select_tumor_slices.py` | Tümör içeren slice'ları ayıklama |
| `prepare_splits.py`, `make_patient_split.py` | Hasta bazlı train/val ayrımı |
| `models/` | Eğitilmiş model ağırlıkları (Git LFS) |
| `input/` | Örnek CT slice'ları ve histopatoloji görüntüleri |
| `templates/index.html` | Web arayüzü |

## Veri Setleri

Veri setleri boyutları nedeniyle depoya dahil edilmemiştir:

- **NLSTseg**: NLST CT taramaları üzerinde piksel düzeyinde tümör maskeleri
- **LC25000**: Akciğer ve kolon histopatoloji görüntüleri
