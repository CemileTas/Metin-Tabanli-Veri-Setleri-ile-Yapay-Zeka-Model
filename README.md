# Dogal Dil Isleme Odevi 1: Metin Tabanli Veri Setleri

Bu repo, Dogal Dil Isleme dersi kapsamindaki 1. Odev icin hazirlanmistir. Odev, secilen bir metin tabanli veri seti uzerinde on isleme adimlarinin uygulanmasini, Zipf analizi yapilmasini ve TF-IDF ile Word2Vec yontemleriyle vektorlestirme islemlerini icermektedir.

## Veri Seti

*   **Kullanim Amaci:** Bu proje, genellikle musteri yorumlari, haber metinleri, sosyal medya gonderileri gibi Turkce metin verilerini islemek uzere tasarlanmistir. Veri seti, duygu analizi, konu modelleme, metin siniflandirma gibi dogal dil isleme gorevleri icin bir temel olusturabilir.
*   **Kaynak ve Temin Etme:** Bu odev icin kullanilacak veri seti **ogrenci tarafindan** secilmelidir. Ornek olarak, projede referans alinan Kaggle not defterinde kullanilan `turkish-customer-reviews-for-binary-classification` gibi bir veri seti ([https://www.kaggle.com/datasets/burhanbilenn/duygu-analizi-icin-urun-yorumlari](https://www.kaggle.com/datasets/burhanbilenn/duygu-analizi-icin-urun-yorumlari)) veya benzeri Turkce metin iceren herhangi bir veri seti kullanilabilir.
*   **Yerlestirme:** Sectiginiz veri setini (genellikle bir `.csv` dosyasi) indirip projenin ana dizinindeki `veri/` klasorunun icine yerlestirmeniz gerekmektedir. Kod icerisindeki `01_veri_on_isleme.py` betiginde yer alan `HAM_VERI_DOSYASI` degiskenini, kendi dosya adiniza gore (ornegin `veri/benim_veri_setim.csv`) guncellemeniz gerekebilir.

## Kurulum

Projeyi calistirmak icin asagidaki kutuphanelerin kurulu olmasi gerekmektedir:

```bash
pip3 install pandas numpy nltk scikit-learn matplotlib gensim
```

Ayrica, NLTK kutuphanesinin bazi veri paketlerine ihtiyaci vardir. Asagidaki Python kodunu calistirarak bu verileri indirebilirsiniz:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

## Kullanim

Proje kodlarini adim adim calistirmak icin:

1.  **Veri Setini Hazirlayin:** Yukarida "Veri Seti" bolumunde aciklandigi gibi, sectiginiz `.csv` dosyasini `veri/` klasorune kopyalayin ve gerekirse `01_veri_on_isleme.py` icindeki dosya yolunu guncelleyin.

2.  **On Isleme ve Ham Analiz:** Asagidaki komutu terminalde calistirarak veri yukleme, ham veri Zipf analizi, on isleme (stopwords, tokenization, lowercasing, lemmatization, stemming) ve temizlenmis veri kaydetme islemlerini yapin:
    ```bash
    python3.11 kodlar/01_veri_on_isleme.py
    ```
    Bu betik, `sonuclar/` klasorune `lemmatization_sonucu.csv`, `stemming_sonucu.csv` dosyalarini ve Zipf grafiklerini (`zipf_*.png`) olusturacaktir.

3.  **Vektorlestirme:** Asagidaki komutu calistirarak temizlenmis veriler uzerinde TF-IDF ve Word2Vec vektorlestirme islemlerini yapin:
    ```bash
    python3.11 kodlar/02_vektorlestirme.py
    ```
    Bu betik, `sonuclar/` klasorune TF-IDF sonuclarini (`tfidf_*.csv`) ve `modeller/` klasorune egitilmis Word2Vec modellerini (`word2vec_*.model`) kaydedecektir.

## Klasor Yapisi

```
.yorum_analizi_projesi/
|-- kodlar/                  # Python betiklerini icerir
|   |-- 01_veri_on_isleme.py
|   |-- 02_vektorlestirme.py
|-- veri/                    # Ham veri setinin yerlestirilecegi klasor
|   |-- (Ornek: turkish_customer_reviews.csv) # Bu dosya manuel eklenmelidir
|-- sonuclar/                # Analiz ve isleme sonuclarinin kaydedilecegi klasor
|   |-- stemming_sonucu.csv
|   |-- lemmatization_sonucu.csv
|   |-- tfidf_lemmatized.csv
|   |-- tfidf_stemmed.csv
|   |-- zipf_ham_veri.png
|   |-- zipf_lemmatized.png
|   |-- zipf_stemmed.png
|-- modeller/                # Egitilmis Word2Vec modellerinin kaydedilecegi klasor
|   |-- word2vec_*.model
|-- .gitignore               # Git tarafindan takip edilmeyecek dosya/klasorler
|-- README.md                # Bu dosya - Proje aciklamasi ve kullanim talimatlari
|-- gereksinimler.md         # Odev gereksinimlerinin detayli listesi
|-- todo.md                  # Proje ilerleme durumu
```

## Notlar

*   Kodlamada Turkce karakter icermeyen degisken ve fonksiyon isimleri kullanilmistir.
*   Kod icerisinde Turkce aciklama satirlari bulunmaktadir.
*   Word2Vec modelleri (`.model` dosyalari) boyutlari buyuk olabilecegi icin `.gitignore` dosyasina eklenmistir. Odev tesliminde bu dosyalar yerine sadece kodlarin paylasilmasi yeterli olabilir (odev tanimina gore).

