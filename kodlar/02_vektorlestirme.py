# -*- coding: utf-8 -*-
"""
Dogal Dil Isleme Odevi 1: Metin Tabanli Veri Setleri ile Yapay Zeka Modelleri Gelistirme

Bu betik, onceden temizlenmis (lemmatized ve stemmed) veri setlerini yukler,
TF-IDF ve Word2Vec yontemleriyle vektorlestirme islemlerini uygular.
TF-IDF sonuclarini CSV olarak kaydeder ve Word2Vec modellerini egitip kaydeder.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import os
import time
import logging

# Loglama ayari
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# --- Konfigurasyon --- #
VERI_KLASORU = "/home/ubuntu/yorum_analizi_projesi/veri"
SONUC_KLASORU = "/home/ubuntu/yorum_analizi_projesi/sonuclar"
MODEL_KLASORU = "/home/ubuntu/yorum_analizi_projesi/modeller"

# On islenmis veri dosyalari (01_veri_on_isleme.py ciktilari)
STEMMED_CSV = os.path.join(SONUC_KLASORU, "stemming_sonucu.csv")
LEMMATIZED_CSV = os.path.join(SONUC_KLASORU, "lemmatization_sonucu.csv")

# TF-IDF cikti dosyalari
TFIDF_LEMMATIZED_CSV = os.path.join(SONUC_KLASORU, "tfidf_lemmatized.csv")
TFIDF_STEMMED_CSV = os.path.join(SONUC_KLASORU, "tfidf_stemmed.csv")

# Dizinlerin var oldugundan emin ol
os.makedirs(SONUC_KLASORU, exist_ok=True)
os.makedirs(MODEL_KLASORU, exist_ok=True)

# --- 1. Temizlenmis Verileri Yukleme --- #
print(f"\n--- 1. Temizlenmis Verileri Yukleme ---")

try:
    df_lemmatized = pd.read_csv(LEMMATIZED_CSV, encoding="utf-8")
    df_stemmed = pd.read_csv(STEMMED_CSV, encoding="utf-8")
    print(f"{LEMMATIZED_CSV} ve {STEMMED_CSV} basariyla yuklendi.")
    # NaN degerleri bos string ile doldur
    df_lemmatized["metin"] = df_lemmatized["metin"].fillna("")
    df_stemmed["metin"] = df_stemmed["metin"].fillna("")
except FileNotFoundError:
    print(f"HATA: Temizlenmis veri dosyalari bulunamadi. Lutfen once 01_veri_on_isleme.py betigini calistirin.")
    exit()
except Exception as e:
    print(f"HATA: Temizlenmis veriler yuklenirken sorun olustu: {e}")
    exit()

# --- 2. TF-IDF Vektorlestirme --- #
print(f"\n--- 2. TF-IDF Vektorlestirme ---")

def tfidf_vektorlestir_ve_kaydet(df, metin_sutunu, cikti_dosyasi):
    """Verilen DataFrame uzerinde TF-IDF uygular ve sonucu CSV olarak kaydeder."""
    print(f"\n{cikti_dosyasi} icin TF-IDF vektorlestirme baslatiliyor...")
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(df[metin_sutunu])
        # Feature isimlerini (kelimeleri) al
        feature_names = vectorizer.get_feature_names_out()
        # TF-IDF matrisini DataFrame'e cevir
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        # DataFrame'i CSV olarak kaydet
        tfidf_df.to_csv(cikti_dosyasi, index=False, encoding="utf-8")
        print(f"TF-IDF sonuclari kaydedildi: {cikti_dosyasi}")
        print(f"Olusturulan TF-IDF matris boyutu: {tfidf_df.shape}")
    except ValueError as ve:
        print(f"HATA: TF-IDF vektorlestirme sirasinda hata: {ve}")
        print("Veri setinde bos veya sadece stopword iceren metinler olabilir.")
    except Exception as e:
        print(f"HATA: TF-IDF vektorlestirme sirasinda beklenmedik hata: {e}")

# Lemmatized veri icin TF-IDF
tfidf_vektorlestir_ve_kaydet(df_lemmatized, "metin", TFIDF_LEMMATIZED_CSV)

# Stemmed veri icin TF-IDF
tfidf_vektorlestir_ve_kaydet(df_stemmed, "metin", TFIDF_STEMMED_CSV)

# --- 3. Word2Vec Vektorlestirme --- #
print(f"\n--- 3. Word2Vec Vektorlestirme ---")

# Word2Vec icin parametre setleri
parametreler = [
    {"model_type": "cbow", "window": 2, "vector_size": 100},
    {"model_type": "skipgram", "window": 2, "vector_size": 100},
    {"model_type": "cbow", "window": 4, "vector_size": 100},
    {"model_type": "skipgram", "window": 4, "vector_size": 100},
    {"model_type": "cbow", "window": 2, "vector_size": 300},
    {"model_type": "skipgram", "window": 2, "vector_size": 300},
    {"model_type": "cbow", "window": 4, "vector_size": 300},
    {"model_type": "skipgram", "window": 4, "vector_size": 300}
]

def word2vec_egit_ve_kaydet(df, metin_sutunu, veri_tipi):
    """Verilen DataFrame uzerinde belirtilen parametrelerle Word2Vec modellerini egitir ve kaydeder."""
    print(f"\n{veri_tipi.capitalize()} veri seti icin Word2Vec modelleri egitiliyor...")

    # Veriyi Word2Vec'in bekledigi formata getir (liste icinde kelime listeleri)
    try:
        cumleler = [metin.split() for metin in df[metin_sutunu]]
    except Exception as e:
        print(f"HATA: Word2Vec icin cumleler hazirlanirken hata: {e}")
        return

    # Bos cumleleri filtrele
    cumleler = [c for c in cumleler if c]
    if not cumleler:
        print(f"Uyari: {veri_tipi.capitalize()} veri setinde egitim icin uygun cumle bulunamadi.")
        return

    for params in parametreler:
        model_tipi = params["model_type"]
        pencere = params["window"]
        vektor_boyutu = params["vector_size"]
        sg = 1 if model_tipi == "skipgram" else 0

        # Model isimlendirmesi
        model_adi = f"word2vec_{veri_tipi}_{model_tipi}_win{pencere}_dim{vektor_boyutu}.model"
        model_yolu = os.path.join(MODEL_KLASORU, model_adi)

        print(f"\nModel Egitiliyor: {model_adi}")
        print(f"Parametreler: type={model_tipi}, window={pencere}, size={vektor_boyutu}")

        start_time = time.time()
        try:
            # Modeli egit
            model = Word2Vec(sentences=cumleler,
                             vector_size=vektor_boyutu,
                             window=pencere,
                             sg=sg, # 0: CBOW, 1: Skip-gram
                             min_count=1, # Minimum kelime frekansi (kucuk veri setleri icin 1)
                             workers=4) # Kullanilacak islemci cekirdegi sayisi

            egitim_suresi = time.time() - start_time
            print(f"Egitim tamamlandi. Sure: {egitim_suresi:.2f} saniye")

            # Modeli kaydet
            model.save(model_yolu)
            model_boyutu_mb = os.path.getsize(model_yolu) / (1024 * 1024)
            print(f"Model kaydedildi: {model_yolu} (Boyut: {model_boyutu_mb:.2f} MB)")

            # Ornek benzerlik sorgusu (Rapor icin)
            # Modelin kelime haznesinden rastgele veya anlamli bir kelime sec
            if model.wv.index_to_key:
                ornek_kelime = model.wv.index_to_key[0] # Ilk kelimeyi al
                try:
                    benzer_kelimeler = model.wv.most_similar(ornek_kelime, topn=5)
                    print(f"Ornek Benzerlik (\'{ornek_kelime}\' icin ilk 5):")
                    for kelime, skor in benzer_kelimeler:
                        print(f"  - {kelime}: {skor:.4f}")
                except KeyError:
                    print(f"Uyari: Ornek kelime \'{ornek_kelime}\' model sozlukte bulunamadi.")
                except Exception as sim_e:
                    print(f"HATA: Benzerlik sorgusu sirasinda hata: {sim_e}")
            else:
                print("Uyari: Modelin kelime haznesi bos, benzerlik sorgusu yapilamiyor.")

        except Exception as e:
            print(f"HATA: Model egitimi veya kaydi sirasinda hata: {e}")

# Lemmatized veri icin Word2Vec modellerini egit
word2vec_egit_ve_kaydet(df_lemmatized, "metin", "lemmatized")

# Stemmed veri icin Word2Vec modellerini egit
word2vec_egit_ve_kaydet(df_stemmed, "metin", "stemmed")

print("\nBetik 02_vektorlestirme.py tamamlandi.")

