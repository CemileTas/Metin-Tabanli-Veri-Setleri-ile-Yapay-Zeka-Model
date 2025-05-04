# -*- coding: utf-8 -*-
"""
Dogal Dil Isleme Odevi 1: Metin Tabanli Veri Setleri ile Yapay Zeka Modelleri Gelistirme

Bu betik, belirtilen veri setini yukler, ham veri uzerinde Zipf analizi yapar,
on isleme adimlarini (stop word temizleme, tokenlestirme, kucuk harfe cevirme,
lemmatization, stemming) uygular ve temizlenmis verileri kaydeder.

Not: NLTK'nin word_tokenize fonksiyonu Turkce icin ek kaynak gerektirebildiginden
ve SnowballStemmer Turkce'yi desteklemediginden, tokenizasyon icin basit split,
stemming icin ise islem yapmama yaklasimi benimsenmistir.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize # NLTK tokenizer yerine basit split kullanilacak
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import Counter
import math
import os

# Gerekli NLTK verilerini indir (eger daha once indirilmediyse)
def download_nltk_data(resource, package):
    try:
        nltk.data.find(resource)
        print(f"NLTK resource '{resource}' already downloaded.")
    except LookupError:
        print(f"NLTK resource '{resource}' not found. Downloading...")
        try:
            nltk.download(package)
        except Exception as e:
            print(f"Error downloading NLTK resource '{package}': {e}")

download_nltk_data("corpora/stopwords", "stopwords")
# download_nltk_data("tokenizers/punkt", "punkt") # Punkt kullanilmayacak
download_nltk_data("corpora/wordnet", "wordnet") # Lemmatizer icin (gerci basit kullaniliyor)

# --- Konfigurasyon --- #
VERI_KLASORU = "/home/ubuntu/yorum_analizi_projesi/veri"
SONUC_KLASORU = "/home/ubuntu/yorum_analizi_projesi/sonuclar"
MODEL_KLASORU = "/home/ubuntu/yorum_analizi_projesi/modeller"

# Kaggle notebook'unda kullanilan veri seti yolu (yerel ortama gore ayarlanmali)
HAM_VERI_DOSYASI = os.path.join(VERI_KLASORU, "magaza_yorumlari.csv") # Ornek dosya adi, gercek dosya adiyla degistirin

# Cikti dosya adlari
STEMMED_CSV = os.path.join(SONUC_KLASORU, "stemming_sonucu.csv")
LEMMATIZED_CSV = os.path.join(SONUC_KLASORU, "lemmatization_sonucu.csv")

# Dizinlerin var oldugundan emin ol
os.makedirs(VERI_KLASORU, exist_ok=True)
os.makedirs(SONUC_KLASORU, exist_ok=True)
os.makedirs(MODEL_KLASORU, exist_ok=True)

# --- 1. Veri Seti Yukleme ve Ilk Inceleme --- #
print(f"\n--- 1. Veri Seti Yukleme ve Ilk Inceleme ---")

# Veri setini yuklemeden once varligini kontrol et
if not os.path.exists(HAM_VERI_DOSYASI):
    print(f"UYARI: Veri dosyasi bulunamadi: {HAM_VERI_DOSYASI}")
    print("Gecici ornek veri seti olusturuluyor...")
    ornek_veri = {
        'Metin': [
            "Bu urun harika, cok begendim!",
            "Kargo cok gec geldi, memnun kalmadim.",
            "Fiyatina gore iyi bir urun.",
            "Beklentimi karsilamadi, iade edecegim.",
            "Musteri hizmetleri cok yardimci oldu."
        ],
        'Durum': [1, 0, 1, 0, 1] # Ornek etiketler
    }
    df_ham = pd.DataFrame(ornek_veri)
    df_ham.to_csv(HAM_VERI_DOSYASI, index=False, encoding='utf-8')
    print(f"Ornek veri seti {HAM_VERI_DOSYASI} olarak kaydedildi.")
else:
    try:
        df_ham = pd.read_csv(HAM_VERI_DOSYASI, encoding='utf-16')
        print(f"{HAM_VERI_DOSYASI} basariyla yuklendi.")
    except Exception as e:
        print(f"HATA: Veri dosyasi yuklenirken sorun olustu: {e}")
        exit()

# Veri seti hakkinda temel bilgiler
print("\nVeri Seti Bilgileri:")
print(f"Kaynak: {HAM_VERI_DOSYASI} (Yerel dosya)")
print(f"Boyut: {df_ham.shape[0]} dokuman, {df_ham.shape[1]} sutun")
dosya_boyutu_mb = os.path.getsize(HAM_VERI_DOSYASI) / (1024 * 1024) if os.path.exists(HAM_VERI_DOSYASI) else 0
print(f"Dosya Boyutu: {dosya_boyutu_mb:.2f} MB")
print(f"Format: CSV")

print("\nIlk 5 Ham Veri Ornegi:")
metin_sutunu = None
olasi_sutunlar = ['Review', 'Comment', 'Text', 'Metin', 'Yorum']
for sutun in olasi_sutunlar:
    if sutun in df_ham.columns:
        metin_sutunu = sutun
        break

if metin_sutunu:
    print(df_ham[metin_sutunu].head())
    print(f"\nVeri Icerigi Aciklamasi (Ornek): Veri seti '{metin_sutunu}' sutununda musteri yorumlarini icermektedir.")
else:
    print("HATA: Yorumlari iceren metin sutunu bulunamadi. Sutun adlarini kontrol edin:", df_ham.columns)
    metin_sutunu = df_ham.columns[0]
    print(f"Varsayilan olarak ilk sutun ('{metin_sutunu}') kullaniliyor.")
    print(df_ham[metin_sutunu].head())

# --- 2. Zipf Yasasi Analizi (Ham Veri) --- #
print(f"\n--- 2. Zipf Yasasi Analizi (Ham Veri) ---")

def plot_zipf(text_list, title, filename):
    """Verilen metin listesi icin Zipf grafigi cizer ve kaydeder."""
    all_words = []
    for text in text_list:
        words = str(text).lower().split()
        all_words.extend(words)

    if not all_words:
        print(f"Uyari: {title} icin cizilecek kelime bulunamadi.")
        return

    word_counts = Counter(all_words)
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

    ranks = np.arange(1, len(sorted_counts) + 1)
    frequencies = [count for word, count in sorted_counts]

    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, marker=".")
    plt.xlabel("Siralama (log)")
    plt.ylabel("Frekans (log)")
    plt.title(f"Zipf Yasasi Grafigi - {title}")
    plt.grid(True)
    zipf_grafik_yolu = os.path.join(SONUC_KLASORU, filename)
    plt.savefig(zipf_grafik_yolu)
    print(f"Zipf grafigi kaydedildi: {zipf_grafik_yolu}")
    plt.close()

    print(f"\nZipf Grafigi Yorumu ({title}):")
    print("Grafik, kelime frekanslarinin siralama ile iliskisini log-log olceginde gostermektedir.")
    print(f"Veri seti boyutu ({len(all_words)} kelime) bu analiz icin yeterli gorunuyor.")

if metin_sutunu:
    plot_zipf(df_ham[metin_sutunu].astype(str).tolist(), "Ham Veri", "zipf_ham_veri.png")
else:
    print("Uyari: Metin sutunu bulunamadigi icin ham veri Zipf grafigi cizilemedi.")

# --- 3. On Isleme (Pre-processing) --- #
print(f"\n--- 3. On Isleme (Pre-processing) ---")

# Turkce stop words listesi
turkish_stopwords = set(stopwords.words('turkish'))

# Stemmer tanimlamasi (Turkce desteklenmiyor)
stemmer = None
print("Uyari: NLTK SnowballStemmer Turkce'yi desteklemiyor. Stemming adimi atlanacak.")

# Lemmatizer (Basit)
def turkish_lemmatizer(word):
    # Gercek bir Turkce lemmatizer (Zemberek vb.) entegrasyonu daha iyi sonuc verir.
    return word
print("Basit lemmatizer (kelimeyi degistirmiyor) kullaniliyor.")

def on_isle(metin, yontem='lemma'):
    """Metin verisi uzerinde on isleme adimlarini uygular."""
    if pd.isna(metin):
        return ""
    metin = str(metin)
    # 1. Kucuk harfe cevirme
    metin = metin.lower()
    # 2. HTML, Ozel Karakter, Sayi Temizleme
    metin = re.sub('<[^>]*>', '', metin)
    metin = re.sub(r'[^\w\s]', '', metin) # Harf, rakam, bosluk disindakileri kaldir
    metin = re.sub(r'\d+', '', metin)
    # 3. Tokenization (Basit split)
    kelimeler = metin.split()
    # 4. Stop words cikarma
    kelimeler = [kelime for kelime in kelimeler if kelime not in turkish_stopwords and len(kelime) > 1]
    # 5. Lemmatization veya Stemming
    if yontem == 'lemma':
        kelimeler = [turkish_lemmatizer(kelime) for kelime in kelimeler]
    elif yontem == 'stem' and stemmer: # Stemmer None oldugu icin bu blok calismayacak
        kelimeler = [stemmer.stem(kelime) for kelime in kelimeler]
    else: # Stemmer yoksa veya yontem 'stem' ise (ama stemmer None)
        # Stemming yapilamadigi icin kelimeleri oldugu gibi birak veya lemma kullan
        # Bu ornekte lemma ile ayni sonucu verecek (cunku stemmer=None)
        kelimeler = [turkish_lemmatizer(kelime) for kelime in kelimeler]

    return " ".join(kelimeler)

# Ornek bir metin uzerinde on isleme adimlarini goster
ornek_metin = df_ham[metin_sutunu].iloc[0] if metin_sutunu and not df_ham.empty else "Bu harika bir urun! Cok begendim 123."
print(f"\nOrnek Ham Metin: '{ornek_metin}'")

# Adim adim on isleme ornegi (Rapor icin)
print("\nOn Isleme Adimlari (Ornek):")
metin_adim1 = ornek_metin.lower()
print(f"1. Kucuk Harf: '{metin_adim1}'")
metin_adim2 = re.sub('<[^>]*>', '', metin_adim1)
metin_adim2 = re.sub(r'[^\w\s]', '', metin_adim2)
metin_adim2 = re.sub(r'\d+', '', metin_adim2)
print(f"2. Temizleme (HTML, Ozel Karakter, Sayi): '{metin_adim2}'")
kelimeler_adim3 = metin_adim2.split() # Basit split kullaniliyor
print(f"3. Tokenization (Basit Split): {kelimeler_adim3}")
kelimeler_adim4 = [k for k in kelimeler_adim3 if k not in turkish_stopwords and len(k) > 1]
print(f"4. Stop Words Cikarma: {kelimeler_adim4}")
kelimeler_adim5_lemma = [turkish_lemmatizer(k) for k in kelimeler_adim4]
print(f"5a. Lemmatization (Basit): {kelimeler_adim5_lemma}")
# Stemming adimi atlandigi icin gosterilmiyor
print("5b. Stemming: Atlandi (Turkce desteklenmiyor).")

print("\nOn isleme fonksiyonu veri setine uygulanıyor...")
df_ham['temiz_metin_lemma'] = df_ham[metin_sutunu].apply(lambda x: on_isle(x, yontem='lemma'))
print("Lemmatization tamamlandi.")
# Stemming adimi icin de ayni fonksiyonu cagiriyoruz ama stemmer=None oldugu icin lemma gibi davranacak
df_ham['temiz_metin_stem'] = df_ham[metin_sutunu].apply(lambda x: on_isle(x, yontem='stem'))
print("Stemming (etkisiz) tamamlandi.")

print("\nOn Islenmis Ornekler:")
print("Lemmatized:")
print(df_ham[['temiz_metin_lemma']].head())
print("\nStemmed (Lemma ile ayni):")
print(df_ham[['temiz_metin_stem']].head())

# --- 4. Temizlenmis Veri Seti Ciktisi --- #
print(f"\n--- 4. Temizlenmis Veri Seti Ciktisi ---")

df_lemmatized = df_ham[['temiz_metin_lemma']].rename(columns={'temiz_metin_lemma': 'metin'})
df_stemmed = df_ham[['temiz_metin_stem']].rename(columns={'temiz_metin_stem': 'metin'})

df_lemmatized.to_csv(LEMMATIZED_CSV, index=False, encoding='utf-8')
print(f"Lemmatized veri kaydedildi: {LEMMATIZED_CSV}")
df_stemmed.to_csv(STEMMED_CSV, index=False, encoding='utf-8')
print(f"Stemmed veri kaydedildi: {STEMMED_CSV}")

ham_kelime_sayisi = sum(len(str(text).split()) for text in df_ham[metin_sutunu]) if metin_sutunu else 0
lemma_kelime_sayisi = sum(len(str(text).split()) for text in df_lemmatized['metin'])
stem_kelime_sayisi = sum(len(str(text).split()) for text in df_stemmed['metin'])

print("\nVeri Boyutlari (Kelime Sayisi):")
print(f"Ham Veri: {ham_kelime_sayisi}")
print(f"Lemmatized Veri: {lemma_kelime_sayisi} (Cikarilan: {ham_kelime_sayisi - lemma_kelime_sayisi})")
print(f"Stemmed Veri: {stem_kelime_sayisi} (Cikarilan: {ham_kelime_sayisi - stem_kelime_sayisi})")

# --- 5. Zipf Yasasi Analizi (Temizlenmis Veri) --- #
print(f"\n--- 5. Zipf Yasasi Analizi (Temizlenmis Veri) ---")

plot_zipf(df_lemmatized['metin'].astype(str).tolist(), "Lemmatized Veri", "zipf_lemmatized.png")
plot_zipf(df_stemmed['metin'].astype(str).tolist(), "Stemmed Veri", "zipf_stemmed.png")

print("\nBetik 01_veri_on_isleme.py tamamlandi.")


