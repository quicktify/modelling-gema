# Dokumentasi Model NLP untuk Analisis Ulasan Aplikasi

## Daftar Isi
1. [Sentiment & Emotion Analysis](#sentiment--emotion-analysis)
2. [Spam Detection](#spam-detection)
3. [Rating Estimation](#rating-estimation)

## Sentiment & Emotion Analysis

### Deskripsi
Model ini dirancang untuk menganalisis sentimen dan emosi dari ulasan aplikasi dalam bahasa Indonesia. Model ini dapat mengklasifikasikan ulasan menjadi:
- Sentimen: Positive/Negative
- Emosi: Anger, Fear, Joy, Love, Sadness, Surprise

### Teknologi yang Digunakan
- Python 3.12
- Scikit-learn
- Sastrawi (untuk stemming bahasa Indonesia)
- Imbalanced-learn (untuk menangani data tidak seimbang)
- Pandas & NumPy
- Matplotlib & Seaborn

### Fitur Utama
1. **Preprocessing Teks**
   - Pembersihan teks (URL, mention, hashtag)
   - Normalisasi kata slang
   - Penghapusan stopwords
   - Stemming (opsional)

2. **Model Klasifikasi**
   - Support Vector Machine (SVM)
   - TF-IDF Vectorization
   - SMOTE untuk menangani data tidak seimbang

3. **Evaluasi Model**
   - Confusion Matrix
   - Classification Report
   - Visualisasi hasil

### File Model
- `sentiment_model.pkl`: Model untuk klasifikasi sentimen
- `emotion_model.pkl`: Model untuk klasifikasi emosi

## Spam Detection

### Deskripsi
Model ini dirancang untuk mendeteksi ulasan spam dalam bahasa Indonesia. Model ini dapat mengklasifikasikan ulasan menjadi spam atau non-spam.

### Teknologi yang Digunakan
- Python 3.12
- Scikit-learn
- Sastrawi
- Pandas & NumPy
- Matplotlib & Seaborn

### Fitur Utama
1. **Preprocessing Teks**
   - Pembersihan teks
   - Normalisasi kata slang
   - Penghapusan stopwords
   - Stemming (opsional)

2. **Custom Feature Extraction**
   - Panjang teks
   - Jumlah kata
   - Rata-rata panjang kata
   - Rasio huruf kapital
   - Jumlah tanda tanya dan seru
   - Deteksi URL
   - Pengulangan karakter
   - Kata tanya
   - Kata hiperbolik/promosi

3. **Model Klasifikasi**
   - Naive Bayes
   - Support Vector Machine (SVM)
   - TF-IDF Vectorization

4. **Evaluasi Model**
   - Confusion Matrix
   - Classification Report
   - Visualisasi hasil

## Rating Estimation

### Deskripsi
Model ini dirancang untuk memperkirakan rating aplikasi berdasarkan ulasan pengguna dalam bahasa Indonesia.

### Status
Model ini masih dalam tahap pengembangan.

## Cara Penggunaan

### Prasyarat
1. Python 3.12 atau lebih baru
2. Install dependencies:
```bash
pip install pandas numpy scikit-learn imbalanced-learn Sastrawi matplotlib seaborn
```

### Menjalankan Model
1. **Sentiment & Emotion Analysis**
```python
import pickle

# Load model
with open('sentiment_model.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)
with open('emotion_model.pkl', 'rb') as f:
    emotion_model = pickle.load(f)

# Prediksi
text = "Aplikasi ini sangat bagus dan mudah digunakan!"
sentiment = sentiment_model.predict([text])
emotion = emotion_model.predict([text])
```

2. **Spam Detection**
```python
import pickle

# Load model
with open('spam_model.pkl', 'rb') as f:
    spam_model = pickle.load(f)

# Prediksi
text = "Aplikasi ini sangat bagus dan mudah digunakan!"
is_spam = spam_model.predict([text])
```

## Struktur Folder
```
Modelling/
├── Sentiment_Emotion/
│   ├── classification-sentiment.ipynb
│   ├── new_classification_sentiment_emotion.ipynb
│   ├── emotion_model.pkl
│   ├── sentiment_model.pkl
│   ├── confusion_matrix_emosi.png
│   └── confusion_matrix_sentimen.png
├── Spam_Detection/
│   └── spam_detection.ipynb
└── Estimation_Rating/
    └── (dalam pengembangan)
```

## Kontributor
- [Nama Kontributor]

## Lisensi
[Spesifikasikan lisensi]
