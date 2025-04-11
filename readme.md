# 📱 Analisis Sentimen Review Aplikasi TikTok

Project ini melakukan analisis sentimen terhadap review pengguna aplikasi TikTok dari Google Play Store menggunakan algoritma Deep Learning. Tujuan utama adalah mengklasifikasikan review ke dalam tiga kategori sentimen: **positif**, **netral**, dan **negatif**.

## 🧠 Metode yang Digunakan

* Scraping review aplikasi TikTok menggunakan `google-play-scraper`
* Preprocessing teks (pembersihan teks, stopwords, tokenisasi)
* Pelabelan sentimen secara manual/otomatis
* Training model deep learning dengan berbagai skema:
   * LSTM + Word2Vec
   * LSTM + TF-IDF
   * CNN + Word2Vec

## ⚙️ Cara Menjalankan Proyek

###  A. Jalankan di Google Colab (Disarankan)

1. Buka Google Colab: https://colab.research.google.com
2. Klik tab **"Upload"** lalu pilih file `Pelatihan_model_Analisa_Sentimen_aplikasi_TikTok.ipynb`
3. Setelah notebook terbuka, jalankan setiap sel secara berurutan (Shift + Enter)
4. Pastikan dependencies sudah diinstal, jika belum:

```python
!pip install -r requirements.txt
```

Jika menggunakan file ZIP:

```python
# Upload file ZIP
from google.colab import files
uploaded = files.upload()  # Pilih file ZIP proyek

# Ekstrak file
import zipfile
import os

# Nama file ZIP yang diupload
zip_name = list(uploaded.keys())[0]

# Ekstrak file
with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall('.')
    
# Cetak folder hasil ekstrak
extracted_folder = zip_name.split('.')[0]
print(f"File telah diekstrak ke folder: {extracted_folder}")

# Pindah ke direktori proyek
os.chdir(extracted_folder)

# Install dependencies
!pip install -r requirements.txt
```

###  B. Jalankan Secara Lokal

1. Pastikan Python sudah terinstall di sistem kamu (versi 3.8+ disarankan)
2. Clone atau download project ini ke komputermu
3. Buka terminal (atau Anaconda Prompt)
4. Install semua dependensi:

```bash
pip install -r requirements.txt
```

5. Buka file `.ipynb` menggunakan Jupyter Notebook atau Jupyter Lab:

```bash
jupyter notebook pelatihan_model_Analisa_Sentimen_aplikasi_TikTok.ipynb
```

6. Jalankan setiap sel secara berurutan

## 🗃️ File Utama

* `pelatihan_model_Analisa_Sentimen_aplikasi_TikTok.ipynb` – Notebook utama yang berisi proses end-to-end: preprocessing, training, evaluasi, dan inference.
* `requirements.txt` – Daftar library dan versi yang dibutuhkan.

## 📊 Hasil Akurasi

| Skema | Metode | Split Data | Akurasi | Precision | Recall | F1-Score |
|-------|--------|------------|---------|-----------|--------|----------|
| 1     | LSTM + Word2Vec | 80/20 | 96.11% | 0.98 | 0.93 | 0.95 |
| 2     | LSTM + TF-IDF | 80/20 | 69.46% | 0.48 | 0.37 | 0.34 |
| 3     | CNN + Word2Vec | 70/30 | 96.00% | 0.98 | 0.93 | 0.95 |

### Detail Performa Model per Kategori Sentimen (Skema 1: LSTM + Word2Vec)

| Kategori | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 1.00 | 0.87 | 0.93 | 5,200 |
| Neutral | 1.00 | 0.93 | 0.96 | 1,300 |
| Positive | 0.95 | 1.00 | 0.97 | 13,500 |

### Detail Performa Model per Kategori Sentimen (Skema 3: CNN + Word2Vec)

| Kategori | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 1.00 | 0.87 | 0.93 | 7,800 |
| Neutral | 1.00 | 0.92 | 0.96 | 1,950 |
| Positive | 0.94 | 1.00 | 0.97 | 20,250 |

## 📚 Requirements

Semua dependensi dapat diinstal melalui perintah berikut:

```bash
pip install -r requirements.txt
```

Berikut adalah beberapa library utama yang digunakan:
- numpy
- pandas
- tensorflow
- scikit-learn
- nltk
- google-play-scraper
- matplotlib
- seaborn
- gensim

## 🔄 Alur Kerja Proyek

1. **Pengumpulan Data**
   - Scraping review aplikasi TikTok dari Google Play Store
   - Menyimpan data mentah dalam format CSV

2. **Preprocessing Data**
   - Pembersihan teks (menghapus karakter khusus, emoji, dll)
   - Penghapusan stopwords 
   - Case folding (mengubah semua teks menjadi huruf kecil)
   - Tokenisasi dan stemming

3. **Ekstraksi Fitur**
   - Word2Vec: Mengubah teks menjadi vektor dengan mempertahankan konteks kata
   - TF-IDF: Mengubah teks menjadi representasi numerik berdasarkan frekuensi kata

Berikut versi markdown yang sudah dirapikan agar kode tampil dengan indah dan konsisten:

---

### 4. **Arsitektur Model**

---

#### - **Model 1**: LSTM dengan embedding matrix dari Word2Vec (split data 80/20)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model1 = Sequential()
model1.add(Embedding(MAX_NUM_WORDS, 100, weights=[embedding_matrix],
                     input_length=MAX_SEQUENCE_LENGTH, trainable=False))
model1.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model1.add(Dense(64, activation='relu'))
model1.add(Dropout(0.3))
model1.add(Dense(3, activation='softmax'))

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.summary()

history1 = model1.fit(X_train_pad, y_train_cat, epochs=5, batch_size=512,
                      validation_data=(X_test_pad, y_test_cat))
```

---

#### - **Model 2**: LSTM dengan TF-IDF (split data 80/20)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

# Reshape untuk LSTM (harus 3D)
X_train_tfidf_3d = np.expand_dims(X_train_tfidf, axis=2)
X_test_tfidf_3d = np.expand_dims(X_test_tfidf, axis=2)

model2 = Sequential()
model2.add(LSTM(128, input_shape=(X_train_tfidf_3d.shape[1], 1)))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(3, activation='softmax'))

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history2 = model2.fit(X_train_tfidf_3d, y_train_cat, epochs=5, batch_size=512,
                      validation_data=(X_test_tfidf_3d, y_test_cat))
```

---

#### - **Model 3**: CNN dengan Word2Vec (split data 70/30)

```python
# Split ulang 70/30
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    df['clean_text'], df['label_encoded'],
    test_size=0.3, stratify=df['label_encoded'], random_state=42
)

# Tokenizing ulang
X_train2_seq = tokenizer.texts_to_sequences(X_train2)
X_test2_seq = tokenizer.texts_to_sequences(X_test2)

X_train2_pad = pad_sequences(X_train2_seq, maxlen=MAX_SEQUENCE_LENGTH)
X_test2_pad = pad_sequences(X_test2_seq, maxlen=MAX_SEQUENCE_LENGTH)

y_train2_cat = to_categorical(y_train2, num_classes=3)
y_test2_cat = to_categorical(y_test2, num_classes=3)

# CNN model
model3 = Sequential()
model3.add(Embedding(MAX_NUM_WORDS, 100, weights=[embedding_matrix],
                     input_length=MAX_SEQUENCE_LENGTH, trainable=False))
model3.add(Conv1D(128, 5, activation='relu'))
model3.add(GlobalMaxPooling1D())
model3.add(Dense(64, activation='relu'))
model3.add(Dropout(0.3))
model3.add(Dense(3, activation='softmax'))

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history3 = model3.fit(X_train2_pad, y_train2_cat, epochs=5, batch_size=512,
                      validation_data=(X_test2_pad, y_test2_cat))
```

---


5. **Evaluasi**
   - Pengujian akurasi model pada data testing
   - Analisis classification report (precision, recall, f1-score)
   - Perbandingan performa ketiga skema model

6. **Inference**
   - Model dapat digunakan untuk memprediksi sentimen pada review baru
   ```python
   def predict_sentiment(text):
       clean = remove_stopwords(clean_text(text))
       seq = tokenizer.texts_to_sequences([clean])
       pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
       pred = model1.predict(pad)
       return le.inverse_transform([np.argmax(pred)])
   
   # Contoh
   predict_sentiment("This app is amazing and fun to use!")
   # Output: array(['positive'], dtype=object)
   ```

## 📌 Kesimpulan

Berdasarkan hasil percobaan dengan tiga skema arsitektur model yang berbeda:

1. **LSTM + Word2Vec** dan **CNN + Word2Vec** memberikan performa terbaik dengan akurasi sekitar 96%.
2. **LSTM + TF-IDF** menunjukkan performa yang lebih rendah dengan akurasi hanya 69.46%.
3. Word2Vec terbukti lebih efektif daripada TF-IDF untuk merepresentasikan teks review dalam konteks analisis sentimen.
4. Semua model menunjukkan hasil yang sangat baik dalam mengidentifikasi sentimen positif (recall ~100%), namun sedikit lebih rendah untuk sentimen negatif (recall ~87%).

## 🙌 Terima Kasih
