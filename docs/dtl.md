# Decision Tree Learning (CART)

Implementasi DTL menggunakan **CART untuk klasifikasi**, sesuai pilihan pada
spesifikasi tugas halaman 11. Kode algoritma hanya menggunakan NumPy dan standard
library Python. Scikit-learn dipakai terpisah sebagai pembanding.

## Cara kerja

Setiap node menghitung Gini impurity `1 - sum(p_k ** 2)`. Kandidat split dinilai
dengan pengurangan Gini setelah impurity kedua anak dibobot menurut jumlah
sampelnya. Kandidat dengan pengurangan terbesar dipilih secara greedy.

- **Numerik:** kandidat ambang berada di antara dua nilai berbeda yang berurutan.
  Pengurutan dan cumulative class counts memungkinkan seluruh kandidat dinilai
  tanpa mengulang pemindaian data untuk setiap ambang.
- **Kategorikal:** setiap kategori dicoba sebagai cabang kiri, dengan seluruh
  kategori lainnya di kanan. Ini adalah split biner satu kategori melawan sisanya;
  implementasi tidak mencari seluruh kombinasi subset kategori.
- **Null:** `None`, `NaN`, `pandas.NA`, `NaT`, dan string kosong diperlakukan sebagai
  nilai hilang. Untuk setiap kandidat, model mencoba mengarahkan seluruh nilai
  hilang ke kiri dan ke kanan, lalu memilih hasil terbaik dari data training.
  Split khusus nilai hilang melawan nilai terisi juga dicoba.
- **Prediksi:** null mengikuti arah yang dipelajari; jika tidak ada null pada node
  training, arah default mengikuti anak dengan sampel lebih banyak (kiri jika seri).
  Kategori baru mengikuti arah yang sama. Data prediksi tidak mengubah metadata.
- **Penghentian:** kelas murni, tidak ada split valid, `max_depth`,
  `min_samples_split`, `min_samples_leaf`, atau `min_impurity_decrease`.
  Impurity decrease dibobot dengan proporsi sampel node terhadap seluruh training
  set. Split dengan gain nol diizinkan secara default agar interaksi seperti XOR
  masih dapat dipelajari.

Pohon deterministik untuk data dan urutan baris/kolom yang sama. Urutan kelas
adalah hasil `numpy.unique`; seri prediksi memilih kelas pertama. Seri split
mengikuti urutan fitur dan kandidat; kategori mengikuti urutan kemunculan training.
Pohon tidak menyimpan salinan dataset training setelah `fit` selesai.

## Penggunaan

Jalankan dari root repository, atau tambahkan root repository ke import path
notebook. Contoh berikut mengasumsikan CSV sudah diletakkan di `data/`.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from src.algorithms.dtl import DTL

train = pd.read_csv("data/train.csv")
X = train.drop(columns=["id", "is_canceled"])
y = train["is_canceled"]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = DTL(max_depth=12, min_samples_leaf=5)
model.fit(X_train, y_train)
prediction = model.predict(X_valid)
probabilities = model.predict_proba(X_valid)  # kolom sesuai model.classes_
print("Validation accuracy:", model.score(X_valid, y_valid))

model.save("output/models/dtl.pkl")
restored = DTL.load("output/models/dtl.pkl")
```

Nama label aktual dataset adalah `is_canceled`; `id` tidak dipakai sebagai fitur.
Pisahkan training/validation sebelum mempelajari preprocessing atau model. Inferensi
tipe fitur, kategori, dan arah null hanya dipelajari dari argumen `fit`.

Fitur yang seluruh nilai terisinya dapat dikonversi menjadi angka otomatis dianggap
numerik, termasuk string angka. Kolom lain dianggap kategorikal. Untuk kategori
yang direpresentasikan dengan kode angka, gunakan `feature_types`, misalnya
`DTL(feature_types=["numeric", "categorical"])`. Kolom training yang seluruhnya
null secara default dianggap numerik; tentukan tipe eksplisit jika diperlukan.
Kolom DataFrame saat prediksi harus memiliki nama dan urutan yang sama dengan
training. Untuk array, pemanggil bertanggung jawab menjaga urutan fitur.

API publik: `DTL` / `DecisionTreeClassifier`, `fit`, `predict`, `predict_proba`,
`score`, `save`, dan `load`. Entry point awal repo tetap dapat dipakai:
`from src.algorithms.dtl import dtl; model = dtl(X_train, y_train, max_depth=12)`.
Statistik tersedia sebagai `tree_depth_`, `n_nodes_`, `n_leaves_`, dan
`feature_importances_` (akumulasi pengurangan impurity berbobot, dinormalisasi).
File save memakai pickle; load hanya artifact lokal yang dipercaya.

## Pembanding scikit-learn

`get_baseline("dtl")` di `src/algorithms/sklearn_baselines.py` menghasilkan
`sklearn.tree.DecisionTreeClassifier(criterion="gini", random_state=0)`, sesuai
spesifikasi CART. Model pembanding tidak diimpor oleh implementasi from scratch.

Lanjutkan contoh penggunaan di atas untuk membandingkan kedua algoritma pada
representasi numerik yang sama:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from src.algorithms.sklearn_baselines import get_baseline

categorical = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numeric = [column for column in X_train.columns if column not in categorical]
preprocessing = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric),
    ("categorical", make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    ), categorical),
])
X_train_processed = preprocessing.fit_transform(X_train)
X_valid_processed = preprocessing.transform(X_valid)

scratch_processed = DTL(max_depth=12, min_samples_leaf=5)
baseline = get_baseline("dtl", max_depth=12, min_samples_leaf=5)
for name, candidate in [("CART from scratch", scratch_processed), ("scikit-learn", baseline)]:
    candidate.fit(X_train_processed, y_train)
    prediction = candidate.predict(X_valid_processed)
    print(name, accuracy_score(y_valid, prediction), f1_score(y_valid, prediction))
```

Transformer hanya di-fit pada `X_train`. Kedua model memakai partisi, target,
representasi fitur, dan batas kompleksitas yang sama. Perbandingan fitur mentah
dengan baseline one-hot juga membandingkan strategi representasi/null. Pohon
tidak harus identik karena strategi pemecahan seri dan presisi numerik berbeda.

### Hasil pemeriksaan dataset lokal

Eksperimen pada `T2-IF3170-2026-2027.zip`: 21.898 baris training dan 5.475 baris
validation, stratified 80:20, `random_state=42`, `max_depth=12`,
`min_samples_leaf=5`. `id` dan `is_canceled` dikeluarkan dari fitur.

| Model / representasi | Accuracy | F1 kelas 1 |
| --- | --- | --- |
| CART from scratch, 17 fitur mentah | 0.850046 | 0.765227 |
| CART from scratch, 30 fitur hasil preprocessing | 0.853699 | 0.768564 |
| Scikit-learn CART, 30 fitur hasil preprocessing yang sama | 0.854064 | 0.770072 |

Hasil ini merupakan pemeriksaan implementasi pada satu split, bukan hasil tuning
atau skor leaderboard. Save/load mempertahankan label dan probabilitas validation
secara identik. Model juga berhasil memprediksi 6.844 baris `test.csv` tanpa fit
ulang pada test. File test tidak memiliki label sehingga tidak dihitung skornya.

Visualisasi bonus belum diimplementasikan; fokus modul ini adalah spesifikasi
wajib DTL. Workflow CLI bersama masih berupa placeholder milik repo.
