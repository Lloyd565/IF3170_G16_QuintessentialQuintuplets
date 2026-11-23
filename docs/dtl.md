# Decision Tree Learning

DTL menggunakan entropy dan information gain untuk memilih percabangan biner.
Model dikhususkan untuk klasifikasi `is_canceled`: 0 atau 1.

Alur repo: CSV -> pemisahan train/validation -> cleaning -> encoding/scaling ->
seleksi fitur dan PCA opsional -> training. DTL menerima matriks numerik yang
sudah bersih. Kategori dan null ditangani oleh preprocessing yang sudah tersedia,
sehingga model tidak mengulang encoding atau imputasi.

Parameter: `max_depth`, `min_samples_split`, `min_samples_leaf`, dan
`min_impurity_decrease`. Information gain untuk parameter terakhir dibobot
dengan proporsi sampel node terhadap seluruh training set.

```python
from src.algorithms.dtl import dtl
from src.core.serialization import saveModel, loadModel
from src.processing.pipeline import runPreprocessing

X_train_prepared, y_train, state = runPreprocessing(
    X_train, y_train, fit=True, dtlPreprocessing=True)
X_valid_prepared, _, _ = runPreprocessing(X_valid, state=state)
model = dtl(X_train_prepared, y_train, max_depth=14, min_samples_leaf=2,
            min_impurity_decrease=0.0001)
prediction = model.predict(X_valid_prepared)
probabilities = model.predict_proba(X_valid_prepared)
saveModel(model, "output/models/dtl.pkl")
restored = loadModel("output/models/dtl.pkl")
```

Workflow memanggil `trainDTL()` dan menyimpan model melalui `saveDTLModel()`.
Prediksi dan urutan fitur mengikuti `workflows/predict.py`.
Pembanding `get_baseline("dtl")` memakai
`DecisionTreeClassifier(criterion="entropy", random_state=0)`.

Untuk preprocessing khusus DTL, gunakan `--dtl-preprocessing` pada perintah
`train` atau `run`, atau `runPreprocessing(..., fit=True, dtlPreprocessing=True)`.
Profil ini mempertahankan penanda nilai kosong, memberi kategori kosong kode
tersendiri, dan menghitung tanggal pemesanan dari tanggal kedatangan dikurangi
lead time. Fitur numerik tidak distandardisasi. Profil disimpan dalam state
preprocessing dan otomatis dipakai kembali saat prediksi.

Stratified 10-fold CV (seed 42, preprocessing di-fit pada training tiap fold)
memberi rata-rata Macro F1 0.8352 untuk profil DTL dengan depth 14, leaf 2, dan
`min_impurity_decrease=0.0001`, dibanding baseline 0.8284. Ini hasil validasi
lokal; skor Kaggle dihitung terpisah pada data test.
