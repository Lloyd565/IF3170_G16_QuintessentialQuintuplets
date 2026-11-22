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

model = dtl(X_train_prepared, y_train, max_depth=12, min_samples_leaf=5)
prediction = model.predict(X_valid_prepared)
probabilities = model.predict_proba(X_valid_prepared)
saveModel(model, "output/models/dtl.pkl")
restored = loadModel("output/models/dtl.pkl")
```

Workflow memanggil `trainDTL()` dan menyimpan model melalui `saveDTLModel()`.
Prediksi dan urutan fitur mengikuti `workflows/predict.py`.
Pembanding `get_baseline("dtl")` memakai
`DecisionTreeClassifier(criterion="entropy", random_state=0)`.
