# IF3170_G16_QuintessentialQuintuplets

Tugas Besar 2 IF3170 Inteligensi Artifisial: implementasi KNN, Decision Tree Learning (DTL), Logistic Regression, SVM, dan Naive Bayes dari awal untuk memprediksi pembatalan pemesanan penginapan. Program mencakup pengolahan data, evaluasi, dan perbandingan dengan scikit-learn.

## Persiapan

Gunakan Python 3.10 atau lebih baru. Jalankan perintah berikut dari direktori utama repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Letakkan `train.csv`, `test.csv`, dan `sample_submission.csv` di folder `data/`.

## Menjalankan Program

Mode interaktif:

```bash
python -m src.main --interactive
```

Ganti placeholder `<...>` dengan nilai yang sesuai.

Pelatihan, validasi, dan perbandingan dengan scikit-learn:

```text
python -m src.main compare --model <model> --input <data_latih.csv>
```

Prediksi menggunakan model tersimpan:

```text
python -m src.main predict --model <model> --model-dir <folder_model> --input <data_uji.csv> --template <sample_submission.csv>
```

Contoh penggunaan dengan KNN:

```bash
python -m src.main compare --model knn --input data/train.csv --k 5
python -m src.main predict --model knn --model-dir output/models --input data/test.csv --template data/sample_submission.csv
```

Pilihan model: `knn`, `dtl`, `logistic_regression`, `svm`, dan `naive_bayes`. Hasil program disimpan di folder `output/`. Gunakan `python -m src.main --help` untuk melihat perintah yang tersedia.

## Pembagian Tugas

| Nama | NIM | Tugas |
| --- | --- | --- |
| Audric Yusuf Maynard Simatupang | 13524010 | Logistic Regression, bonus Logistic Regression, integrasi, seleksi fitur |
| Mahatma Brahmana | 13524015 | Naive Bayes, optimasi Naive Bayes, bonus graf Naive Bayes |
| Ishak Palentino Napitupulu | 13524022 | — |
| Nathanael Gunawan | 13524066 | K-Nearest Neighbors, bonus cost-sensitive, transformasi data |
| Richard Samuel Simanullang | 13524112 | Reduksi dimensi, SVM, bonus SVM |
