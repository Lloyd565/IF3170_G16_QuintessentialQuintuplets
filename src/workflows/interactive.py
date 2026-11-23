import json
import math

from ..core.config import DATA_DIR, MEDIA_DIR
from ..core.registry import getModelNames


def askChoice(message, options, default, *, multiple=False):
    print(f"\n{message}")
    for index, (_, label) in enumerate(options, start=1):
        print(f"{index}. {label}")
    defaultIndex = next(index for index, (value, _) in enumerate(options, start=1) if value == default)
    example = "1 atau 1, 3" if multiple else "1"
    while True:
        answer = input(f"Pilih nomor (cth. {example}; Enter = {defaultIndex}): ").strip()
        if not answer:
            return [default] if multiple else default
        try:
            indices = [int(value) for value in answer.replace(",", " ").split()]
        except ValueError:
            print(f"Pilihan tidak valid. Masukkan nomor 1 sampai {len(options)}.")
            continue
        if not indices or any(index < 1 or index > len(options) for index in indices):
            print(f"Pilihan tidak valid. Masukkan nomor 1 sampai {len(options)}.")
            continue
        if not multiple and len(indices) != 1:
            print("Pilih satu nomor saja.")
            continue
        if len(set(indices)) != len(indices):
            print("Nomor pilihan tidak boleh berulang.")
            continue
        values = [options[index - 1][0] for index in indices]
        return values if multiple else values[0]


def askBoolean(message, default):
    return askChoice(message, [(True, "Ya"), (False, "Tidak")], default)


def askNumber(
    message, examples, default, *, integer=False, minimum=None, maximum=None,
    exclusiveMinimum=False, exclusiveMaximum=False, emptyLabel="lewati",
):
    defaultLabel = str(default) if default is not None else emptyLabel
    kind = "bilangan bulat" if integer else "bilangan riil"
    limits = []
    if minimum is not None:
        limits.append(f"{'>' if exclusiveMinimum else '>='} {minimum}")
    if maximum is not None:
        limits.append(f"{'<' if exclusiveMaximum else '<='} {maximum}")
    requirement = f"Masukkan {kind} terhingga" + (" dengan nilai " + " dan ".join(limits) if limits else "") + "."
    while True:
        answer = input(f"{message} (cth. {examples}) [Enter = {defaultLabel}]: ").strip()
        if not answer:
            return default
        try:
            value = int(answer) if integer else float(answer)
            valid = math.isfinite(value)
        except (ValueError, OverflowError):
            print(requirement)
            continue
        if minimum is not None:
            valid = valid and (value > minimum if exclusiveMinimum else value >= minimum)
        if maximum is not None:
            valid = valid and (value < maximum if exclusiveMaximum else value <= maximum)
        if not valid:
            print(requirement)
            continue
        return value


def askText(message, example, default="", *, required=False):
    defaultLabel = str(default) if default else "wajib diisi" if required else "lewati"
    while True:
        answer = input(f"{message} (cth. {example}) [Enter = {defaultLabel}]: ").strip()
        if answer:
            return answer
        if default or not required:
            return str(default)
        print("Isian ini wajib diisi.")


def askModels():
    labels = {
        "knn": "KNN", "dtl": "Decision Tree (DTL)",
        "logistic_regression": "Logistic Regression", "svm": "SVM",
        "naive_bayes": "Naive Bayes",
    }
    options = [(name, labels.get(name, name)) for name in getModelNames()]
    options.append(("all", "Semua model"))
    while True:
        names = askChoice("Pilih model:", options, "all", multiple=True)
        if "all" in names and len(names) > 1:
            print("Pilihan semua model tidak boleh digabung dengan model lain.")
            continue
        return names


def askParameters(message, names, *, baseline=False):
    examples = {
        "knn": {"k": 5}, "dtl": {"max_depth": 10},
        "logistic_regression": {"epochs": 200}, "svm": {"C": 2},
        "naive_bayes": {"smoothing": 1e-8},
    }
    if baseline:
        examples.update({
            "knn": {"n_neighbors": 5}, "logistic_regression": {"max_iter": 200},
            "naive_bayes": {"var_smoothing": 1e-8},
        })
    example = json.dumps({names[0]: examples.get(names[0], {})})
    while True:
        answer = askText(message, example, "{}")
        try:
            parameters = json.loads(answer)
        except json.JSONDecodeError:
            print("Parameter harus berupa JSON yang valid.")
            continue
        if not isinstance(parameters, dict) or any(
            name not in names or not isinstance(values, dict)
            for name, values in parameters.items()
        ):
            print("Parameter harus memetakan model yang dipilih ke objek parameter.")
            continue
        return parameters


def askModelParameters(names):
    parameters = {}
    if "knn" in names:
        print("\nParameter KNN")
        parameters["knn"] = {
            "k": askNumber("Parameter k untuk KNN", "1, 2, 3", 3, integer=True, minimum=1),
        }
    if "dtl" in names:
        print("\nParameter Decision Tree (DTL)")
        parameters["dtl"] = {
            "max_depth": askNumber("Kedalaman maksimum pohon", "5, 10, 20", None, integer=True, minimum=0, emptyLabel="tanpa batas"),
            "min_samples_split": askNumber("Minimum sampel untuk percabangan", "2, 5, 10", 2, integer=True, minimum=2),
            "min_samples_leaf": askNumber("Minimum sampel pada daun", "1, 2, 5", 1, integer=True, minimum=1),
            "min_impurity_decrease": askNumber("Minimum penurunan impurity", "0, 0.001, 0.01", 0.0, minimum=0),
        }
    if "logistic_regression" in names:
        print("\nParameter Logistic Regression")
        options = {
            "epochs": askNumber("Jumlah epoch", "50, 100, 500", 100, integer=True, minimum=1),
            "learningRate": askNumber("Learning rate", "0.001, 0.01, 0.1", 0.01, minimum=0, exclusiveMinimum=True),
        }
        options["regularization"] = askChoice("Pilih regularisasi:", [
            ("none", "Tanpa regularisasi"), ("l1", "L1"), ("l2", "L2"), ("elasticnet", "Elastic Net"),
        ], "none")
        if options["regularization"] != "none":
            options["regularizationStrength"] = askNumber("Kekuatan regularisasi", "0.001, 0.01, 0.1", 0.01, minimum=0)
        if options["regularization"] == "elasticnet":
            options["l1Ratio"] = askNumber("Rasio L1 untuk Elastic Net", "0.25, 0.5, 0.75", 0.5, minimum=0, maximum=1)
        options["shuffle"] = askBoolean("Acak urutan data setiap epoch?", True)
        parameters["logistic_regression"] = options
    if "svm" in names:
        print("\nParameter SVM")
        options = {
            "kernel": askChoice("Pilih kernel SVM:", [
                ("linear", "Linear"), ("poly", "Polynomial"), ("rbf", "Gaussian (RBF)"),
            ], "rbf"),
            "C": askNumber("Parameter C untuk SVM", "0.1, 1, 10", 1.0, minimum=0, exclusiveMinimum=True),
        }
        if options["kernel"] == "poly":
            options["degree"] = askNumber("Derajat polynomial", "2, 3, 4", 3, integer=True, minimum=1)
            options["coef0"] = askNumber("Konstanta coef0", "0, 0.5, 1", 0.0)
        if options["kernel"] in ("poly", "rbf"):
            gamma = askChoice("Pilih pengaturan gamma:", [("scale", "Otomatis (scale)"), ("manual", "Isi nilai sendiri")], "scale")
            options["gamma"] = "scale" if gamma == "scale" else askNumber("Nilai gamma", "0.01, 0.1, 1", 0.1, minimum=0, exclusiveMinimum=True)
        options["tol"] = askNumber("Toleransi optimasi SVM", "0.0001, 0.001, 0.01", 0.001, minimum=0, exclusiveMinimum=True)
        options["max_passes"] = askNumber("Maksimum putaran tanpa perubahan", "5, 10, 20", 5, integer=True, minimum=1)
        options["max_iter"] = askNumber("Maksimum iterasi SVM", "10000, 100000, 500000", 100000, integer=True, minimum=1)
        options["class_weight"] = askChoice("Pilih bobot kelas SVM:", [(None, "Tanpa pembobotan"), ("balanced", "Seimbang (balanced)")], None)
        parameters["svm"] = options
    if "naive_bayes" in names:
        print("\nParameter Naive Bayes")
        parameters["naive_bayes"] = {
            "smoothing": askNumber("Smoothing untuk Naive Bayes", "1e-9, 1e-8, 1e-6", 1e-9, minimum=0),
            "pick_features": askBoolean("Gunakan seleksi fitur khusus Naive Bayes?", False),
        }
    if askBoolean("Tambahkan parameter lanjutan melalui JSON?", False):
        for name, values in askParameters("Parameter tambahan model", names).items():
            parameters.setdefault(name, {}).update(values)
    return parameters


def interactiveArguments():
    print("Tekan Enter untuk menggunakan nilai bawaan.")
    command = askChoice("Pilih perintah:", [
        ("run", "Latih model dan buat prediksi (run)"),
        ("train", "Latih model (train)"),
        ("predict", "Prediksi dengan model tersimpan (predict)"),
        ("evaluate", "Evaluasi model tersimpan (evaluate)"),
        ("compare", "Bandingkan dengan scikit-learn (compare)"),
    ], "run")
    training = command in ("train", "run", "compare")
    defaultInput = "" if command == "evaluate" else DATA_DIR / ("train.csv" if training else "test.csv")
    path = askText(
        "Lokasi CSV berlabel untuk evaluasi" if command == "evaluate" else "Lokasi CSV input",
        "data/validasi.csv" if command == "evaluate" else "data/train.csv" if training else "data/test.csv",
        defaultInput, required=True,
    )
    names = askModels()
    selected = list(getModelNames()) if "all" in names else names
    arguments = [command, "--input", path, "--model", *names]
    if training:
        if not askBoolean("Gunakan cleaning?", True):
            arguments.append("--no-cleaning")
        if not askBoolean("Gunakan transformation?", True):
            arguments.append("--no-transformation")
        count = askNumber("Jumlah fitur mRMR", "5, 10, 16", None, integer=True, minimum=1, emptyLabel="semua fitur")
        if count is not None:
            arguments.extend(("--features", str(count)))
        pca = askChoice("Pilih pengaturan PCA:", [
            ("none", "Tanpa PCA"), ("count", "Tentukan jumlah komponen"), ("variance", "Tentukan target variansi kumulatif"),
        ], "none")
        if pca == "count":
            components = askNumber("Jumlah komponen PCA", "2, 5, 10", 2, integer=True, minimum=1)
            arguments.extend(("--components", str(components)))
        elif pca == "variance":
            components = askNumber("Target variansi PCA", "0.9, 0.95, 0.99", 0.95, minimum=0, maximum=1, exclusiveMinimum=True, exclusiveMaximum=True)
            arguments.extend(("--components", str(components)))
        if askBoolean("Gunakan oversampling?", False):
            arguments.append("--oversample")
        validationLabel = "Proporsi validasi untuk perbandingan" if command == "compare" else "Proporsi validasi; 0 berarti tanpa validasi"
        validationSize = askNumber(
            validationLabel, "0.1, 0.2, 0.3", 0.2, minimum=0, maximum=1,
            exclusiveMinimum=command == "compare", exclusiveMaximum=True,
        )
        arguments.extend(("--validation-size", str(validationSize)))
        seed = askNumber("Seed acak untuk pembagian data dan training", "0, 42, 123", 42, integer=True, minimum=0)
        arguments.extend(("--random-state", str(seed)))
        parameters = askModelParameters(selected)
        if parameters:
            arguments.extend(("--model-params", json.dumps(parameters)))
    else:
        if askBoolean("Data sudah dipreproses sesuai model?", False):
            arguments.append("--prepared")
    costSensitive = any(name != "svm" for name in selected) and askBoolean(
        "Gunakan cost-sensitive (biaya FP dua kali FN; ambang 2/3 untuk semua model kecuali SVM)?", False,
    )
    if costSensitive:
        arguments.append("--cost-sensitive")
    if "logistic_regression" in selected and not costSensitive:
        threshold = askNumber("Ambang prediksi Logistic Regression", "0.4, 0.5, 0.6", 0.5, minimum=0, maximum=1)
        if threshold != 0.5:
            arguments.extend(("--threshold", str(threshold)))
    if training:
        compareModels = command == "compare" or validationSize > 0 and askBoolean("Bandingkan dengan scikit-learn?", False)
        if compareModels:
            arguments.append("--compare")
            if askBoolean("Ubah parameter acuan scikit-learn melalui JSON?", False):
                baseline = askParameters("Parameter acuan scikit-learn", selected, baseline=True)
                arguments.extend(("--baseline-params", json.dumps(baseline)))
        if "dtl" in selected and askBoolean("Buat gambar bonus Decision Tree (DTL)?", False):
            arguments.append("--dtl-bonus")
            topN = askNumber("Jumlah maksimum node percabangan pada gambar DTL", "5, 10, 20", 10, integer=True, minimum=1)
            arguments.extend(("--dtl-bonus-top-n", str(topN)))
            output = askText("Lokasi gambar bonus DTL", "output/media/dtl_tree.png", MEDIA_DIR / "dtl_tree.png")
            arguments.extend(("--dtl-bonus-output", output))
        if "logistic_regression" in selected and askBoolean("Buat video bonus Logistic Regression?", False):
            arguments.append("--logres-bonus")
            feature = askText("Nama atau indeks fitur untuk lintasan parameter", "0, 1, days_before_arrival", "0")
            arguments.extend(("--bonus-feature", feature))
            output = askText("Lokasi video bonus Logistic Regression", "output/media/training.mp4", MEDIA_DIR / "logistic_regression_training.mp4")
            arguments.extend(("--bonus-output", output))
        if "naive_bayes" in selected and askBoolean("Buat graf bonus Naive Bayes?", False):
            arguments.append("--naive-bayes-bonus")
            output = askText("Lokasi gambar bonus Naive Bayes", "output/media/naive_bayes_graph.png", MEDIA_DIR / "naive_bayes_graph.png")
            arguments.extend(("--naive-bayes-bonus-output", output))
        if "svm" in selected and askBoolean("Buat gambar bonus SVM (bidang pemisah tiap kernel)?", False):
            arguments.append("--svm-bonus")
            sample = askNumber("Jumlah sampel untuk gambar SVM", "200, 300, 500", 300, integer=True, minimum=2)
            arguments.extend(("--svm-bonus-sample", str(sample)))
            output = askText("Lokasi gambar bonus SVM", "output/media/svm_boundaries.png", MEDIA_DIR / "svm_boundaries.png")
            arguments.extend(("--svm-bonus-output", output))
    if command == "run":
        path = askText("Lokasi CSV test", "data/test.csv", DATA_DIR / "test.csv")
        arguments.extend(("--test-input", path))
    if command in ("run", "predict"):
        template = askText("Lokasi sample submission", "data/sample_submission.csv")
        if template:
            arguments.extend(("--template", template))
    return arguments
