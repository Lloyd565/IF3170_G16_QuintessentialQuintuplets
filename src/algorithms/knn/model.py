from .distance import compute_distance


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        if len(X_train) != len(y_train):
            raise ValueError("Jumlah baris harus sama dengan jumlah label.")

        if not 1 <= self.k <= len(X_train):
            raise ValueError("k harus antara 1 dan jumlah baris latih.")

        if any(label not in (0, 1) for label in y_train):
            raise ValueError("Label KNN biner harus 0 atau 1.")

        self.X_train = X_train
        self.y_train = y_train
        return self

    def _predict_one(self, row):
        distances = []
        for i, train_row in enumerate(self.X_train):
            distance = compute_distance(row, train_row)
            distances.append((distance, i))

        distances.sort()
        k_hasil = distances[:self.k]
        labels = []
        for _, indeks in k_hasil:
            labels.append(self.y_train[indeks])
        suara_satu = labels.count(1)
        if suara_satu > len(labels) / 2:
            return 1
        return 0

    def predict(self, X):
        hasil = []
        for row in X:
            hasil.append(self._predict_one(row))
        return hasil

    def predict_proba(self, X):
        hasil = []

        for row in X:
            distances = []
            for i, train_row in enumerate(self.X_train):
                distance = compute_distance(row, train_row)
                distances.append((distance, i))
            distances.sort()
            k_hasil = distances[:self.k]
            labels = []
            for _, indeks in k_hasil:
                labels.append(self.y_train[indeks])
            p1 = labels.count(1) / len(labels)
            p0 = 1 - p1
            hasil.append([p0, p1])

        return hasil