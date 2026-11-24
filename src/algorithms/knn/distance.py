import math


def compute_distance(a, b):
    """Compute the Euclidean distance between two feature rows."""
    if len(a) != len(b):
        raise ValueError("Jumlah fitur kedua baris harus sama.")

    total = 0
    for i in range(len(a)):
        selisih = a[i] - b[i]
        total += selisih * selisih
    return math.sqrt(total)

