from pathlib import Path
import pickle


def saveModel(model, path: str | Path) -> Path:
    path = Path(path)
    try:
        content = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    except (OSError, pickle.PickleError, TypeError, AttributeError) as error:
        raise ValueError(f"Model tidak dapat disimpan ke {path}.") from error
    return path


def loadModel(path: str | Path):
    path = Path(path)
    try:
        with path.open("rb") as source:
            return pickle.load(source)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Berkas model tidak ditemukan: {path}.") from error
    except (OSError, pickle.PickleError, EOFError, AttributeError, ImportError, ValueError) as error:
        raise ValueError(f"Model tidak dapat dibaca dari {path}.") from error
