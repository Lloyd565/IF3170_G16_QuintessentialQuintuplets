#referensi yang digunakan : https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FFMpegWriter.html

from numbers import Integral
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np

from ....core.config import MEDIA_DIR
from ..logres import LogisticRegression, logLoss, validateFeatures, validateTarget


def validateHistory(model):
    if not isinstance(model, LogisticRegression):
        raise TypeError("model harus berupa objek LogisticRegression yang sudah dilatih.")
    if any(getattr(model, name, None) is None for name in (
        "parameterHistory", "trainingFeatures", "trainingTarget",
    )):
        raise ValueError("Riwayat parameter belum tersedia; latih model dengan recordHistory=True.")
    features = validateFeatures(model.trainingFeatures)
    target = validateTarget(model.trainingTarget, len(features))
    try:
        history = np.asarray(model.parameterHistory, dtype=float)
        objective = np.asarray(model.lossHistory, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Riwayat pelatihan harus berisi nilai numerik terhingga.") from error
    if history.ndim != 2 or history.shape[1] != features.shape[1] + 1 or len(history) < 2:
        raise ValueError("Riwayat parameter harus memuat bias dan semua bobot dari awal hingga akhir pelatihan.")
    if objective.shape != (len(history),) or not np.isfinite(history).all() or not np.isfinite(objective).all():
        raise ValueError("Riwayat parameter dan loss harus terhingga dan memiliki jumlah epoch yang sama.")
    if not np.allclose(history[-1], np.concatenate(([model.bias], model.weights))):
        raise ValueError("Parameter terakhir pada riwayat harus sesuai dengan model terlatih.")
    return features, target, history, objective


def lossSurface(features, target, parameters, biasGrid, weightGrid, featureIndex):
    selected = features[:, featureIndex]
    offset = features @ parameters[1:] - selected * parameters[featureIndex + 1]
    biases, weights = biasGrid.ravel(), weightGrid.ravel()
    losses = np.zeros(len(biases), dtype=float)
    for start in range(0, len(biases), 64):
        stop = min(start + 64, len(biases))
        for row in range(0, len(features), 2048):
            end = min(row + 2048, len(features))
            scores = (
                offset[row:end, None]
                + selected[row:end, None] * weights[None, start:stop]
                + biases[None, start:stop]
            )
            scores *= (1.0 - 2.0 * target[row:end, None])
            np.logaddexp(0.0, scores, out=scores)
            losses[start:stop] += scores.sum(axis=0)
    return (losses / len(features)).reshape(biasGrid.shape)


def parameterBounds(values):
    minimum, maximum = float(np.min(values)), float(np.max(values))
    margin = max(0.15 * (maximum - minimum), 0.1)
    return minimum - margin, maximum + margin


def validateVideoOptions(outputPath, featureIndex, featureCount, fps, maxFrames, gridSize, dpi):
    try:
        path = Path(outputPath)
    except TypeError as error:
        raise ValueError("Lokasi video harus berupa nama berkas MP4.") from error
    if path.suffix.lower() != ".mp4":
        raise ValueError("Video harus disimpan dalam berkas dengan ekstensi .mp4.")
    if isinstance(featureIndex, bool) or not isinstance(featureIndex, Integral) or not 0 <= featureIndex < featureCount:
        raise ValueError("featureIndex harus berupa indeks kolom yang tersedia pada model.")
    for name, value, minimum in (
        ("fps", fps, 1), ("maxFrames", maxFrames, 2), ("gridSize", gridSize, 8), ("dpi", dpi, 40),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} harus berupa bilangan bulat minimal {minimum}.")
    return path


def checkVideoWriter():
    from matplotlib.animation import writers

    if not writers.is_available("ffmpeg"):
        raise ValueError("FFmpeg tidak tersedia; pasang FFmpeg agar video MP4 dapat dibuat.")


def visualizeBonus(
    model,
    outputPath=MEDIA_DIR / "logistic_regression_training.mp4",
    *,
    featureIndex=0,
    fps=12,
    maxFrames=80,
    gridSize=40,
    dpi=120,
):
    features, target, history, objective = validateHistory(model)
    path = validateVideoOptions(outputPath, featureIndex, features.shape[1], fps, maxFrames, gridSize, dpi)
    checkVideoWriter()

    from matplotlib.animation import FFMpegWriter
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    losses = np.asarray([
        logLoss(features @ parameters[1:] + parameters[0], target, parameters[1:])
        for parameters in history
    ])
    biasPath, weightPath = history[:, 0], history[:, featureIndex + 1]
    biasBounds, weightBounds = parameterBounds(biasPath), parameterBounds(weightPath)
    biasGrid, weightGrid = np.meshgrid(
        np.linspace(*biasBounds, gridSize), np.linspace(*weightBounds, gridSize),
    )
    epochs = np.unique(np.linspace(0, len(history) - 1, min(len(history), maxFrames), dtype=int))
    surfaces = []
    for epoch in epochs:
        if features.shape[1] == 1 and surfaces:
            surfaces.append(surfaces[0])
        else:
            surfaces.append(lossSurface(features, target, history[epoch], biasGrid, weightGrid, featureIndex))
    if not np.isfinite(losses).all() or not all(np.isfinite(surface).all() for surface in surfaces):
        raise ValueError("Perhitungan kontur log-loss menghasilkan nilai tidak terhingga.")
    minimum = min(float(surface.min()) for surface in surfaces)
    maximum = max(float(surface.max()) for surface in surfaces)
    if maximum <= minimum:
        maximum = minimum + 1e-6
    levels = np.linspace(minimum, maximum, 18)
    featureNames = getattr(model, "featureNames", None)
    featureName = str(featureNames[featureIndex]) if featureNames is not None else f"feature {featureIndex}"
    figure = Figure(figsize=(12, 6), facecolor="#f8fafc")
    FigureCanvasAgg(figure)
    contourAxis, lossAxis = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.25, 1]})
    figure.subplots_adjust(left=0.075, right=0.97, bottom=0.18, top=0.82, wspace=0.38)
    figure.suptitle("Logistic regression: loss contours and parameter trajectory", fontsize=16, y=0.97)
    sliceLabel = (
        "Exact two-parameter loss surface"
        if features.shape[1] == 1 else
        "2D loss slice: other weights fixed at the current epoch; the path is projected"
    )
    figure.text(0.5, 0.895, sliceLabel, ha="center", fontsize=10, color="#475569")
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(minimum, maximum), cmap="viridis"), ax=contourAxis, pad=0.025,
    )
    colorbar.set_label("Log-loss", fontsize=10)
    hasPenalty = not np.allclose(losses, objective, rtol=1e-8, atol=1e-10)
    lossBounds = parameterBounds(np.concatenate((losses, objective)))
    writer = FFMpegWriter(
        fps=fps, codec="libx264",
        metadata={"title": "Logistic regression training", "artist": "IF3170"},
        extra_args=["-pix_fmt", "yuv420p", "-threads", "1", "-movflags", "+faststart"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=".mp4", delete=False) as targetFile:
            temporary = Path(targetFile.name)
        with writer.saving(figure, str(temporary), dpi=dpi):
            for epoch, surface in zip(epochs, surfaces):
                contourAxis.clear()
                lossAxis.clear()
                contourAxis.contourf(biasGrid, weightGrid, surface, levels=levels, cmap="viridis")
                if surface.max() > surface.min():
                    lineLevels = np.linspace(surface.min(), surface.max(), 10)[1:-1]
                    lines = contourAxis.contour(
                        biasGrid, weightGrid, surface, levels=lineLevels, colors="white", linewidths=0.55, alpha=0.65,
                    )
                    contourAxis.clabel(lines, inline=True, fmt="%.3f", fontsize=7)
                contourAxis.plot(biasPath[:epoch + 1], weightPath[:epoch + 1], color="#fb923c", linewidth=2, label="Parameter path")
                contourAxis.scatter([biasPath[0]], [weightPath[0]], c="white", edgecolors="#0f172a", s=42, zorder=5, label="Start")
                contourAxis.scatter([biasPath[epoch]], [weightPath[epoch]], c="#f43f5e", edgecolors="white", s=65, zorder=6, label="Current")
                contourAxis.set(xlim=biasBounds, ylim=weightBounds, xlabel=r"$\theta_0$ (bias)")
                contourAxis.set_ylabel(r"$\theta_{%d}$ (%s)" % (featureIndex + 1, featureName), fontsize=10)
                contourAxis.set_title(f"Epoch {epoch} / {len(history) - 1}", fontsize=12)
                contourAxis.legend(loc="best", fontsize=8)
                lossAxis.plot(np.arange(epoch + 1), losses[:epoch + 1], color="#2563eb", linewidth=2, label="Log-loss")
                if hasPenalty:
                    lossAxis.plot(np.arange(epoch + 1), objective[:epoch + 1], color="#9333ea", linestyle="--", label="Regularized objective")
                lossAxis.scatter([epoch], [losses[epoch]], c="#f43f5e", s=35, zorder=5)
                lossAxis.set(xlim=(0, len(history) - 1), ylim=lossBounds, xlabel="Epoch", ylabel="Loss")
                lossAxis.set_title(f"Training log-loss: {losses[epoch]:.5f}", fontsize=12)
                lossAxis.grid(alpha=0.2)
                lossAxis.legend(fontsize=8)
                footer = figure.text(
                    0.5, 0.045,
                    f"{len(features):,} training rows | bias = {biasPath[epoch]:.4f} | "
                    f"weight = {weightPath[epoch]:.4f} | checkpoints recorded after each epoch",
                    ha="center", fontsize=9, color="#475569",
                )
                writer.grab_frame()
                footer.remove()
        temporary.replace(path)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise ValueError("Video tidak dapat dibuat; periksa lokasi output dan pemasangan FFmpeg.") from error
    finally:
        figure.clear()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
