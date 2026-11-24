from numbers import Integral
from pathlib import Path
from textwrap import fill

import numpy as np
from matplotlib.figure import Figure

from ..criteria import entropy


def draw_tree(model, feature_names=None, save_path=None, top_n=10):
    """Draw at most top_n split nodes, breadth-first; None displays all splits."""
    if model.tree_ is None:
        raise RuntimeError("Latih model dengan fit() sebelum menggambar tree.")
    if top_n is not None and (
        isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n < 1
    ):
        raise ValueError("top_n harus berupa integer >= 1 atau None.")
    if feature_names is None:
        feature_names = [f"x[{i}]" for i in range(model.n_features_in_)]
    if isinstance(feature_names, str) or len(feature_names) != model.n_features_in_:
        raise ValueError("Jumlah nama fitur harus sama dengan jumlah fitur model.")
    feature_names = [str(name) for name in feature_names]

    nodes = [(model.tree_, 0)]
    children = {}
    for index, (node, depth) in enumerate(nodes):
        if node.feature_index is None or (top_n is not None and len(children) >= top_n):
            continue
        children[index] = (len(nodes), len(nodes) + 1)
        nodes.extend(((node.left, depth + 1), (node.right, depth + 1)))

    positions = {}
    order = []
    stack = [0]
    leaf_count = 0
    while stack:
        index = stack.pop()
        order.append(index)
        if index in children:
            stack.extend(reversed(children[index]))
        else:
            positions[index] = leaf_count
            leaf_count += 1
    for index in reversed(order):
        if index in children:
            left, right = children[index]
            positions[index] = (positions[left] + positions[right]) / 2

    depth = max(level for _, level in nodes)
    figure = Figure(figsize=(max(7, 2.8 * leaf_count), max(3, 1.5 * (depth + 1))))
    ax = figure.subplots()
    labels = []
    for index, (node, level) in enumerate(nodes):
        hidden = node.feature_index is not None and index not in children
        if index in children:
            condition = f"{feature_names[node.feature_index]} <= {node.split_value:.6g}"
            title = fill(condition, width=30)
        elif hidden:
            title = "Subtree hidden"
        else:
            prediction = model.classes_[np.argmax(node.class_counts)]
            title = f"Leaf: is_canceled = {prediction}"
        counts = node.class_counts.tolist()
        label = (
            f"{title}\nentropy = {float(entropy(node.class_counts)):.3f}"
            f"\nsamples = {int(node.class_counts.sum())}\ncounts = {counts}"
        )
        color = "#dbeafe" if index in children else "#fee2e2" if np.argmax(counts) else "#dcfce7"
        labels.append(ax.text(
            positions[index], -level, label, ha="center", va="center", fontsize=9,
            parse_math=False, usetex=False, zorder=3,
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="#f3f4f6" if hidden else color,
                edgecolor="#6b7280", linestyle="--" if hidden else "-",
            ),
        ))
    for parent, branches in children.items():
        for child, relation in zip(branches, ("True", "False")):
            start = (positions[parent], -nodes[parent][1])
            end = (positions[child], -nodes[child][1])
            ax.annotate(
                "", xy=end, xytext=start, zorder=4,
                arrowprops=dict(
                    arrowstyle="->", color="#6b7280", shrinkA=3, shrinkB=3,
                    patchA=labels[parent].get_bbox_patch(), patchB=labels[child].get_bbox_patch(),
                ),
            )
            ax.text(
                (start[0] + end[0]) / 2, (start[1] + end[1]) / 2, relation,
                ha="center", va="center", fontsize=8, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1),
                parse_math=False, usetex=False,
            )
    ax.set_xlim(-0.6, leaf_count - 0.4)
    ax.set_ylim(-depth - 0.5, 0.5)
    ax.axis("off")
    limit = "all" if top_n is None else top_n
    ax.set_title(f"DTL: {len(children)} splits shown (top_n={limit})", pad=20, usetex=False)
    figure.text(
        0.5, 0.01,
        "True: <= threshold | False: > threshold | counts = [not canceled (0), canceled (1)]\n"
        "Dashed boxes contain hidden subtrees.",
        ha="center", fontsize=9, parse_math=False, usetex=False,
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return figure
