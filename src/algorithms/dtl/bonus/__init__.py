from .graph import draw_tree

visualize_bonus = draw_tree


def visualizeBonus(model, featureNames, outputPath, topN=10):
    return draw_tree(model, feature_names=featureNames, save_path=outputPath, top_n=topN)


__all__ = ["draw_tree", "visualize_bonus", "visualizeBonus"]
