def visualizeBonus(model, featureNames, outputPath):
    from .graph import draw_graph

    return draw_graph(model, featureNames, outputPath)


__all__ = ["visualizeBonus"]
