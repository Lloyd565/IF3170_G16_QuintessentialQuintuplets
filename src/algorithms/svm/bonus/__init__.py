def visualizeBonus(features, target, outputPath, **svmParameters):
    from .boundary import plot_boundaries

    return plot_boundaries(features, target, outputPath, **svmParameters)


__all__ = ["visualizeBonus"]
