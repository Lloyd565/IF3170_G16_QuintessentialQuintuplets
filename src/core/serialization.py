"""Placeholders for persisting models and fitted preprocessing artifacts."""


def save_model(*args, **kwargs):
    """Save a model and related artifacts using a future agreed format."""
    raise NotImplementedError


def load_model(*args, **kwargs):
    """Load a model and related artifacts using a future agreed format."""
    raise NotImplementedError
