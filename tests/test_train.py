import joblib, types, json, pathlib, sklearn
from src.utils import load_config
from src.train import LogisticRegression, load_digits

def test_config_schema():
    cfg = load_config()
    assert isinstance(cfg["C"], float)
    assert isinstance(cfg["solver"], str)
    assert isinstance(cfg["max_iter"], int)

def test_model_instance():
    X, y = load_digits(return_X_y=True)
    cfg = load_config()
    model = LogisticRegression(
        C=cfg["C"], solver=cfg["solver"], max_iter=cfg["max_iter"]
    ).fit(X, y)
    assert isinstance(model, LogisticRegression)

def test_accuracy_threshold():
    X, y = load_digits(return_X_y=True)
    cfg = load_config()
    model = LogisticRegression(
        C=cfg["C"], solver=cfg["solver"], max_iter=cfg["max_iter"]
    ).fit(X, y)
    acc = model.score(X, y)
    assert acc > 0.92   # conservative threshold
