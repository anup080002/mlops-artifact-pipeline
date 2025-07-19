import joblib, json
from sklearn.datasets import load_digits
try:
    from .utils import load_config      # when imported as a package (in tests)
except ImportError:
    from utils import load_config       # when run as a script


CFG = load_config()
model = joblib.load("artifacts/model_train.pkl")
X, _ = load_digits(return_X_y=True)
preds = model.predict(X)
print(f"Inference done on {len(preds)} samples; example preds: {preds[:10]}")
