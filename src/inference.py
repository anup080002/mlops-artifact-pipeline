import joblib, json
from sklearn.datasets import load_digits
from utils import load_config

CFG = load_config()
model = joblib.load("artifacts/model_train.pkl")
X, _ = load_digits(return_X_y=True)
preds = model.predict(X)
print(f"Inference done on {len(preds)} samples; example preds: {preds[:10]}")
