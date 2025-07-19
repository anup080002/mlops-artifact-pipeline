import joblib, pathlib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import load_config

CFG = load_config()
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=CFG["test_size"], random_state=CFG["random_state"]
)
model = LogisticRegression(
    C=CFG["C"], solver=CFG["solver"], max_iter=CFG["max_iter"], n_jobs=-1
).fit(X_tr, y_tr)

acc = accuracy_score(y_te, model.predict(X_te))
print(f"Validation accuracy: {acc:.4f}")

out = pathlib.Path("artifacts")
out.mkdir(exist_ok=True)
joblib.dump(model, out / "model_train.pkl")
