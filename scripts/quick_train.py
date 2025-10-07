import os, dill, pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

ART = Path("artifacts")
TRAIN = ART / "train.csv"
MODEL = ART / "model.pkl"
PREP  = ART / "preprocessor.pkl"

def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        dill.dump(obj, f)

if not TRAIN.exists():
    raise FileNotFoundError(f"Missing {TRAIN}. Make sure artifacts/train.csv exists.")

df = pd.read_csv(TRAIN)

# Try to guess the target column; override by setting TARGET env var if needed.
guesses = [
    os.getenv("TARGET", "").strip(),
    "target", "label", "Target", "Label",
    "math score","math_score","mathscore",
    "reading score","reading_score","writingscore","writing score"
]
target = next((c for c in guesses if c and c in df.columns), None) or df.columns[-1]
if target not in df.columns:
    raise ValueError(f"Could not find target column. Available: {list(df.columns)}")

X = df.drop(columns=[target])
y = df[target]

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), num_cols),  # with_mean=False plays nicer with sparse
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0
)

# Pick classifier vs regressor based on y type/uniques
is_classification = (not pd.api.types.is_float_dtype(y)) or (y.nunique() <= max(2, int(len(y)*0.02)))

model = (
    RandomForestClassifier(n_estimators=300, random_state=42)
    if is_classification else
    RandomForestRegressor(n_estimators=250, random_state=42)
)

pipe = Pipeline([("prep", preprocessor), ("model", model)])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

pipe.fit(X_tr, y_tr)

# Save separate artifacts so your predict_pipeline can load them individually
save(preprocessor, PREP)
save(pipe.named_steps["model"], MODEL)

print("\n=== TRAIN COMPLETE ===")
print(f"Target column: {target}")
print(f"Saved: {MODEL}")
print(f"Saved: {PREP}")
