import pandas as pd
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Fetch UCI Student Performance dataset (Mathematics records)
student_data = fetch_ucirepo(id=320)
X = student_data.data.features.copy()
targets = student_data.data.targets.copy()

# Keep G3 only as the target; never use it as an input feature.
y = (targets["G3"] < 10).astype(int)

# Save a copy of the dataset with G3 for dashboard metrics/reference.
df = X.copy()
df["G3"] = targets["G3"]
df.to_csv("student_data.csv", index=False)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Dataset shape:", df.shape)
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(
    y_test, predictions, target_names=["Not at risk", "At risk"]
))

joblib.dump(model, "student_model.pkl")
print("Saved student_data.csv and student_model.pkl")
