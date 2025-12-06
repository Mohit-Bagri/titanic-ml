
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib
import shap

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("data/train.csv")

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET = "Survived"

df = df[FEATURES + [TARGET]].dropna(subset=[TARGET])

X = df[FEATURES]
y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train/Test shapes:")
print(X_train.shape, X_test.shape)



numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]

numeric_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ]
)


pipeline = Pipeline(
    [
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n=== BASELINE LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

print("CV ROC-AUC:", cv_scores)
print("Mean CV ROC-AUC:", cv_scores.mean())


param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
}

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

y_pred_best = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n=== BEST LOGISTIC REGRESSION ===")
print("Best Params:", grid.best_params_)
print("ROC-AUC:", roc_auc_score(y_test, y_proba))


shap.initjs()

lr_model = best_model.named_steps["model"]
prep = best_model.named_steps["prep"]

X_test_transformed = prep.transform(X_test)

explainer = shap.LinearExplainer(lr_model, X_test_transformed)
shap_values = explainer.shap_values(X_test_transformed)

shap.summary_plot(
    shap_values,
    X_test_transformed,
    plot_type="bar",
    show=False
)

plt.tight_layout()
plt.savefig("artifacts/shap_summary_bar.png", dpi=300)
plt.close()

rf_pipeline = Pipeline(
    [
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]
)

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

print("\n=== RANDOM FOREST ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("F1:", f1_score(y_test, rf_pred))


MODEL_PATH = "api/model/titanic_best_pipeline.joblib"
META_PATH = "api/model/titanic_metadata.json"

joblib.dump(best_model, MODEL_PATH)

metadata = {
    "features": FEATURES,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "model_type": "LogisticRegression",
    "best_params": grid.best_params_,
    "cv_roc_auc": cv_scores.mean(),
    "test_roc_auc": roc_auc_score(y_test, y_proba),
    "version": "1.0",
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("\n✅ Model and metadata saved")


loaded_model = joblib.load(MODEL_PATH)
loaded_pred = loaded_model.predict(X_test)

print("\n=== LOADED MODEL CHECK ===")
print("Accuracy:", accuracy_score(y_test, loaded_pred))
