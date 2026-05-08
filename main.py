import warnings
import pandas as pd
import openpyxl

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

pd.set_option("display.max_columns", None)

# Base data preparation
dataset = pd.read_csv("Titanic-Dataset.csv")
y = dataset["Survived"]
X = dataset[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]

# Data pipelines
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
missing_values = dataset.isna().sum()

print(rf"""
 _____ ___ _____  _    _   _ ___ ____ 
|_   _|_ _|_   _|/ \  | \ | |_ _/ ___|
  | |  | |  | | / _ \ |  \| || | |    
  | |  | |  | |/ ___ \| |\  || | |___ 
  |_| |___| |_/_/   \_\_| \_|___\____|
--------------------------------------
Titanic Survival Prediction
--------------------------------------
Dataset information:
      
*Shape - {dataset.shape}

*Target balance:
{
    (
        dataset["Survived"]
        .value_counts(normalize=True)
        .mul(100)
        .round(0)
        .astype(int)
        .astype(str) + "%"
    ).to_string()
}

*Missing values:
{(missing_values[missing_values>0].sort_values(ascending=False)).to_string()}
--------------------------------------
""")

# Pipeline models
logistic_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(penalty="l2", solver="lbfgs", random_state=42)),
])
forest_pipe = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(criterion="entropy", bootstrap=True, random_state=42),
    ),
])
xgb_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(random_state=42)),
])
dummy = DummyClassifier(strategy="most_frequent")

# Hyperparameter distributions for RandomizedSearchCV
param_dist_lr = {
    "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "classifier__max_iter": [100, 500, 1000, 1500, 2000],
}
param_dist_rf = {
    "classifier__n_estimators": [100, 200, 500],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}
param_dist_xgb = {
    "classifier__n_estimators": [100, 200, 500],
    "classifier__learning_rate": [0.01, 0.1, 0.2],
    "classifier__max_depth": [3, 6, 9],
    "classifier__gamma": [0, 0.1, 0.2],
    "classifier__reg_alpha": [0, 0.1],
    "classifier__reg_lambda": [1, 10],
}

# RandomizedSearchCV
search_lr = RandomizedSearchCV(
    estimator=logistic_pipe,
    param_distributions=param_dist_lr,
    n_iter=20,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
)
search_rf = RandomizedSearchCV(
    estimator=forest_pipe,
    param_distributions=param_dist_rf,
    n_iter=20,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
)
search_xgb = RandomizedSearchCV(
    estimator=xgb_pipe,
    param_distributions=param_dist_xgb,
    n_iter=20,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
)


# Fit
dummy.fit(X_train, y_train)
search_lr.fit(X_train, y_train)
search_rf.fit(X_train, y_train)
search_xgb.fit(X_train, y_train)

# Results
models = {
    "Dummy": dummy,
    "LogisticRegression": search_lr,
    "RandomForest": search_rf,
    "XGBoost": search_xgb,
}
results = []
best_params = []
for name, model in models.items():
    predict = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predict),
        "Precision": precision_score(y_test, predict),
        "Recall": recall_score(y_test, predict),
        "F1": f1_score(y_test, predict),
        "Roc-Auc": roc_auc_score(y_test, proba),
    })
    best_params.append({
        "Model": name,
        "Best-Params": getattr(model, "best_params_", None)
    })
df_best_params = pd.DataFrame(best_params)
df_metrics = pd.DataFrame(results)

best_row = df_metrics.loc[df_metrics["F1"].idxmax()]
best_model_name = best_row["Model"]
best_model = models[best_model_name]
tn, fp, fn, tp = confusion_matrix(y_test, best_model.predict(X_test)).ravel()

# Error analysis
errors = X_test.copy()
errors["y_true"] = y_test
errors["y_pred"] = best_model.predict(X_test)
errors["proba_survived"] = best_model.predict_proba(X_test)[:, 1]
errors = errors[errors["y_true"] != errors["y_pred"]]

result = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)
importance_df = pd.DataFrame({
    "Features": X_test.columns,
    "Importance_mean": result.importances_mean,
    "Importance_std": result.importances_std,
}).sort_values("Importance_mean", ascending=False)

print(f"""Model results:
{df_metrics.to_string(index=False)}

Best model by F1: {best_model_name}
{best_row.to_string()}
Confusion matrix:
TN: {tn:<3} | FP: {fp}
FN: {fn:<3} | TP: {tp}

Error analysis:
False positives: {fp}
False negatives: {fn}

Top - 5 important features:
{importance_df["Features"].head(5).to_string(index=False)}""")

# Excel
df_metrics.to_excel("model_results.xlsx", index=False)
errors.to_excel("error_analysis.xlsx", index=False)
importance_df.to_excel("permutation_importance.xlsx", index=False)
df_best_params.to_excel("best_parameters.xlsx", index=False)

print("""--------------------------------------
Saved Files:
- model_results.xlsx
- error_analysis.xlsx
- permutation_importance.xlsx
- best_parameters.xlsx
--------------------------------------""")