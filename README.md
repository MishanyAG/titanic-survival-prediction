# Titanic Survival Prediction

## Goal
Build and compare several machine learning models for predicting passenger survival on the Titanic dataset, then analyze errors and feature importance.

## Dataset
- File: `Titanic-Dataset.csv`
- Rows: 891
- Target: `Survived` (`0` = did not survive, `1` = survived)
- Features used in modeling:
  - Numerical: `Age`, `SibSp`, `Parch`, `Fare`
  - Categorical: `Pclass`, `Sex`, `Embarked`

## EDA
- Class balance is moderately imbalanced: about 62% class `0`, 38% class `1`.
- Missing values:
  - `Age`: many missing values
  - `Embarked`: a few missing values
  - `Cabin`: many missing values (not used in this version of the model)

## Preprocessing
- Train/test split with stratification by target.
- Numerical pipeline:
  - `SimpleImputer(strategy="median")`
  - `StandardScaler()`
- Categorical pipeline:
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`
- Combined with `ColumnTransformer` inside model pipelines.

## Models
- `DummyClassifier(strategy="most_frequent")` as baseline.
- `LogisticRegression`
- `RandomForestClassifier`
- `XGBClassifier`
- Hyperparameter tuning for trainable models with `RandomizedSearchCV` (`cv=5`).

## Metrics
The script reports:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix (`TN`, `FP`, `FN`, `TP`)

Notes:
- `RandomizedSearchCV` optimizes `f1`.
- Best final model in the script is selected by `F1`.

## Results
The script prints a comparison table for all models on the test split and selects the best model by `F1`.

Current run summary:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy | 0.614 | 0.000 | 0.000 | 0.000 | 0.500 |
| LogisticRegression | 0.776 | 0.743 | 0.640 | 0.688 | 0.838 |
| RandomForest | 0.798 | 0.773 | 0.674 | **0.720** | 0.829 |
| XGBoost | 0.798 | 0.797 | 0.640 | 0.710 | 0.809 |

Best model by F1 in this run: `RandomForestClassifier`.
`RandomForestClassifier` achieved the best F1-score in the current run, while `XGBoost` showed a close result. Since F1-score was selected as the main metric, `RandomForestClassifier` was chosen for further error analysis and permutation importance.

## Error Analysis
For the selected best model:
- Save all misclassified rows from `X_test`
- Add:
  - true label (`y_true`)
  - predicted label (`y_pred`)
  - predicted survival probability (`proba_survived`)

Current run:
- Test size: 223 rows
- Misclassified rows: 45
- Confusion matrix of the selected best model:
  - TN: 120
  - FP: 17
  - FN: 28
  - TP: 58

## Feature Importance
- Uses permutation importance on the test set.
- Reports mean and std importance values per feature.

Top features in the current run (permutation importance mean):
1. `Sex`: 0.262
2. `Pclass`: 0.155
3. `Age`: 0.052
4. `Fare`: 0.047
5. `Embarked`: 0.037
6. `SibSp`: 0.023
7. `Parch`: 0.020

## Conclusion

The best model in the current run is `RandomForestClassifier`, selected by F1-score.

Compared with the baseline `DummyClassifier`, all trained models show a clear improvement. This means that the selected passenger features contain useful predictive signal.

The most important features according to permutation importance are `Sex` and `Pclass`. This matches the historical intuition of the Titanic dataset: survival chances were strongly related to passenger gender and ticket class.

The selected model still makes classification errors. In the current run, it makes more false negative errors than false positive errors, meaning that some passengers who actually survived were predicted as not survived. This suggests that future improvements should focus on better feature engineering and improving recall without losing too much precision.

Overall, the project demonstrates a complete classical machine learning workflow: data preparation, preprocessing with pipelines, baseline comparison, model tuning, evaluation, error analysis, and feature importance interpretation.

## Future Improvements

Possible improvements for future versions:

- Add feature engineering:
  - `FamilySize` from `SibSp + Parch + 1`
  - `IsAlone` based on family size
  - `Title` extracted from `Name`
- Try threshold tuning to improve the balance between precision and recall.
- Add PR-AUC, especially because the target classes are moderately imbalanced.
- Add visualizations:
  - target balance plot
  - missing values plot
  - confusion matrix plot
  - feature importance plot
- Compare results before and after feature engineering.
- Move generated files such as `.xlsx` outputs into an `outputs/` folder.

## How to run
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Place `Titanic-Dataset.csv` in the project root.
3. Run:
   - `python main.py`

Generated output files:
- `model_results.xlsx`
- `error_analysis.xlsx`
- `permutation_importance.xlsx`
