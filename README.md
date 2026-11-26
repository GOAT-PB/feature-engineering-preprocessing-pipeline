# Feature Engineering & Preprocessing Pipeline (Titanic survival, binary classification)

A leak-free, reusable scikit-learn preprocessing pipeline: feature engineering → imputation → outlier clipping → scaling → one-hot encoding → model, with EDA, correlation/multicollinearity analysis, two feature-importance methods and a baseline model.

## Files

| File | Purpose |
|---|---|
| `Feature_Engineering_Preprocessing.ipynb` | Main deliverable: executed notebook (explanations, code, plots, results) |
| `preprocessing_pipeline.py` | Reusable module: data loading, custom transformers, pipeline builders, feature-name mapping, MI helper, metrics |
| `requirements.txt` | Python dependencies |
| `data/titanic.csv` | Bundled copy of the dataset (offline fallback) |
| `data/customer-churn-training.csv` | Uploaded 12-row file, used only for the reusability demo (Section 10) |

## Dataset

- **Titanic passengers**, 891 rows × 12 columns; target `Survived` (0/1).
- **Source:** GitHub mirror `datasciencedojo/datasets` (`titanic.csv`) of the Kaggle *Titanic – Machine Learning from Disaster* training file.
  `load_data()` reads `data/titanic.csv` first and the URL second, and raises a clear error if both fail.
- Has numeric features, categorical features, real missing values (`Age` 19.9 %, `Cabin` 77.1 %, `Embarked` 2 rows) and high-cardinality text columns (`Name`, `Ticket`, `Cabin`).
- **Limitations:** small (891 rows), historical, a learning dataset. Results do not generalise to modern problems.
- Why not the uploaded churn file as the main dataset: 12 rows, no missing values, so it cannot support a meaningful stratified split, imputation or importance analysis.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

```bash
jupyter notebook Feature_Engineering_Preprocessing.ipynb     # then Kernel -> Restart & Run All
# or headless:
jupyter nbconvert --to notebook --execute --inplace Feature_Engineering_Preprocessing.ipynb
```
Run it from the project folder (the notebook imports `preprocessing_pipeline.py` and reads `data/`).

Minimal use of the module:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocessing_pipeline import *

df = load_data()
X, y = df.drop(columns=TARGET_COL), df[TARGET_COL]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), TitanicFeatureEngineer())
pipe.fit(X_train, y_train)          # everything is learned from X_train only
print(evaluate_classifier(pipe, X_test, y_test)["metrics"])
```

## Pipeline architecture

```
raw X
 └─ TitanicFeatureEngineer (stateless, row-wise: Title, FamilySize, IsAlone, Deck, FarePerPerson; drops PassengerId/Name/Ticket/Cabin)
     └─ ColumnTransformer (columns chosen by dtype at fit time)
         ├─ numeric:     SimpleImputer(median) → QuantileClipper(1%,99%) → StandardScaler
         └─ categorical: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
             └─ estimator (Logistic Regression baseline; Random Forest for comparison)
```

Leakage rules: split first (`random_state=42`, `stratify=y`); anything that learns from data (median, clipping bounds, mean/std, categories) is a scikit-learn transformer fitted on `X_train` only; feature engineering is stateless; MI, Random Forest importance, correlation, VIF and the keep/drop decision all use training data only; cross-validation re-fits the whole pipeline inside each fold.

## Key findings (from the executed notebook)

- **Data:** `Cabin` 77.5 % and `Age` 19.2 % missing in the training split; `Fare`, `SibSp`, `Parch` strongly right-skewed. No duplicates.
- **Engineered features** show clear survival differences (e.g. `Title`: Mr 15.3 % vs Mrs 78.5 %; `FamilySize` is non-linear: 30 % alone, 72.7 % for a family of 4, near 0 % for large families).
- **Multicollinearity:** VIF up to 229 (`FamilySize`, `SibSp`, `Parch`, built from each other). Cross-validation showed dropping them changes scores by ≤ 0.007, so they were kept (option `drop_family_components=True` for coefficient interpretation).
- **Importance:** `Sex`, `Title`, `Fare`/`FarePerPerson` rank high in both Mutual Information and Random Forest; `Age` ranks 2nd for the forest but 11th for MI (Spearman between rankings 0.56).
- **Baseline (Logistic Regression), test set (179 rows):** accuracy 0.849, precision 0.828, recall 0.768, F1 0.797, ROC-AUC 0.870. Dummy model: 0.615 / 0.50. Test uncertainty is about ±5 points; Random Forest was not clearly better in cross-validation.
- **Validation:** 19 automated assertions (split, train-only statistics, no NaN, unseen categories, feature names, reproducibility) passed in the executed run.

## Testing status (what was and was not verified)

- The notebook was executed top to bottom (31/31 code cells, no errors, no warnings) with Python 3.12, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4, matplotlib 3.11.2, seaborn 0.13.2. It was also re-run from a clean copy of the folder, and the URL fallback of `load_data()` was tested.
- Older library versions (the lower bounds in `requirements.txt`) were **not** tested.
- No hyper-parameter tuning was done; the model is a baseline.
