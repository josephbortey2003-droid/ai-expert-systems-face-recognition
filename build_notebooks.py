"""Builds the practice notebooks for the AI & Expert Systems repo.
Run: .venv/Scripts/python build_notebooks.py   (then execute them with nbconvert)
Plain json only, so it does not need nbformat installed."""
import json, pathlib

OUT = pathlib.Path(__file__).parent / "notebooks"
OUT.mkdir(exist_ok=True)


def nb(cells):
    out = []
    for kind, src in cells:
        src = src.strip("\n")
        c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def save(name, cells):
    (OUT / name).write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print("wrote", name)


# ---------------------------------------------------------------- heart disease
save("03_heart_disease_classification.ipynb", [
("markdown", """
# Hospital heart disease task — binary classification

**Goal:** predict whether a patient has heart disease (`heart_disease` = 1) from 10 risk factors.

**Dataset:** `hospital_synthetic_dataset.xlsx` (from the course Google Drive, *Assignments → Hospital heart disease task*).

**What this notebook covers (same flow as the class ROC/AUC tutorial):**
1. Load and explore the data (EDA)
2. Preprocess: encode Yes/No and Male/Female, split train/test, scale
3. Train 3 models: Logistic Regression, Random Forest, Gradient Boosting
4. Evaluate: confusion matrix, accuracy, precision, recall, F1
5. ROC curve and AUC for all models
6. Threshold tuning
7. What the results mean
"""),
("code", """
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, classification_report,
                             roc_curve, roc_auc_score, accuracy_score)
RANDOM_STATE = 42
df = pd.read_excel('../data/hospital_synthetic_dataset.xlsx')
print(df.shape)
df.head()
"""),
("markdown", "## 1. Explore the data"),
("code", """
df.info()
print('\\nMissing values per column:\\n', df.isna().sum())
df.describe().T
"""),
("code", """
print(df['heart_disease'].value_counts())
print('Share with heart disease: %.1f%%' % (100 * df['heart_disease'].mean()))
sns.countplot(x='heart_disease', data=df); plt.title('Class balance'); plt.show()
"""),
("markdown", "**Class balance matters:** if one class were 90% of the data, a model could score 90% accuracy by always guessing it. That is why we also look at recall, precision and AUC, not only accuracy."),
("code", """
num_cols = ['age', 'blood_pressure', 'cholesterol', 'bmi', 'physical_activity', 'stress_level']
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, c in zip(axes.ravel(), num_cols):
    sns.boxplot(x='heart_disease', y=c, data=df, ax=ax); ax.set_title(c)
plt.tight_layout(); plt.show()
"""),
("code", """
cat_cols = ['sex', 'smoker', 'diabetes', 'family_history']
for c in cat_cols:
    print(pd.crosstab(df[c], df['heart_disease'], normalize='index').round(2), '\\n')
"""),
("markdown", "## 2. Preprocess\nYes/No → 1/0, Male → 1, Female → 0. Then a stratified 80/20 train/test split (stratified = both sets keep the same share of heart-disease cases)."),
("code", """
data = df.copy()
for c in ['smoker', 'diabetes', 'family_history']:
    data[c] = data[c].map({'Yes': 1, 'No': 0})
data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})
X = data.drop(columns='heart_disease'); y = data['heart_disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print('train', X_train.shape, 'test', X_test.shape)
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0); plt.title('Correlation matrix'); plt.show()
"""),
("markdown", "## 3. Train three models\nLogistic Regression needs scaled features (so it sits in a pipeline with `StandardScaler`). Tree models do not need scaling."),
("code", """
models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
}
probs = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    probs[name] = m.predict_proba(X_test)[:, 1]
    pred = (probs[name] >= 0.5).astype(int)
    print(f'===== {name} =====')
    print('Accuracy: %.3f   AUC: %.3f' % (accuracy_score(y_test, pred), roc_auc_score(y_test, probs[name])))
    print(classification_report(y_test, pred, digits=3))
"""),
("markdown", """
## 4. Confusion matrices
|                | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN (true negative) | FP (false positive — healthy person flagged) |
| **Actual 1** | FN (false negative — sick person missed) | TP (true positive) |

In medicine a **false negative is usually the worst mistake** (a sick patient sent home), so **recall** (TP / (TP+FN)) is important.
"""),
("code", """
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, p) in zip(axes, probs.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, (p >= 0.5).astype(int))).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout(); plt.show()
"""),
("markdown", "## 5. ROC curves and AUC\nThe ROC curve plots **TPR (recall)** against **FPR** for every threshold. **AUC** = area under it: 1.0 is perfect, **0.5 is no better than a coin flip** (the dashed diagonal)."),
("code", """
plt.figure(figsize=(7, 6))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, p):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess (AUC = 0.5)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate (Recall)'); plt.title('ROC curves — heart disease')
plt.legend(); plt.show()
"""),
("markdown", "## 6. Cross-validation (a fairer check than one split)\nWith only ~100 test patients, one split can be lucky or unlucky. 5-fold cross-validation trains and tests 5 times on different parts of the data."),
("code", """
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=cv, scoring='roc_auc')
    print(f'{name:20s} AUC = {s.mean():.3f} ± {s.std():.3f}')
"""),
("markdown", "## 7. Threshold tuning\nThe default rule is \"predict 1 if probability ≥ 0.5\". Lowering the threshold catches more sick patients (higher recall) but raises false alarms."),
("code", """
p = probs['Logistic Regression']
rows = []
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    tn, fp, fn, tp = confusion_matrix(y_test, (p >= t).astype(int)).ravel()
    rows.append({'threshold': t, 'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
                 'recall': tp / (tp + fn), 'false_positive_rate': fp / (fp + tn)})
pd.DataFrame(rows).round(3)
"""),
("code", """
rf = models['Random Forest']
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
imp.plot.barh(); plt.title('Random Forest feature importance'); plt.show()
coef = pd.Series(models['Logistic Regression'][-1].coef_[0], index=X.columns).sort_values()
coef.plot.barh(color='tab:orange'); plt.title('Logistic Regression coefficients (scaled features)'); plt.show()
"""),
("markdown", "## 8. Conclusion\n_(filled in after running — see the cell below)_"),
])

# ---------------------------------------------------------------- iris
save("04_iris_multiclass_classification.ipynb", [
("markdown", """
# Iris dataset task — multi-class classification

**Goal:** classify iris flowers into 3 species (setosa, versicolor, virginica) from 4 measurements.

This is the **multi-class** version of the confusion matrix (like the *ball colours* notes from class): with 3 classes the confusion matrix is 3×3, and precision/recall are calculated **per class**.
"""),
("code", """
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, roc_auc_score
iris = load_iris(as_frame=True)
df = iris.frame.copy(); df['species'] = df['target'].map(dict(enumerate(iris.target_names)))
print(df.shape); df.head()
"""),
("markdown", "## 1. Explore"),
("code", """
print(df['species'].value_counts())
df.groupby('species').mean(numeric_only=True).round(2)
"""),
("code", """
sns.pairplot(df.drop(columns='target'), hue='species', height=2.2); plt.show()
"""),
("markdown", "**Observation to look for:** petal length and petal width separate *setosa* completely; *versicolor* and *virginica* overlap a little — that is where mistakes will happen."),
("markdown", "## 2. Split and train three models"),
("code", """
X = df[iris.feature_names]; y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    'K-Nearest Neighbours (k=5)': make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
}
for name, m in models.items():
    m.fit(X_train, y_train); pred = m.predict(X_test)
    print(f'===== {name}  accuracy = {accuracy_score(y_test, pred):.3f}')
    print(classification_report(y_test, pred, target_names=iris.target_names, digits=3))
"""),
("markdown", """
## 3. The 3×3 confusion matrix
- Rows = the **true** species, columns = the **predicted** species.
- The **diagonal** = correct predictions. Anything off the diagonal is a mistake, and it tells you *which* species was confused with which.
- **Precision for a class** = correct predictions of that class ÷ everything predicted as that class (read down the column).
- **Recall for a class** = correct predictions of that class ÷ everything that truly is that class (read across the row).
"""),
("code", """
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (name, m) in zip(axes, models.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, m.predict(X_test)), display_labels=iris.target_names).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout(); plt.show()
"""),
("code", """
# Working precision and recall out by hand for one model, to show where the numbers come from
cm = confusion_matrix(y_test, models['Logistic Regression'].predict(X_test))
for i, name in enumerate(iris.target_names):
    tp = cm[i, i]; precision = tp / cm[:, i].sum(); recall = tp / cm[i, :].sum()
    print(f'{name:12s} TP={tp:2d}  precision={precision:.3f}  recall={recall:.3f}')
"""),
("markdown", "## 4. Multi-class ROC AUC (one-vs-rest) and cross-validation"),
("code", """
for name, m in models.items():
    auc = roc_auc_score(y_test, m.predict_proba(X_test), multi_class='ovr')
    cv = cross_val_score(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print(f'{name:28s} one-vs-rest AUC = {auc:.3f}   5-fold CV accuracy = {cv.mean():.3f} ± {cv.std():.3f}')
"""),
("code", """
plt.figure(figsize=(12, 6))
plot_tree(models['Decision Tree'], feature_names=iris.feature_names, class_names=iris.target_names, filled=True); plt.show()
"""),
("markdown", "## 5. Conclusion\n_(filled in after running — see the cell below)_"),
])

# ---------------------------------------------------------------- linear regression
save("05_linear_regression_practice.ipynb", [
("markdown", """
# Linear regression practice

Mr. Harry: *"Read up on linear regression and logistic regression, and practise both thoroughly."*

**Linear regression** predicts a **number** (e.g. weight in kg). **Logistic regression** predicts a **probability of a class** (e.g. heart disease yes/no) — see notebooks 03 and 04.

Data: the 5,000 synthetic patient records from the earlier AI & ES assignment (Height → Weight).
"""),
("code", """
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_excel('../data/synthetic_patient_records.xlsx', sheet_name='Patient_Records')
adults = df[df['Age'] >= 18]
X = adults[['Height_cm']]; y = adults['Weight_kg']
print(len(adults), 'adult records')
"""),
("markdown", "## 1. The formula by hand: y = b0 + b1·x\nb1 (slope) = covariance(x, y) / variance(x); b0 (intercept) = mean(y) − b1·mean(x)."),
("code", """
x = X['Height_cm'].values; yy = y.values
b1 = np.cov(x, yy, ddof=1)[0, 1] / np.var(x, ddof=1)
b0 = yy.mean() - b1 * x.mean()
print(f'By hand:      weight = {b0:.2f} + {b1:.3f} × height')
lr = LinearRegression().fit(X, y)
print(f'scikit-learn: weight = {lr.intercept_:.2f} + {lr.coef_[0]:.3f} × height   (should match)')
"""),
("markdown", "**Reading the slope:** every extra 1 cm of height adds about b1 kg of predicted weight."),
("code", """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression().fit(X_train, y_train); pred = lr.predict(X_test)
print('MAE  = %.2f kg  (average size of the error)' % mean_absolute_error(y_test, pred))
print('RMSE = %.2f kg  (punishes big errors more)' % mean_squared_error(y_test, pred) ** 0.5)
print('R²   = %.3f     (share of the variation in weight explained by height)' % r2_score(y_test, pred))
plt.scatter(X_test, y_test, s=6, alpha=.4, label='actual')
xs = np.linspace(X['Height_cm'].min(), X['Height_cm'].max(), 50).reshape(-1, 1)
plt.plot(xs, lr.predict(pd.DataFrame(xs, columns=['Height_cm'])), 'r', label='fitted line')
plt.xlabel('Height (cm)'); plt.ylabel('Weight (kg)'); plt.legend(); plt.show()
"""),
("markdown", "## 2. Residuals — checking the model\nA residual = actual − predicted. Good linear models have residuals scattered randomly around 0 with no pattern."),
("code", """
res = y_test - pred
plt.scatter(pred, res, s=6, alpha=.4); plt.axhline(0, color='r'); plt.xlabel('Predicted weight'); plt.ylabel('Residual'); plt.show()
"""),
("markdown", "## 3. Multiple linear regression (more than one input)"),
("code", """
X2 = adults[['Height_cm', 'Age', 'Haemoglobin_g_dL']]
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)
m2 = LinearRegression().fit(X_train, y_train)
print(dict(zip(X2.columns, m2.coef_.round(3))), 'intercept', round(m2.intercept_, 2))
print('R² with 3 inputs = %.3f' % r2_score(y_test, m2.predict(X_test)))
"""),
])

# ---------------------------------------------------------------- ollama
save("06_ollama_llama3.2_1b.ipynb", [
("markdown", """
# Running a local LLM: Ollama + Llama 3.2 1B

Mr. Harry: *"Download Ollama, then pull and run Llama 3.2 1B — it's lightweight and runs efficiently on CPU."*

- **Ollama** = the *inference software* (the "Netflix" in his DeepSeek analogy) that serves a model on your own computer.
- **Llama 3.2 1B** = the *model* (the "movie"): 1 billion parameters, about 1.3 GB, runs on a normal laptop CPU.

Setup (terminal):
```
ollama pull llama3.2:1b
ollama run llama3.2:1b
```
Ollama also exposes a local API on `http://localhost:11434`, which this notebook calls from Python.
"""),
("code", """
import json, time, urllib.request
def ask(prompt, model='llama3.2:1b'):
    body = json.dumps({'model': model, 'prompt': prompt, 'stream': False}).encode()
    req = urllib.request.Request('http://localhost:11434/api/generate', data=body, headers={'Content-Type': 'application/json'})
    t = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    print(f"[{time.time() - t:.1f}s, {out.get('eval_count', '?')} tokens]")
    return out['response']
print(ask('In two sentences, what is the difference between a model and an inference engine?'))
"""),
("code", """
print(ask('Explain a confusion matrix to a first-year student in 3 short bullet points.'))
"""),
("markdown", "**Note:** a 1B model is small — it is fast on CPU but makes more factual mistakes than big models. Always check its answers."),
])
