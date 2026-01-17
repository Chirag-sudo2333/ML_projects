import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# -----------------------------
# Step 0: Dataset
# -----------------------------
def dataset():
    data_dict = {
        'Age': [25, 32, 47, 51, 23, 36, 44, 28, 39, 50],
        'Income': [50000, 60000, 120000, 90000, 45000, 80000, 110000, 52000, 75000, 95000],
        'Education': ['Bachelors', 'Masters', 'PhD', 'Bachelors', 'High School', 'Masters', 'PhD', 'Bachelors', 'Masters', 'PhD'],
        'City': ['New York', 'Chicago', 'LA', 'Houston', 'Chicago', 'New York', 'LA', 'Houston', 'Chicago', 'New York'],
        'YearsAtJob': [2, 8, 15, 10, 1, 7, 12, 3, 6, 11],
        'OwnsHouse': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes'],
        'LoanDefault': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    }
    return data_dict

df = pd.DataFrame(dataset())
df.columns = df.columns.str.lower().str.replace(' ', '_')
target = 'loandefault'

# -----------------------------
# Step 1: Inspect dataset
# -----------------------------
print("Target distribution (%):\n", df[target].value_counts(normalize=True)*100)
print("\nDataset description:\n", df.describe())
print("\nColumn types:\n", df.dtypes)

# -----------------------------
# Step 2: Separate numeric & categorical
# -----------------------------
num_col = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_col = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

if target in num_col:
    num_col.remove(target)
if target in cat_col:
    cat_col.remove(target)

print("\nNumeric columns:", num_col)
print("Categorical columns:", cat_col)

# -----------------------------
# Step 3: Preprocessing & pipeline
# -----------------------------
num_cnv = StandardScaler()
cat_cnv = OneHotEncoder(handle_unknown='ignore')

preprocess = ColumnTransformer([
    ('num', num_cnv, num_col),
    ('cat', cat_cnv, cat_col)
])

ppln = Pipeline([
    ('preprocessor', preprocess),
    ('classifier', LogisticRegression(max_iter=1000))
])

# -----------------------------
# Step 4: Train/Test split
# -----------------------------
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

ppln.fit(X_train, y_train)
y_pred = ppln.predict(X_test)

# -----------------------------
# Step 5: Basic evaluation
# -----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# Step 6: Check encoded features
# -----------------------------
preprocessor = ppln.named_steps['preprocessor']
onehot_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_col)
final_features = num_col + list(onehot_features)
print("\nEncoded feature names:\n", final_features)

# -----------------------------
# Step 7: Feature importance
# -----------------------------
coefs = ppln.named_steps['classifier'].coef_[0]
num_coefs = coefs[:len(num_col)]
cat_coefs = coefs[len(num_col):]

print("\nNumeric feature coefficients:")
for f, c in zip(num_col, num_coefs):
    print(f"{f}: {c:.3f}")

print("\nCategorical feature coefficients:")
for f, c in zip(onehot_features, cat_coefs):
    print(f"{f}: {c:.3f}")

# -----------------------------
# Step 8: Prediction probabilities
# -----------------------------
y_prob = ppln.predict_proba(X_test)[:, 1]
print("\nPredicted probabilities of default:\n", y_prob)

# Threshold tuning
threshold = 0.6
y_pred_new = (y_prob > threshold).astype(int)
print("\nClassification report with threshold 0.6:\n", classification_report(y_test, y_pred_new))

# -----------------------------
# Step 9: Misclassified samples
# -----------------------------
misclassified = X_test[y_test != y_pred_new]
print("\nMisclassified samples:\n", misclassified)

# -----------------------------
# Step 10: Probability histogram
# -----------------------------
plt.hist(y_prob, bins=10)
plt.title("Predicted Probabilities of Default")
plt.xlabel("Probability")
plt.ylabel("Count")
plt.show()
