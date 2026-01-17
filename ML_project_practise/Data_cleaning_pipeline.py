import pandas as pd
import numpy as np

# -----------------------------------
# 1. LOAD RAW DATA
# -----------------------------------
df = pd.read_csv("raw_data.csv")

# -----------------------------------
# 2. QUICK INSPECTION
# -----------------------------------
print(df.head())
print(df.dtypes)
print(df.isnull().sum())

# -----------------------------------
# 3. STANDARDISE COLUMN NAMES
# -----------------------------------
df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(" ", "_")
)

# -----------------------------------
# 4. REMOVE DUPLICATES
# -----------------------------------
df = df.drop_duplicates()

# -----------------------------------
# 5. REPLACE COMMON GARBAGE VALUES WITH NaN
# -----------------------------------
garbage_values = ["?", "unknown", "Unknown", "N/A", "na", "null", ""]
df.replace(garbage_values, np.nan, inplace=True)

# -----------------------------------
# 6. DYNAMIC NUMERIC CONVERSION
# -----------------------------------
# Convert object columns to numeric where possible
object_cols = df.select_dtypes(include=["object"]).columns

for col in object_cols:
    numeric_version = pd.to_numeric(df[col], errors="coerce")
    if numeric_version.notna().sum() > 0:
        df[col] = numeric_version

# -----------------------------------
# 7. HANDLE IMPOSSIBLE VALUES (SANITY CHECKS)
# -----------------------------------
if "age" in df.columns:
    df.loc[df["age"] <= 0, "age"] = np.nan
if "salary" in df.columns:
    df.loc[df["salary"] < 0, "salary"] = np.nan

# Add more sanity checks here if needed, e.g., scores <= max_value

# -----------------------------------
# 8. HANDLE MISSING VALUES
# -----------------------------------
# Numerical → median
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical → mode
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -----------------------------------
# 9. ORDERED (ORDINAL) ENCODING
# -----------------------------------
ordered_mappings = {
    "education_level": {
        "primary": 1,
        "secondary": 2,
        "bachelor": 3,
        "master": 4,
        "phd": 5
    },
    "experience_level": {
        "junior": 1,
        "mid": 2,
        "senior": 3
    }
}

for col, mapping in ordered_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# -----------------------------------
# 10. ONE-HOT ENCODE UNORDERED CATEGORICALS
# -----------------------------------
unordered_cols = [c for c in df.select_dtypes(include=["object"]).columns 
                  if c not in ordered_mappings.keys()]

df = pd.get_dummies(df, columns=unordered_cols, drop_first=True)

# -----------------------------------
# 11. BOOLEAN → INT
# -----------------------------------
bool_cols = df.select_dtypes(include=["bool"]).columns
for col in bool_cols:
    df[col] = df[col].astype(int)

# -----------------------------------
# 12. FINAL CHECK
# -----------------------------------
print(df.info())
print(df.head())
print("Missing values remaining:\n", df.isnull().sum())

# -----------------------------------
# DATA IS NOW:
# ✔ CLEAN
# ✔ NUMERICAL / ENCODED
# ✔ NO DUPLICATES
# ✔ NO MISSING VALUES
# ✔ ML-READY
# -----------------------------------
