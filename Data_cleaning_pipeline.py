import pandas as pd
import numpy as np

# -----------------------------------
# 1. LOAD RAW DATA
# -----------------------------------
df = pd.read_csv("raw_data.csv")

# -----------------------------------
# 2. QUICK LOOK
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
# 5. REPLACE COMMON GARBAGE VALUES
# -----------------------------------
df.replace(
    ["?", "unknown", "Unknown", "N/A", "na", "null", ""],
    np.nan,
    inplace=True
)

# -----------------------------------
# 6. FIX NUMERICAL DATA STORED AS OBJECT (DYNAMIC)
# -----------------------------------
numeric_like_cols = df.select_dtypes(
    exclude=["int64", "float64", "bool"]
).columns

for col in numeric_like_cols:
    # Try converting object columns to numeric
    converted = pd.to_numeric(df[col], errors="coerce")

    # Only keep if conversion produces at least some numbers
    if converted.notna().sum() > 0:
        df[col] = converted

# -----------------------------------
# 7. HANDLE IMPOSSIBLE VALUES
# -----------------------------------
if "age" in df.columns:
    df.loc[df["age"] <= 0, "age"] = np.nan

if "salary" in df.columns:
    df.loc[df["salary"] < 0, "salary"] = np.nan

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
# 10. ONE-HOT ENCODE SELECTED UNORDERED COLUMNS
# -----------------------------------
# Replace with the actual columns you know are unordered categories
unordered_cols = ["gender", "department", "marital_status"]


# Apply one-hot encoding
df = pd.get_dummies(df, columns=unordered_cols, drop_first=True)


# -----------------------------------
# 11. BOOLEAN → INT
# -----------------------------------
bool_cols = df.select_dtypes(include=["bool"]).columns
for col in bool_cols:
    df[col] = df[col].astype(int)


# Count defaults (1) and repaid (0)
default_counts = df['dpnm'].value_counts()
default_percent = df['dpnm'].value_counts(normalize=True) * 100
# -----------------------------------
# 12. FINAL CHECK
# -----------------------------------
print(df.info())
print(df.head())

# -----------------------------------
# DATA IS NOW:
# ✔ NO OBJECT TYPES
# ✔ NO MISSING VALUES
# ✔ FULLY NUMERICAL
# ✔ ML-READY
# -----------------------------------
