import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
data = pd.read_excel('/Users/macair45/Downloads/Datasets/LendingClub.xlsx')
df = pd.DataFrame(data)
non_int_col = df.select_dtypes(exclude= ['Float64', 'int64']).columns

for col in non_int_col:
    try:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
        df[col] = df[col].fillna(0).astype(int)
    except:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except:
            pass
X, y = df.drop(columns=['Loan status']), df['Loan status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=3)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression())
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print('Accuracy is ',accuracy_score(y_pred, y_test)*100, '%')



# Count defaults (1) and repaid (0)
default_counts = df['dpnm'].value_counts()
default_percent = df['dpnm'].value_counts(normalize=True) * 100

# Display results
print("Counts of loan outcomes:")
print(default_counts)
print("\nPercentage of loan outcomes:")
print(default_percent.round(2))

