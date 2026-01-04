import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
data = {
    "Age": [
        22, 25, 28, 30, 35, 40, 45, 50, 23, 27,
        32, 38, 42, 48, 29, 34, 37, 41, 46, 52,
        24, 26, 31, 36, 39, 44, 49, 54, 21, 33,
        35, 28, 47, 51, 43, 38, 26, 29, 45, 50
    ],

    "Education": [
        "High School", "Bachelor", "Bachelor", "Master", "Master",
        "PhD", "PhD", "PhD", "High School", "Bachelor",
        "Master", "Master", "PhD", "PhD", "Bachelor",
        "Master", "Master", "PhD", "PhD", "PhD",
        "High School", "Bachelor", "Bachelor", "Master", "Master",
        "PhD", "PhD", "PhD", "High School", "Master",
        "Master", "Bachelor", "PhD", "PhD", "PhD",
        "Master", "Bachelor", "Bachelor", "PhD", "PhD"
    ],

    "Salary": [
        20000, 30000, 35000, 45000, 60000, 80000, 90000, 100000,
        22000, 32000, 48000, 55000, 75000, 95000, 37000,
        52000, None, 78000, 88000, 110000,
        21000, 31000, None, 50000, 58000, 82000, 97000, 115000,
        19000, 54000, 60000, 36000, 92000, 108000, 86000,
        57000, 34000, 38000, None, 102000
    ],

    "Loan_Repaid": [
        0, 1, 1, 1, 1, 1, 1, 1, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 0, 1, 1, 1, 1, 1, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
}
df = pd.DataFrame(data)
edu_map = {
    "High School": 0,
    "Bachelor": 1,
    "Master": 2,
    "PhD": 3
}
scaler = StandardScaler()
df['Education'] = df["Education"].map(edu_map)
df = df.drop_duplicates()
print(np.sum(df.duplicated()))
df['Salary'] = df['Salary'].fillna(np.mean(df['Salary']))
print(np.sum(df.isnull()))
X = (df.drop(columns=['Loan_Repaid'])).to_numpy()
y = (df['Loan_Repaid']).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=3)
X_scaled = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_scaled, y_train)
y_pred = model.predict(X_scaled_test)
print(model.predict_proba(X_scaled_test))
print('Accuracy is ', 100* accuracy_score(y_pred, y_test), '%.')