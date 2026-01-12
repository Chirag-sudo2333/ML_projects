import pandas as pd

# Load the dataset
file_path = "/Users/macair45/Downloads/pink.csv"
df = pd.read_csv(file_path)

# Count defaults (1) and repaid (0)
default_counts = df['dpnm'].value_counts()
default_percent = df['dpnm'].value_counts(normalize=True) * 100

# Display results
print("Counts of loan outcomes:")
print(default_counts)
print("\nPercentage of loan outcomes:")
print(default_percent.round(2))
