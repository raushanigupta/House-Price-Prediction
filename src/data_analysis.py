
import pandas as pd
import numpy as np

# Load training and testing datasets
train_path = "data/train.csv"
test_path = "data/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Display project information
print("=" * 50)
print("       HOUSE PRICE PREDICTION PROJECT")
print("=" * 50)

print("\nTraining Dataset Shape:")
print(train_df.shape)

print("\nTesting Dataset Shape:")
print(test_df.shape)

print("\nFirst 5 Rows:")
print(train_df.head())

print("\nDataset Information:")
print(train_df.info())

print("\nStatistical Summary:")
print(train_df.describe())

print("\nMissing Values:")
print(train_df.isnull().sum().sort_values(ascending=False).head(20))

print("\nTarget Variable - SalePrice:")
print(train_df["SalePrice"].describe())

print("\nDataset Analysis Completed!")
