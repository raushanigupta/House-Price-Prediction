
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load dataset
df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Separate features and target
X = df.drop(columns=["SalePrice", "Id"])
y = np.log1p(df["SalePrice"])

# Keep test dataset IDs
test_ids = test_df["Id"]
X_test = test_df.drop(columns=["Id"])

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("Numerical Features:", len(numerical_features))
print("Categorical Features:", len(categorical_features))

# Numerical preprocessing
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Split training data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform validation and test data
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Create folders if needed
os.makedirs("models", exist_ok=True)

# Save preprocessing artifacts
joblib.dump(preprocessor, "models/preprocessor.pkl")
joblib.dump(
    (X_train_processed, X_val_processed, y_train, y_val),
    "models/processed_training_data.pkl"
)
joblib.dump(X_test_processed, "models/processed_test_data.pkl")
joblib.dump(test_ids, "models/test_ids.pkl")

print("\nPreprocessing Completed Successfully!")
print("Training Shape:", X_train_processed.shape)
print("Validation Shape:", X_val_processed.shape)
print("Testing Shape:", X_test_processed.shape)
