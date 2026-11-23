
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load validation data
X_train, X_val, y_train, y_val = joblib.load(
    "models/processed_training_data.pkl"
)

# Load trained models
model = joblib.load("models/best_model.pkl")

# Convert sparse data if required
if hasattr(X_val, "toarray"):
    X_val = X_val.toarray()

# Predictions
predictions_log = model.predict(X_val)

# Convert log prices to actual prices
actual = np.expm1(y_val)
predicted = np.expm1(predictions_log)

# Calculate metrics
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("\nHOUSE PRICE MODEL EVALUATION")
print("-" * 40)
print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

# Create reports folder
os.makedirs("reports", exist_ok=True)

# Actual vs Predicted graph
plt.figure(figsize=(8, 6))
plt.scatter(actual, predicted, alpha=0.6)
plt.plot(
    [actual.min(), actual.max()],
    [actual.min(), actual.max()],
    linestyle="--"
)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("reports/actual_vs_predicted.png")
plt.show()

# Residual distribution
residuals = actual - predicted

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Residual Error Distribution")
plt.tight_layout()
plt.savefig("reports/residual_distribution.png")
plt.show()

print("\nEvaluation completed successfully!")
