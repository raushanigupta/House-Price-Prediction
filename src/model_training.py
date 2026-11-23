
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load preprocessed data
X_train, X_val, y_train, y_val = joblib.load(
    "models/processed_training_data.pkl"
)

# Gradient Boosting requires dense input
if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()
    X_val = X_val.toarray()

# Convert log prices back to actual prices
y_val_actual = np.expm1(y_val)

# Define ML models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

print("=" * 60)
print("       HOUSE PRICE PREDICTION - MODEL TRAINING")
print("=" * 60)

# Train and evaluate models
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions_log = model.predict(X_val)
    predictions = np.expm1(predictions_log)

    mae = mean_absolute_error(y_val_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_val_actual, predictions))
    r2 = r2_score(y_val_actual, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

# Compare model performance
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="RMSE")

print("\n" + "=" * 60)
print("             MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

# Select best model using lowest RMSE
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

# Save best model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

# Save comparison results
os.makedirs("reports", exist_ok=True)
results_df.to_csv("reports/model_comparison.csv", index=False)

print("\nBest Model:", best_model_name)
print("Best model saved successfully!")
print("Model comparison saved successfully!")
