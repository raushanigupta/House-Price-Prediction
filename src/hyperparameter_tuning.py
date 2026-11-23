
import joblib
import numpy as np
import pandas as pd
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor


# Load processed training and validation data
X_train, X_val, y_train, y_val = joblib.load(
    "models/processed_training_data.pkl"
)

# Convert sparse matrices to dense arrays
if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()

if hasattr(X_val, "toarray"):
    X_val = X_val.toarray()

# Load existing best model for comparison
baseline_model = joblib.load("models/best_model.pkl")


# Hyperparameter tuning for Gradient Boosting
gbr_params = {
    "n_estimators": [100, 150, 200, 250],
    "learning_rate": [0.03, 0.05, 0.07, 0.1],
    "max_depth": [2, 3, 4],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "subsample": [0.8, 0.9, 1.0]
}

gbr_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_distributions=gbr_params,
    n_iter=8,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\nTuning Gradient Boosting...")
gbr_search.fit(X_train, y_train)

best_gbr = gbr_search.best_estimator_


# Hyperparameter tuning for XGBoost
xgb_params = {
    "n_estimators": [200, 300, 400],
    "learning_rate": [0.02, 0.03, 0.05],
    "max_depth": [2, 3, 4],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5]
}

xgb_search = RandomizedSearchCV(
    XGBRegressor(
        random_state=42,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=1
    ),
    param_distributions=xgb_params,
    n_iter=8,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\nTuning XGBoost...")
xgb_search.fit(X_train, y_train)

best_xgb = xgb_search.best_estimator_


# Compare tuned models with existing model
candidates = {
    "Existing Best Model": baseline_model,
    "Tuned Gradient Boosting": best_gbr,
    "Tuned XGBoost": best_xgb
}

results = []

for name, candidate in candidates.items():

    predictions_log = candidate.predict(X_val)

    actual = np.expm1(y_val)
    predicted = np.expm1(predictions_log)

    rmse = np.sqrt(mean_squared_error(actual, predicted))

    results.append({
        "Model": name,
        "Validation_RMSE": rmse
    })

    print(f"{name}: ${rmse:,.2f}")


# Select model with lowest validation RMSE
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Validation_RMSE")

best_name = results_df.iloc[0]["Model"]
best_model = candidates[best_name]

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df.to_csv(
    "reports/tuning_comparison.csv",
    index=False
)

if best_name != "Existing Best Model":
    joblib.dump(best_model, "models/best_model.pkl")
    print("\nImproved model saved successfully!")
else:
    print("\nExisting model is still the best. Original model retained.")

print("\nBEST MODEL:", best_name)
print("\nTuning completed successfully!")
