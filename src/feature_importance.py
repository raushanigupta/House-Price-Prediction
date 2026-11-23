
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load trained model and preprocessing pipeline
model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Get original feature names
df = pd.read_csv("data/train.csv")
X = df.drop(columns=["SalePrice", "Id"])

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Get model feature importance
if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_

elif hasattr(model, "coef_"):
    importance = abs(model.coef_)

else:
    raise ValueError("Selected model does not support feature importance.")

# Create feature importance dataframe
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort and select top 15 features
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

top_features = importance_df.head(15)

print("\nTOP 15 IMPORTANT FEATURES")
print("--------------------------------")
print(top_features.to_string(index=False))

# Save results
os.makedirs("reports", exist_ok=True)

importance_df.to_csv(
    "reports/feature_importance.csv",
    index=False
)

# Create visualization
plt.figure(figsize=(11, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("House Features")
plt.title("Top 15 Features Affecting House Prices")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    "reports/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFeature importance analysis completed!")
