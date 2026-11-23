
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load tuning comparison results
df = pd.read_csv("reports/tuning_comparison.csv")

# Sort models by RMSE
df = df.sort_values("Validation_RMSE")

print("\nMODEL PERFORMANCE COMPARISON")
print("--------------------------------")
print(df.to_string(index=False))

# Create reports folder
os.makedirs("reports", exist_ok=True)

# Create visualization
plt.figure(figsize=(10, 6))

bars = plt.bar(
    df["Model"],
    df["Validation_RMSE"]
)

# Add RMSE values above bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"${height:,.0f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.title("House Price Prediction - Model Comparison",
          fontsize=15, fontweight="bold")

plt.xlabel("Machine Learning Models")
plt.ylabel("Validation RMSE ($)")

plt.xticks(rotation=12)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

# Save graph
plt.savefig(
    "reports/model_comparison_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nModel comparison visualization completed!")
