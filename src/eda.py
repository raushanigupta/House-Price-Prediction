
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/train.csv")

# Create visualizations
sns.set_theme(style="whitegrid")

# 1. Sale Price Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["SalePrice"], kde=True)
plt.title("House Price Distribution")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.savefig("reports/price_distribution.png")
plt.show()

# 2. House Area vs Sale Price
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x="GrLivArea", y="SalePrice")
plt.title("Living Area vs House Price")
plt.xlabel("Living Area (sq ft)")
plt.ylabel("Sale Price")
plt.savefig("reports/area_vs_price.png")
plt.show()

# 3. Correlation Heatmap
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.savefig("reports/correlation_heatmap.png")
plt.show()

# 4. Missing Values Analysis
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\nTop Missing Values:")
print(missing.head(20))

print("\nEDA Completed Successfully!")
