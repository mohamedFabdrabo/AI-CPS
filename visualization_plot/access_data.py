import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Path to the dataset
data_path = "/root/league_tables.csv"
# Load the dataset
df = pd.read_csv(data_path)

# Split the dataset into 80% training and 20% testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save to CSV files
train_df.to_csv("dataset_training.csv", index=False)
test_df.to_csv("dataset_testing.csv", index=False)

# Define the dependent and independent variables
X = df[['GP', 'W', 'D', 'L', 'F', 'A', 'GD']]
y = df['P']

# Add a constant to the independent variables
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Residuals and fitted values
fitted_values = model.fittedvalues
residuals = model.resid

# Standardized residuals
standardized_residuals = residuals / np.std(residuals)

# Influence (Leverage and Cook's Distance)
influence = model.get_influence()
leverage = influence.hat_matrix_diag
cooks_d = influence.cooks_distance[0]

# 1. Scatter Plot
plt.figure(figsize=(10, 6))

for column in X.columns[1:]:  # Skip the constant column
    plt.scatter(df[column], y, alpha=0.5, label=f'{column} vs P (Training)')
    plt.plot(df[column], fitted_values, color='red', label=f'{column} vs Fitted (Model)')

plt.xlabel('Predictors')
plt.ylabel('Target (P)')
plt.title('Scatter Plot of Predictors vs Target')
plt.legend(loc='upper right')
plt.savefig('scatter_plot.pdf')
plt.show()

# 2. Box Plot
plt.figure(figsize=(10, 6))
df.boxplot(column=['GP', 'W', 'D', 'L', 'F', 'A', 'GD', 'P'], grid=False, patch_artist=True)
plt.title('Box Plot of All Variables')
plt.savefig('box_plot.pdf')
plt.show()

# 3. Diagnostic Plots
plt.figure(figsize=(15, 15))

# Residuals vs Fitted
plt.subplot(2, 2, 1)
plt.scatter(fitted_values, residuals, alpha=0.5)
plt.axhline(0, linestyle='--', color='red')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
