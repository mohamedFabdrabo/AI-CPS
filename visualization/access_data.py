import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

# Load the dataset
df = pd.read_csv('/root/league_tables.csv')

# Inspect the data
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values if any
df = df.dropna()

# Convert columns to appropriate data types
df = df.apply(pd.to_numeric, errors='ignore')

# Check for outliers using the IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# Detect outliers in the 'P' column (Points)
outliers = detect_outliers_iqr(df, 'P')
print("Outliers detected:")
print(outliers)

# Remove outliers
df = df[~df.index.isin(outliers.index)]

# Define independent variables (X) and dependent variable (y)
X = df[['GP', 'W', 'D', 'L', 'F', 'A', 'GD']]  # Independent variables
y = df['P']  # Dependent variable (Points)

# Add a constant to the independent variables matrix (for the intercept)
X = sm.add_constant(X)

# Build the OLS model
ols_model = sm.OLS(y, X)
ols_results = ols_model.fit()

# Print the summary of the OLS model
print(ols_results.summary())

# Correlation matrix
correlation_matrix = df[['GP', 'W', 'D', 'L', 'F', 'A', 'GD']].corr()
print(correlation_matrix)

# Ridge regression
ridge = Ridge(alpha=1.0)  # You can tune alpha to control regularization strength
ridge.fit(X[['GP', 'W', 'D', 'L', 'F', 'A', 'GD']], y)

# Output Ridge coefficients
print("Ridge coefficients:", ridge.coef_)

# ------------- Plotting ----------------

# Diagnostic Plots (2x2 grid)
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Residuals vs Fitted Plot
residuals = ols_results.resid
fitted_values = ols_results.fittedvalues
axs[0, 0].scatter(fitted_values, residuals)
axs[0, 0].axhline(0, color='red', linestyle='--')
axs[0, 0].set_title("Residuals vs Fitted")
axs[0, 0].set_xlabel("Fitted values")
axs[0, 0].set_ylabel("Residuals")

# QQ Plot for Normality
sm.graphics.qqplot(residuals, line='45', ax=axs[0, 1])
axs[0, 1].set_title("QQ Plot")

# Leverage vs Residuals squared
sm.graphics.plot_leverage_resid2(ols_results, ax=axs[1, 0])
axs[1, 0].set_title("Leverage vs Residuals Squared")

# Cook's Distance
axs[1, 1].plot(np.arange(len(residuals)), ols_results.get_influence().cooks_distance[0], 'o')
axs[1, 1].set_title("Cook's Distance")

# Adjust layout for the diagnostic plots
plt.tight_layout()

# Save Diagnostic Plots as part of the final PDF
plt.savefig('/tmp/DiagnosticPlots.pdf')

# Scatter Plot with OLS Regression Line
plt.figure(figsize=(10, 6))
plt.scatter(X['GP'], y, color='orange', label='Data')

# OLS regression line
y_pred = ols_results.predict(X)
plt.plot(X['GP'], y_pred, color='red', label='OLS Regression Line')

# Customize the plot
plt.title("Scatter Plot with OLS Regression Line")
plt.xlabel("Games Played (GP)")
plt.ylabel("Points (P)")
plt.legend()
plt.grid(True)

# Save the Scatter Plot with OLS model
plt.savefig('/tmp/ScatterWithOLS.pdf')

# Ridge Regression Coefficients Plot
plt.figure(figsize=(10, 6))
plt.bar(X.columns[1:], ridge.coef_)  # Skip the constant term
plt.title('Ridge Regression Coefficients')
plt.xlabel('Feature')
plt.ylabel('Coefficient Value')

# Save Ridge coefficients plot
plt.savefig('/tmp/RidgeCoefficients.pdf')

# Correlation Matrix Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Features')

# Save the Correlation Matrix Heatmap
plt.savefig('/tmp/CorrelationMatrix.pdf')

# Show all plots
plt.show()

print("All plots saved as 'DiagnosticPlots.pdf', 'ScatterWithOLS.pdf', 'RidgeCoefficients.pdf', and 'CorrelationMatrix.pdf' in the /tmp directory.")
