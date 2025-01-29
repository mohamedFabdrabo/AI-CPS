import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('/root/league_tables.csv')


# Assuming df is the DataFrame obtained from the scraping function
# Let's inspect the data
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values if any
df = df.dropna()

# Convert columns to appropriate data types
# For example, if there are columns with numeric data stored as strings, convert them to numeric
df = df.apply(pd.to_numeric, errors='ignore')

# Check for outliers using the IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]



# Print the column names to inspect
print(df.columns)

# Example: Detect outliers in a numeric column (replace 'numeric_column' with the actual column name)
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
model = sm.OLS(y, X)
results = model.fit()

# Print the summary of the model
print(results.summary())


correlation_matrix = df[['GP', 'W', 'D', 'L', 'F', 'A', 'GD']].corr()
print(correlation_matrix)

# Prepare the features (X) and target (y)
X = df[['GP', 'W', 'D', 'L', 'F', 'A', 'GD']]
y = df['P']

# Apply Ridge regression
ridge = Ridge(alpha=1.0)  # You can tune alpha to control regularization strength
ridge.fit(X, y)

# Output Ridge coefficients
print("Ridge coefficients:", ridge.coef_)
