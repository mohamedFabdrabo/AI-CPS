import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load your dataset
df = pd.read_csv('/home/ubuntu/AI-CPS/data/processed/your_data.csv')  # replace with your actual dataset path

# Define independent variables (X) and dependent variable (y)
X = df[['GP', 'W', 'D', 'L', 'F', 'A', 'GD']]  # Example independent variables
X = sm.add_constant(X)  # Add constant to the model
y = df['P']  # Dependent variable

# Fit the OLS model
ols_model = sm.OLS(y, X).fit()

# Create the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X['GP'], y=y, color='orange', label='Training Data')
sns.scatterplot(x=X['GP'], y=y, color='blue', label='Testing Data')
plt.plot(X['GP'], ols_model.fittedvalues, color='red', label='OLS Fit Line')
plt.legend()
plt.title("OLS Scatter Plot")
plt.xlabel('GP')
plt.ylabel('P')
plt.savefig('scatter_plot.pdf')

# Create the box plot for the data
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Boxplot of Dataset Features")
plt.savefig('box_plot.pdf')

# Diagnostic Plots (e.g., residuals vs fitted)
sm.graphics.plot_regress_exog(ols_model, "GP")
plt.savefig('diagnostic_plots.pdf')

# Show plots
plt.show()

