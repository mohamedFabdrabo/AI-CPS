import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def create_experiment_directory(base_dir="experiments"):
    """
    Creates a new experiment directory in the specified base directory.
    The new directory will have the name 'experimentXX', where XX is the next available number.

    Parameters:
    - base_dir (str): The base directory where experiments are stored.

    Returns:
    - str: The path of the newly created experiment directory.
    """
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Create the new experiment directory
    new_experiment_dir = os.path.join(base_dir, 'OLS_results')
    os.makedirs(new_experiment_dir, exist_ok=True)

    print(f"New experiment directory created: {new_experiment_dir}")
    return new_experiment_dir

def plot_residuals(y_test, y_pred, output_path):
    """
    Plots residuals (actual - predicted).

    Parameters:
    - y_test: True values.
    - y_pred: Predicted values.
    - output_path: Full path to save the plot.
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Residual')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_predicted_vs_actual(y_test, y_pred, output_path):
    """
    Plots predicted values vs. actual values.

    Parameters:
    - y_test: True values.
    - y_pred: Predicted values.
    - output_path: Full path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def create_directory_structure(base_path):
    """
    Ensure that the necessary directories exist in the file system.
    """
    os.makedirs(os.path.join(base_path, "learningBase"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "knowledgeBase"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "activationBase"), exist_ok=True)
    os.makedirs(os.path.join('tmp', "knowledgeBase"), exist_ok=True)

def save_metrics(model,y_test, y_pred, output_path):
    """
    Save training metrics and evaluation results to a file.

    Parameters:
    - model: The OLS model.
    - y_test: True test values.
    - y_pred: Predicted test values.
    - output_path: Full path to save the metrics file.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    with open(output_path, "w") as f:
        f.write("Training Metrics:\n")
        f.write(f"OLS Model: {model.summary()}\n")
        f.write("Test Metrics:\n")
        f.write(f"Mean Absolute Error: {mae:.4f}\n")
        f.write(f"Mean Squared Error: {mse:.4f}\n")

def plot_model_performance(X_test, y_test, y_pred, output_path):
    """
    Plots scatter plots to visualize model performance against each feature in X_test.
    
    Parameters:
    X_test (pd.DataFrame): Test feature dataset.
    y_test (pd.Series or np.ndarray): Actual target values.
    y_pred (np.ndarray): Predicted target values.
    """
    feature_names = X_test.columns
    num_features = X_test.shape[1]
    fig, axes = plt.subplots(nrows=num_features, figsize=(6, num_features * 4))
    
    if num_features == 1:
        axes = [axes]  # Ensure axes is iterable for single feature case
    
    for i, feature in enumerate(feature_names):
        axes[i].scatter(X_test[feature], y_test, label="Actual", alpha=0.6)
        axes[i].scatter(X_test[feature], y_pred, label="Predicted", alpha=0.6)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Target Value")
        axes[i].legend()
        axes[i].set_title(f"Performance against {feature}")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# main method and script
if __name__ == '__main__':
    # read the processed cleaned data
    train_path = 'tmp/learningBase/train/training_data.csv'
    test_path = 'tmp/learningBase/validation/test_data.csv'

    train_data = pd.read_csv(train_path)
    test_data  = pd.read_csv(test_path)

    X_train, X_test, y_train, y_test = train_data.iloc[:, :-1],test_data.iloc[:, :-1], train_data['P'], test_data['P']
    # Add a constant to the independent variables matrix (for the intercept)
    X_train = sm.add_constant(X_train)
    X_test  = sm.add_constant(X_test)
    
    # create coorelation matrix:
    correlation_matrix = train_data.corr()
    print(correlation_matrix)

    # Build the OLS model
    model = sm.OLS(y_train, X_train).fit() 
    # Summary to verify the architecture
    print("Model created: \n" , model.summary())
    
    
    y_pred_ols = model.predict(X_test)
    print("\n ------ OLS model Performance: --------")
    print("MAE:", mean_absolute_error(y_test, y_pred_ols))
    print("MSE:", mean_squared_error(y_test, y_pred_ols))


    # saving training and testing data
    print("Saving training and testing information: ")
    # create the base path 
    base_path = create_experiment_directory()

    # Save visualizations and metrics
    plot_residuals(y_test, y_pred_ols, f"{base_path}/residual_plot.png")
    plot_predicted_vs_actual(y_test, y_pred_ols, f"{base_path}/predicted_vs_actual.png")
    save_metrics(model, y_test, y_pred_ols, f"{base_path}/training_metrics.txt")
    plot_model_performance(test_data.iloc[:, :-1], y_test, y_pred_ols, f"{base_path}/FeatureByFeaturePred.png")
    # Save the model
    # Save the model to a file
    with open(f"{base_path}/ols_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("data/model-file/ols_model.pkl", "wb") as f:
        pickle.dump(model, f)

