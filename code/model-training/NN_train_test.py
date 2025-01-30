import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
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

    # Get a list of all existing experiment directories
    existing_experiments = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("experiment")
    ]

    # Extract the numbers from existing experiment directories
    experiment_numbers = [
        int(d.replace("experiment", "").zfill(4)) for d in existing_experiments if d.replace("experiment", "").isdigit()
    ]

    # Determine the next experiment number
    next_experiment_number = max(experiment_numbers, default=0) + 1
    next_experiment_name = f"experiment{str(next_experiment_number).zfill(4)}"

    # Create the new experiment directory
    new_experiment_dir = os.path.join(base_dir, next_experiment_name)
    os.makedirs(new_experiment_dir, exist_ok=True)

    print(f"New experiment directory created: {new_experiment_dir}")
    return new_experiment_dir

def create_nn_model(input_dim, hidden_layers=[64, 32], activation='relu', optimizer='adam', loss='mse', metrics=['mae']):
    """
    Create a neural network model for regression.

    Parameters:
    - input_dim (int): Number of input features.
    - hidden_layers (list): List of integers representing the number of neurons in each hidden layer.
    - activation (str): Activation function for the hidden layers.
    - optimizer (str or tf.keras.optimizers): Optimizer to use for training.
    - loss (str): Loss function to use.
    - metrics (list): List of metrics to evaluate during training.

    Returns:
    - model (tf.keras.Model): Compiled neural network model.
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(hidden_layers[0], activation=activation, input_dim=input_dim))
    
    # Hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation=activation))
    
    # Output layer
    model.add(Dense(1))  # Single output for regression
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model

def plot_training_curves(history, output_path):
    """
    Plots training and validation loss curves.

    Parameters:
    - history: The history object returned by model.fit() in TensorFlow.
    - output_path: Full path to save the plot.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    #plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()  # Ensure plots don’t remain open in the Docker environment

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

def save_metrics(history, y_test, y_pred, output_path):
    """
    Save training metrics and evaluation results to a file.

    Parameters:
    - history: The history object from model.fit().
    - y_test: True test values.
    - y_pred: Predicted test values.
    - output_path: Full path to save the metrics file.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    with open(output_path, "w") as f:
        f.write("Training Metrics:\n")
        f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
        #f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n\n")
        f.write("Test Metrics:\n")
        f.write(f"Mean Absolute Error: {mae:.4f}\n")
        f.write(f"Mean Squared Error: {mse:.4f}\n")

# main method and script
if __name__ == '__main__':
    # read the processed cleaned data
    train_path = 'tmp/learningBase/train/training_data.csv'
    test_path = 'tmp/learningBase/validation/test_data.csv'

    train_data = pd.read_csv(train_path)
    test_data  = pd.read_csv(test_path)

    X_train, X_test, y_train, y_test = train_data.iloc[:, :-1],test_data.iloc[:, :-1], train_data['P'], test_data['P']
    # Assuming the input dimension is 4 (GP, F, A, GD) in this data set
    input_dim = 4
    # Create the model with customizable parameters
    nn_model = create_nn_model(
        input_dim=input_dim,
        hidden_layers=[1024],  # Customize hidden layers
        activation='softplus',           # Activation function
        optimizer=Adam(learning_rate=0.0001),  # Custom optimizer
        loss='mse',                  # Loss function
        metrics=['mae']              # Metrics for evaluation
    )
    # Summary to verify the architecture
    print("Model created: \n" , nn_model.summary())
    
    # train the model using the processed data

    nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = nn_model.fit(X_train, y_train, epochs=100,batch_size=64, verbose=1)
    print('Model trained successfully with epochs: {100}')
    # Predict and evaluate NN
    y_pred_nn = nn_model.predict(X_test).flatten()
    print("\n ------ Neural Network Performance: --------")
    print("MAE:", mean_absolute_error(y_test, y_pred_nn))
    print("MSE:", mean_squared_error(y_test, y_pred_nn))


    # saving training and testing data
    print("Saving training and testing information: ")
    # create the base path 
    base_path = create_experiment_directory('experiments')

    # Save visualizations and metrics
    plot_training_curves(history, f"{base_path}/training_validation_loss.png")
    plot_residuals(y_test, y_pred_nn, f"{base_path}/residual_plot.png")
    plot_predicted_vs_actual(y_test, y_pred_nn, f"{base_path}/predicted_vs_actual.png")
    save_metrics(history, y_test, y_pred_nn, f"{base_path}/training_metrics.txt")

    # Save the model
    nn_model.save(f"{base_path}/currentAiSolution.keras")
    nn_model.save(f"data/model-file/currentAiSolution.keras")

