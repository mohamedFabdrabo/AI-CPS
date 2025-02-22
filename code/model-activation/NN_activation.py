import tensorflow as tf
import pandas as pd
import os

def load_model(model_path):
    """
    Loads a TensorFlow model from the specified directory.

    Parameters:
    - model_path (str): Path to the TensorFlow model file (.h5).

    Returns:
    - model (tensorflow.keras.Model): Loaded TensorFlow model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from: {model_path}")
    return model

def load_activation_data(data_path):
    """
    Loads activation data from a CSV file. Assumes the file has a single row for prediction.

    Parameters:
    - data_path (str): Path to the activation data file (.csv).

    Returns:
    - data (pd.DataFrame): Loaded activation data as a DataFrame.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Activation data file not found: {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"Activation data loaded successfully from: {data_path}")
    return data

def predict(model, data):
    """
    Uses the TensorFlow model to make predictions on the provided data.

    Parameters:
    - model (tensorflow.keras.Model): The loaded TensorFlow model.
    - data (pd.DataFrame): The input data for prediction.

    Returns:
    - predictions (numpy.ndarray): Model predictions.
    """
    predictions = model.predict(data)
    return predictions

if __name__ == "__main__":
    # Specify paths
    model_path = "tmp/knowledgeBase/currentAiSolution.keras"  # Path to the TensorFlow model
    activation_data_path = "tmp/activationBase/activation_data.csv"  # Path to the activation data file

    # Load the model and activation data
    try:
        model = load_model(model_path)
        activation_data = load_activation_data(activation_data_path)

        # Ensure the activation data has only one row
        if len(activation_data) != 1:
            raise ValueError("Activation data must contain exactly one data point.")

        # Make predictions
        # if the correct results is included in the file
        answer = None
        # if the correct result is included
        if(activation_data.shape[1] == 4):
            result =int(round(predict(model, activation_data)[0][0]))
        # if the correct result is not included in the file
        elif(activation_data.shape[1] == 5):
            answer = activation_data['P'][0]
            #remove the last column (Y)
            activation_data = activation_data.iloc[:, :-1]
            result =int(round(predict(model, activation_data)[0][0]))
        else:
            raise ValueError("Activation data must contain exactly 4 features without the Predicted.")
        
        # Display the results
        print(f"Neural Network Prediction result: {result} , The actual Value: {answer}")
    except Exception as e:
        print(f"Error: {e}")
