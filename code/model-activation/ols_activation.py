import pandas as pd
import os
import pickle
import statsmodels.api as sm

def load_activation_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Activation data file not found: {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"Activation data loaded successfully from: {data_path}")
    return data

if __name__ == "__main__":
    # Specify paths
    model_path = "tmp/knowledgeBase/ols_model.pkl"  # Path to the OLS model model
    activation_data_path = "tmp/activationBase/activation_data.csv"  # Path to the activation data file

    # Load the model and activation data
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        activation_data = load_activation_data(activation_data_path)
        # Ensure the activation data has only one row
        if len(activation_data) != 1:
            raise ValueError("Activation data must contain exactly one data point.")

        # Make predictions
        # if the correct results is included in the file
        answer = None
        # if the correct result is included
        if(activation_data.shape[1] == 4):
            activation_data = sm.add_constant(activation_data)
            result = model.predict(activation_data)
        # if the correct result is not included in the file
        elif(activation_data.shape[1] == 5):
            answer = activation_data['P'][0]
            #remove the last column (Y)
            activation_data = activation_data.iloc[:, :-1]
            # activation_data = sm.add_constant(activation_data)
            activation_data.insert(0, 'const', 1)  # Insert a column of ones at the beginning
            result = model.predict(activation_data)
        else:
            raise ValueError("Activation data must contain exactly 4 features without the Predicted.")
        
        # Display the results
        print(f"OLS model Prediction result: {int(result[0])} , The actual Value: {answer}")
    except Exception as e:
        print(f"Error: {e}")
