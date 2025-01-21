import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import os

def remove_outliers(data:pd.DataFrame):
    ############ Removing the outliers ####################
    print("############ Removing the outliers (IQR method) ####################")
    filtered_data = data
    for column in data.columns:
        # calculate IQR for column
        initial_size = filtered_data.shape
        Q1 = np.percentile(data[column], 25, method='midpoint')
        Q3 = np.percentile(data[column], 75, method='midpoint')
        IQR = Q3 - Q1
        print(f"for Column Name: {column}: The IQR_{column} = IQR")
        # to define the upper and lower bounds for the data column
        max = Q3 + 1.5*IQR
        min = Q1 - 1.5*IQR
        print(f"The lower limit for {column}  = {max}")
        print(f"The upper limit for {column}  = {min}")

        # filter the data using the IQR
        filtered_data = filtered_data[(filtered_data[column] >= min) & (filtered_data[column] <= max)]
        print("The shape of the data Before :  " , initial_size)
        print("The shape of the data After  :  ", filtered_data.shape)
    return filtered_data

def remove_nulls(data):
    # remove the rows with null values
    print("############ Removing the null values ####################")
    initial_size = data.shape
    data.dropna(inplace=True)
    print("The shape of the rawdata: " , initial_size)
    print("The shape of the data after removing nulls: ",data.shape)
    return data

def spliting_train_test_sets(data):
    # now let's split the data into train and test datasets
    print("############ splitting into train & test sets ####################")
    train_set = data.sample(frac=0.8, random_state=42)
    test_set = data.drop(train_set.index)

    return (train_set , test_set)

def save_train_test_sets(train, test, train_path, test_path):
    # save the data into csv files
    # Save to CSV files
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

def create_directory_structure(base_path = 'tmp'):
    """
    Ensure that the necessary directories exist in the file system.
    """
    os.makedirs(os.path.join(base_path, "learningBase"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "learningBase/train"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "learningBase/validation"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "activationBase"), exist_ok=True)    
    os.makedirs(os.path.join(base_path, "knowledgeBase"), exist_ok=True)

if __name__ == '__main__':
    # create the requrired file system:
    create_directory_structure('tmp')
    raw_data_path = 'tmp/learningBase/joint_data_collection.csv'
    train_path = 'tmp/learningBase/train/training_data.csv'
    test_path  = 'tmp/learningBase/validation/test_data.csv'
    activation_path = 'tmp/activationBase/activation_data.csv'

    # read the raw data
    rawdata = pd.read_csv(raw_data_path)
    features_of_interest = ['GP', 'F', 'A', 'GD','P']
    rawdata = rawdata[features_of_interest]
    data = remove_nulls(rawdata)
    data = remove_outliers(data)
    (train_set, test_set) = spliting_train_test_sets(data)
    print(" ... Data splitted successfully :) ... ")
    try:
        save_train_test_sets(train_set, test_set, train_path, test_path)
        # activation_data.csv from a random sample of test_set
        activation_data = test_set.sample(n=1)
        # remove the result from the activation data (last row)
        #activation_data = activation_data.iloc[:, :-1]
        activation_data.to_csv(activation_path, index=False)

        print("DONE: ALL Data saved")
    except Exception as e:
        print(f"Error saving the train and testing set : {e}")

