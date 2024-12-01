import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#defining variables, parameters, and hyperperameter for SVM
#also loading and splitting data set

def load_and_preprocess_data(filepath):
    """
    Load and preprocess the breast cancer dataset.

    Parameters:
        filepath (str): Path to the dataset file.

    Returns:
        x_train (numpy.ndarray): Preprocessed training features.
        x_test (numpy.ndarray): Preprocessed testing features.
        y_train (numpy.ndarray): Training labels.
        y_test (numpy.ndarray): Testing labels.
    """

    df = pd.read_csv(filepath, header=None) #load data into a dataframe


    # Replace '?' with NaN, drop rows with missing values, covert data set to floats
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(float)

    df = df.iloc[:, 1:] #remove the first column that is not important for the model

    #split data set into feature and output vectors and convert to numpy array
    x = df.iloc[:, :-1].values #feature vector
    y = df.iloc[:, -1].values #output vector

    #standardize y values: 2 -> -1 (benign), 4 -> 1 (malignant)
    y = np.where(y == 2, -1, 1)  

    #split data into training and test sets, 80%/20% respectively (for now)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    #normalize the train/test feature vectors
    scalar = StandardScaler()
    x_train = scalar.fit_transform(x_train) #fit and transform the training data
    x_test = scalar.transform(x_test) #transform test data with the same scalar

    return x_train, x_test, y_train, y_test