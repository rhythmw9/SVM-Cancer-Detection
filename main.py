from data_preprocessing import load_and_preprocess_data
from SVM_model import train_svm
from evaluation import evaluate_and_plot
import os 

def main():
    """
    Main script for training an SVM on the breast cancer dataset.

    Workflow:
        1. Load and preprocess the data.
        2. Train the SVM with different gamma values.
        3. Evaluate and plot training and testing errors.
        4. Print the best gamma value and model parameters.
    """

    filepath = "breast-cancer-wisconsin.data"

     #check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file '{filepath}' does not exist. Please check the path.")

    # Load and preprocess data
    x_train, x_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Define gamma values
    gammas = [0.01, 0.1, 0.5, 1, 5, 10, 50]

    # Train SVM
    results = train_svm(x_train, y_train, gammas)

    # Evaluate and plot results
    best_gamma, best_a, best_b = evaluate_and_plot(results, x_train, y_train, x_test, y_test)

    # Print final results
    print(f"Best gamma: {best_gamma:.2f}")

if __name__ == "__main__":
    main()