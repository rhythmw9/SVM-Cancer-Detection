import numpy as np 
import matplotlib.pyplot as plt

def evaluate_and_plot(results, x_train, y_train, x_test, y_test):
    """
    Evaluate the SVM model and plot train/test errors against gamma.

    Parameters:
        results (list of tuples): List of results from `train_svm`, where each tuple contains:
            - gamma (float): Regularization parameter.
            - prob_value (float): Objective value after optimization.
            - a (numpy.ndarray): Optimized weight vector.
            - b (float): Optimized bias term.
        x_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training labels.
        x_test (numpy.ndarray): Testing feature matrix.
        y_test (numpy.ndarray): Testing labels.

    Returns:
        best_gamma (float): Best gamma value based on the optimization objective.
        best_a (numpy.ndarray): Weight vector for the best gamma.
        best_b (float): Bias term for the best gamma.
    """

    if not results:
        raise ValueError("The results list is empty. Ensure `train_svm` was executed correctly.")

    train_errors = []
    test_errors = []

    #extract gammas from results
    gammas = [r[0] for r in results]

    for g, _, a, b in results:
        #train error (0-1 loss)
        y_train_pred = np.sign(x_train @ a - b) #predicted output values using newly optimized values
        train_error = np.mean(y_train_pred != y_train) #proportion of training data points that were misclassified
        train_errors.append(train_error)
        
        #evaluating the test set and test error (0-1 loss)
        y_test_pred = np.sign(x_test @ a - b)
        test_error = np.mean(y_test_pred != y_test)
        test_errors.append(test_error)
        
        #print intermediate results
        print(f"Gamma: {g:.2f}, Train Error: {train_error:.2f}, Test Error: {test_error:.2f}")

    #plot train and test errors as a function of gamma
    plt.plot(gammas, train_errors, marker='o', label='Train Error')
    plt.plot(gammas, test_errors, marker='o', label='Test Error')
    plt.xscale('log')
    plt.xlabel('Gamma')
    plt.ylabel('0-1 Loss')
    plt.title('Train and Test Error vs Gamma')

    best_gamma, _, best_a, best_b = min(results, key=lambda x: x[1])
    plt.axvline(x=best_gamma, color='r', linestyle='--', label=f'Best Gamma: {best_gamma:.2f}') #highlight the best gamma with a vertical line
    
    plt.legend()
    plt.show()

    
    

    return best_gamma, best_a, best_b