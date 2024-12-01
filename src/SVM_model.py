import numpy as np
import cvxpy as cp 



def train_svm(x_train, y_train, gammas):
    """
    Train an SVM using the hinge loss with slack variables and an L2 regularization term.

    Parameters:
        x_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training labels, with values -1 or 1.
        gammas (list of floats): Regularization parameters to test.

    Returns:
        results (list of tuples): A list of tuples, each containing:
            - gamma (float): Regularization parameter.
            - prob_value (float): Objective value after optimization.
            - a (numpy.ndarray): Weight vector.
            - b (float): Bias term.
    """
    
    m = x_train.shape[0]  #number of data pts
    n = x_train.shape[1]  #number of features per data pt

    #decision variables
    a = cp.Variable(n) #wieght vector
    b = cp.Variable() #offset
    eta = cp.Variable(m) #slack vector

    #hyperparameter, gamma
    gamma = cp.Parameter(nonneg=True)
    

    #constraints
    constraints = [ y_train[i] * (x_train[i, :] @ a - b) >= 1 - eta[i] for i in range(m) ]
    constraints += [eta >= 0]

    #objective function
    objective = cp.Minimize( cp.norm(a, 2) + gamma * cp.norm(eta, 1) )

    #formulate the problem
    prob = cp.Problem(objective, constraints)

    #solve the problem

    #lists to store results from loop 
    results = []
    
    #loop to iterate through each gamma
    for g in gammas:

        gamma.value = g
        prob.solve()
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Solver failed for gamma={g}")
        results.append((g, prob.value, a.value, b.value))

    return results