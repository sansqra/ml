import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def synthesize_data(w, rows, b):
	# Generate random values for X, Y from normal distribution
	X = np.random.normal(0, 1, (rows, len(w)))
	y = np.dot(X, w) + b
	return X, y

def split_data(X, y):
	# Split data into training set and test data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
	return X_train, X_test, y_train, y_test


def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = x.shape[0] # number of samples

    # initial theta
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])

    # total error, J(theta)
    J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])

    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0/m * sum([(t0 + t1*x[i] - y[i]) for i in range(m)]) 
        grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)])

        # update the theta_temp
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1
    
        # update theta
        t0 = temp0
        t1 = temp1

        # mean squared error
        e = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] ) 

        if abs(J-e) <= ep:
            print('Converged, iterations: ', iter, '!!!')
            converged = True
    
        J = e   # update error 
        iter += 1  # update iter
    
        if iter == max_iter:
            print('Max interactions exceeded!')
            converged = True

    return t0,t1



np.random.seed(5)
X, y = synthesize_data([1,2], 2, 2.2)
x_train, x_test, y_train, y_test = split_data(X, y)
gradient_descent(0.01, x_train, y_train)