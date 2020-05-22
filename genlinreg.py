# Superivised Learning - Linear Regression (multivariate)

import numpy as np
import matplotlib.pyplot as plt

X = np.matrix([
        [1,2,3,4],
        [1,5,6,7],
        [1,8,9,10]
    ])
Y = np.array([99,88,77])

theta = np.zeros((4, 1))

def param_guess():
    global theta
    alpha = 0.1
    i = 0
    while i < 500:
        for i in range(len(theta)):
            theta[i] = theta[i] - (alpha * (1/len(Y)) * sum((theta * X) - Y))
        i += 1

if __name__ == '__main__':
    plt.scatter(X,Y)
    plt.show()
    param_guess()
    xx = list(float(input('=>> ')))
    print(theta * xx)

