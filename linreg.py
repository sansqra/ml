# Supervised Learning - Linear Regression (univariate)

import numpy as np
# import multiprocessing as mp
import matplotlib.pyplot as plt

# Linear Regression cannot fit for all types of data, hence the degree of polynomial
# i.e the resp hyperparamter will have to be selected accordingly, else answers cannot be predicted

x = np.array([1,2,3,4])  #setting x values
y = np.array([98, 2, 45, 12])   #setting corresponding y values to x

print(x[2], y[2]) # print a single tuple to check for accuracy

t1, t2 = 0, 0   # setting parameter values for hypothesis equation
alpha = 1   # learning rate set to one (proabably better ways to set alpha)
m = len(x)  # sample size

'''

hyp = t1 + t2*x - best fit line minimized over J(t1, t1)
 while !converged:
     ti = ti - alpha * 1/m * (sum(hyp(xi) - yi)| (-> 1 to m) )

h_t_(x) = t1 + t2 * x
J(t1, t2) = 1/2m * sum(h(x) - y)^2 | 1 to m

'''


def param_guess():
    global t1, t2

    i = 0

    while i < 90000:    # iterates 20,000 times because convergence hasn't been checked for

     t1 = t1 - (alpha * (1/m) * sum(((t1 + t2 * x) - y)))   
     t2 = t2 - (alpha * (1/m) * sum(((t1 + t2 * x) - y) * x))
     i += 1

    #  it is being assummed that after several iterations, parameters reach some minimum, although
    # although it may not be the best

if __name__ == '__main__':

    plt.scatter(x,y) # plotting a line graph of input against output to see any patterns
    plt.show()
    param_guess()
    xx = float(input('=>> '))
    print(t1 + t2 * xx) # prediction based on best fit line paramaters



