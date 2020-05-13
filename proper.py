import numpy as np
import matplotlib.pyplot as plt

class sigmoid(object):

    def value(x):                        #value of the sigmoid function

        return 1 / (1+np.exp(-x))

    def dydx(x):                        # derivative of the sigmoid function

        return x * (1-x)

np.random.seed(16)
A = np.array([[0,0],[0,1],[1,1],[1,0]]).T
B = np.array([1,1,0,1]).T

np.random.seed(16)      #generating pseudo random numbers to debug easily

W = 2 * np.random.random((2, 1)) - 1
b = 2 * np.random.random((1, 1)) - 1

c = []

def cost(A,B,W,b):
    C = sigmoid.value(np.dot(W.T,A)+b)
    E = C - B
    SE = np.square(E)
    MSE = 0.25 * np.sum(SE)

    return(MSE)

def backpropogation(A,B,W,b,epochs,lr): #lr is the learning rate

    for i in range(epochs):
        C = sigmoid.value(np.dot(W.T,A)+b)
        E = C - B
        sigmoid_derivatives = sigmoid.dydx(C)
        Y = E * sigmoid_derivatives
        YEt = np.dot(Y,A.T).T
        W -= lr * YEt
        b -= lr * np.sum(Y)
        c.append(cost(A,B,W,b))


    print(C)
    plt.plot(c)
    plt.show()

backpropogation(A,B,W,b,10000,0.2)
""""
print(W.shape)
print(W)
print(A.shape)
print(A)
print(B.shape)
print(B)
"""
