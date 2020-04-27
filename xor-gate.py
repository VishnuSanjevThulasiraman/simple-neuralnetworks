import numpy as np
import matplotlib.pyplot as plt
from perceptron import sigmoid

tr_input = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])

output = np.array([[1,1,1,0]]).T

np.random.seed(16)      #generating pseudo random numbers to debug easily

w = 2 * np.random.random((2, 1)) - 1
b = 2 * np.random.random((1, 1)) - 1

lst10 = []   #lists to generate matplot
lst20 = []   #each list is to store
lst30 = []   #the values of the c
lst40 = []   #after each iteration
lst_err = []

for i in range(10000): #200 is the epochs you can change the values here to see how the
                     #accuracy of the network increases as you increase the value of epochs

    input = tr_input
    c = sigmoid.value(np.dot(input, w) + b )

    err = c - output
    change = err * sigmoid.dydx(c)
    e = err[0]+ err[1] + err[2] + err[3]
    e = e / 4

    lst10.append(c[0])
    lst20.append(c[1])
    lst30.append(c[2])
    lst40.append(c[3])
    lst_err.append(e)

    b -= 0.2 * e
    w -= 0.2 * np.dot(input.T, change)

print(' weights = ')
print(w)
print(' bias = ')
print(b)
print(' output = ')
print(c)
plt.plot(lst10)      #to view how the value changes over each iteration
plt.plot(lst20)
plt.plot(lst30)

#print(change)
plt.plot(lst40)
plt.plot(lst_err)
plt.title('Changing value of errors vs #iterations')
#plt.savefig('fig1-tsv.png')
plt.show()
#print(change)
