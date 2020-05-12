"""
Developed by
T Sanjev Vishnu

IIT Bombay, Mumbai
India

Coimbatore,
India

"""

import numpy as np
import matplotlib.pyplot as plt

class sigmoid(object):

    def value(x):                        #value of the sigmoid function

        return 1 / (1+np.exp(-x))

    def dydx(x):                        # derivative of the sigmoid function

        return x * (1-x)

tr_input = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])

tr_changing_output = np.array([[1,1,1,0]]).T # .T is to take transpose

np.random.seed(16)      #generating pseudo random numbers to debug easily

w = 2 * np.random.random((2, 1)) - 1

lst1 = []   #lists to generate matplot
lst2 = []   #each list is to store
lst3 = []   #the values of the changing_output
lst4 = []   #after each iteration

for i in range(200): #200 is the epochs you can change the values here to see how the
                     #accuracy of the network increases as you increase the value of epochs

    input = tr_input
    changing_output = sigmoid.value(np.dot(input, w))

    err = tr_changing_output - changing_output
    change = err * sigmoid.dydx(changing_output)

    lst1.append(changing_output[0])
    lst2.append(changing_output[1])
    lst3.append(changing_output[2])
    lst4.append(changing_output[3])

    w += np.dot(input.T, change)

#print('real_output after training')

print(changing_output)
"""
plt.plot(lst1)      #to view how the value changes over each iteration
plt.plot(lst2)
plt.plot(lst3)


This entire snippet is commented since when i call this script as package
these lines get executed

#print(change)
plt.plot(lst4)
plt.title('Changing value of outputs vs #iterations')
#plt.savefig('fig1-tsv.png')
#plt.show()
"""
