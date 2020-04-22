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

tr_input = np.array([[0,1,0],
                            [1,1,0],
                            [1,0,1],
                            [0,1,1]])

tr_changing_output = np.array([[0,1,1,0]]).T # .T is to take transpose

np.random.seed(16)      #generating pseudo random numbers to debug easily

w = 2 * np.random.random((3, 1)) - 1

lst1 = []   #lists to generate matplot
lst2 = []   #each list is to store
lst3 = []   #the values of the changing_output
lst4 = []   #after each iteration

for i in range(200):

    input = tr_input
    changing_output = sigmoid.value(np.dot(input, w))

    err = changing_output - tr_changing_output
    change = err * sigmoid.dydx(changing_output)

    lst1.append(changing_output[0])
    lst2.append(changing_output[1])
    lst3.append(changing_output[2])
    lst4.append(changing_output[3])

    w += np.dot(input.T, change)

print('real_output after training')
print(changing_output)
plt.plot(lst1)      #to view how the value changes over each iteration
plt.plot(lst2)
plt.plot(lst3)
plt.plot(lst4)
plt.title('Changing value of outputs vs #iterations')
plt.savefig('fig1-tsv.png')
plt.show()
