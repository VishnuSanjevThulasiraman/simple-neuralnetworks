import numpy as np
import matplotlib.pyplot as plt
from perceptron import sigmoid

tr_input = np.array([[0,0],
                            [0,1],
                            [1,0],
                            [1,1]])

output = np.array([[0,1,1,0]]).T

np.random.seed(1)      #generating pseudo random numbers to debug easily

w = 2 * np.random.random((2, 1)) - 1
b = 2 * np.random.random((4, 1)) - 1

lst10 = []   #lists to generate matplot
lst20 = []   #each list is to store
lst30 = []   #the values of the changing_output
lst40 = []   #after each iteration

for i in range(10000): #200 is the epochs you can change the values here to see how the
                     #accuracy of the network increases as you increase the value of epochs

    input = tr_input
    changing_output = sigmoid.value(np.dot(input, w) + b )

    err = changing_output - output
    change = err * sigmoid.dydx(changing_output)

    lst10.append(err[0])
    lst20.append(err[1])
    lst30.append(err[2])
    lst40.append(err[3])

    b -= 0.2 * err
    w -= 0.2 * np.dot(input.T, change)

print(' weights = ')
print(w)
print(' bias = ')
print(b)
print(' output = ')
print(changing_output)
plt.plot(lst10)      #to view how the value changes over each iteration
plt.plot(lst20)
plt.plot(lst30)

#print(change)
plt.plot(lst40)
plt.title('Changing value of errors vs #iterations')
#plt.savefig('fig1-tsv.png')
plt.show()
#print(change)
