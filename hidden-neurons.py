import numpy as np
import matplotlib.pyplot as plt
from perceptron import sigmoid

def hidden_neuron(input, output, n_hn, epochs):

    lst100 = []   #lists to generate matplot
    lst200 = []   #each list is to store
    lst300 = []   #the values of the c
    lst400 = []   #after each iteration
    lst_err0 = []

    np.random.seed(1)

    weights1 = 2 * np.random.random((2, n_hn)) - 1
    bias1 = 2 * np.random.random((1, n_hn)) - 1

    weights2 = 2 * np.random.random((n_hn, 1)) - 1
    bias2 = 2 * np.random.random((n_hn, 1)) - 1

    for i in range(epochs):

        hidden_neuron_output = sigmoid.value(np.dot(input, weights1) + bias1 )
        final_perceptron_output = sigmoid.value(np.dot(hidden_neuron_output, weights2) + bias2 )
        error = final_perceptron_output - output
        change = error * sigmoid.dydx(final_perceptron_output)

        e = error[0] + error[1] + error[2] + error[3]
        e /= 4

        bias1 -= 0.35 * e
        weights1 -= 0.35 * np.dot(input.T, change)

        bias2 -= 0.35 * e
        weights2 -= 0.35 * np.dot(hidden_neuron_output.T, change)

        lst100.append(hidden_neuron_output[0][0])
        lst200.append(hidden_neuron_output[1][0])
        lst300.append(hidden_neuron_output[2][0])
        lst400.append(hidden_neuron_output[3][0])
        lst_err0.append(e[0])



    print('mean error')
    print(e)
    print('bias1')
    print(bias1)
    print('hidden_neuron_output')
    print(hidden_neuron_output)
    print('error')
    print(error)
    plt.plot(lst100)      #to view how the value changes over each iteration
    plt.plot(lst200)
    plt.plot(lst300)
    plt.plot(lst400)
    plt.plot(lst_err0)
    plt.title('Changing value of errors vs #iterations')
    #plt.show()

tr_input = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])

output = np.array([[0,0,0,1]]).T

hidden_neuron(tr_input, output, 2, 100)
