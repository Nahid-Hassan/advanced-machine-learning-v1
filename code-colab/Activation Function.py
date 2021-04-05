'''
Machine Learning Activation Function

** Softmax: f(x) = 1 / 1 + exp(-x)
** Softplus: f(x) = log(1 + exp(-x))
** Relu: f(x) = max(0, x)
'''

from math import exp, log, tanh
from numpy import arange
import matplotlib.pyplot as plt


class Activation:

    def __init__(self):
        pass

    def softmax(self, x):
        return 1 / (1 + exp(-x))

    def relu(self, x):
        return max(0, x)

    def softplus(self, x):
        return log(1 + exp(x))

    def tanh(self, x):
        return tanh(x)

    def leaky_relu(self, x):
        return max(.1*x, x)


# input data generate
input_data = arange(-100, 100, .5)

# create activation class object
a = Activation()

# create plot for different activation function
plt.plot(input_data, list(map(a.softmax, input_data)), label='Softmax')
plt.plot(input_data, list(map(a.relu, input_data)), label='Relu')
plt.plot(input_data, list(map(a.softplus, input_data)), label='Softplus')
plt.plot(input_data, list(map(a.leaky_relu, input_data)), label='Leaky Relu')
plt.plot(input_data, list(map(a.tanh, input_data)), label='Tanh')


plt.xlabel('Input Data -10 to 10')
plt.ylabel('Output')

plt.legend()

plt.show()
