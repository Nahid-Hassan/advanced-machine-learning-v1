"""
Basic implementation of neural network!
# URL: https://youtu.be/CqOfi41LfDw?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1
"""

# import module
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.special import softmax


class NN:

    def __init__(self):
        pass

    # class method for loss calculation
    def activation_function(self, x, name='softmax'):
        if name == 'softmax':
            pass
        elif name == 'mse':
            pass
        elif name == 'relu':
            return max(0, x)
        elif name == 'softplus':
            return np.log(1 + np.exp(x))
        elif name == 'argmax':
            pass
        elif name == 'sigmoid':
            return np.exp(x) / (1 + np.exp(x))
        elif name == 'cross_entropy':
            pass
        elif name == 'tanh':
            return np.tanh(x)
        else:
            print("Not defined!")
    
    # matplotlib animated plot
    def animate(self, i):
        pass

def main():
	pass

if __name__ == '__main__':
	main()