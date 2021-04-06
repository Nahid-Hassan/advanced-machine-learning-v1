# https://youtu.be/sDv4f4s2SB8

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import numpy as np
from itertools import count

x = [.5, 2.3, 2.9]
y = [1.4, 1.9, 3.2]

plt.scatter(x, y, marker='o')

def sum_of_squared_residuals(x_data, y_data, intercept, slope):
    residuals = 0
    predicted = []

    # in_ =list(intercept())
    # print(in_)

    for x, y in zip(x_data, y_data):
        pred = (intercept + (slope * x))
        predicted.append(pred)
        residuals += (y - pred)

    return (residuals, predicted)


# sum_of_squared_residuals(x, y, intercept=0, slope=.64)
a = np.arange(-5, 5, .5)
index = iter(a)

x_val = np.arange(-1, 3, .1)
y_val = [sum_of_squared_residuals(x, y, intercept=c, slope=.64)[0] ** 2 for c in x_val]

def animate(i):
    plt.cla()

    plt.scatter(x, y)
    intercept = 0
    r, y_hat = sum_of_squared_residuals(x, y, intercept=intercept, slope=.64)
    print(abs(r), y_hat)
    l = 'r: ' + str(round(abs(r), 2)) + ', intercept: ' + str(abs(intercept))
    plt.plot(x, y_hat, label=l)
    plt.plot(x_val, y_val)
    plt.legend()


ani = FuncAnimation(plt.gcf(), animate, interval=500)
plt.show()
