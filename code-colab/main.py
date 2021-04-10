import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import math

x = np.arange(0, 1, .1)
print(x)

def softplus(x):
    return np.log(1 + np.exp(x))

w = -34.4
b = 2.14

def animate(i):
    plt.plot()

ani = FuncAnimation(plt.gcf(), animate, interval=500)

plt.legend()
plt.tight_layout()
plt.show()