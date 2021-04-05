import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

plt.style.use('seaborn')

x = []
y = []

index = count()

rand = random.random()

cnt = 4


def animate(i):
    x.append(next(index))
    # a = random.randint(0, 200)
    global cnt
    cnt += 1
    y.append((random.random() * random.randint(-5,5)) + cnt)
    plt.cla()

    if len(x) == 50:
        x.pop(0)
        y.pop(0)

    plt.plot(x, y, '-')


ani = FuncAnimation(plt.gcf(), animate, interval=500)

plt.tight_layout()
plt.show()
