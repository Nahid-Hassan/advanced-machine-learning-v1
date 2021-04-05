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

"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count

actual = 6

posible_pred = list(range(1, 12, 1))

b = list(range(-12, 12, 1)) * 10

index = iter(b)

cnt = 0

def animate(i):
    x = [(actual - posible) * -1 for posible in posible_pred]
    y = [((actual - posible) * -1) ** 2 for posible in posible_pred]

    global cnt
    cnt = cnt + 1

    c = 0
    if cnt >= 13:
        c = next(index)
        c = c * -1
    else:
        c = next(index)
    a = []
    a.append(c)
    print(a)

    plt.cla()
    plt.plot(x, y, 'go', markevery=a)
    plt.plot(x, y)


ani = FuncAnimation(plt.gcf(), animate, interval=1000)
plt.tight_layout()
plt.show()
"""
