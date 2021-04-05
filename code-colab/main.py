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
