import random
import cmath
import math
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)


def my_function(x):
    s, p = x
    return math.sqrt(2 * s ** 2 + p ** 2 + 1) + math.exp(s ** 2 + 2 * p ** 2) - s - p


def optimize(function, error, step_size, starting_pt, return_values):
    random.seed()
    f0 = my_function(starting_pt)
    f1 = sys.float_info.max
    x01 = starting_pt[0]
    x02 = starting_pt[1]
    m = 0  # лічильник1
    M = 1  # лічильник2
    x_values = []
    while (abs(f1 - f0) > error):
        f0 = my_function([x01, x02])
        x1 = random.normalvariate(0, 1) * step_size + x01
        x2 = random.normalvariate(0, 1) * step_size + x02
        f1 = my_function([x1, x2])
        M = 4
        if f1 < f0:
            x01 = x1
            x02 = x2
            step_size = step_size * 1.5
        else:
            m += 1
            if m >= M:
                step_size = step_size / 2
                m = 0
        x_values.append([x1, x2])
    if return_values:
        return x_values
    return f1


results = optimize(my_function, 0.0001, 0.5, [0, 0], return_values=True)

print(results)
x = []
y = []
z = []

for item in results:
    x.append(item[0])
    y.append(item[1])
    z.append(my_function([item[0], item[1]]))

print("number of iterations: " + str(len(results)))
ax.scatter(x, y, z)
plt.show()

