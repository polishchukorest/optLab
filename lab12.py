from scipy import optimize
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize

fig = plt.figure()
ax = Axes3D(fig)


def my_function(x):
    s, p = x
    return math.sqrt(2 * s ** 2 + p ** 2 + 1) + math.exp(s ** 2 + 2 * p ** 2) - s - p


def gradf(x):
    s, p = x
    gs = (2 * s) / math.sqrt(2 * s ** 2 + p ** 2 + 1) + 2 * s * math.exp(s ** 2 + 2 * p ** 2) - 1  # u-
    gp = p / math.sqrt(2 * s ** 2 + p ** 2 + 1) + 4 * p * math.exp(s ** 2 + 2 * p ** 2) - 1  # v-
    return np.asarray((gs, gp))


def conj_grad(function, gradient, starting_point, iterations, error, results):
    i = 0
    k = 0
    r = np.asarray(-gradient(starting_point))
    d = r
    x = starting_point
    sigma_new = np.dot(r.transpose(), r)
    sigma_0 = sigma_new

    while (i < iterations and sigma_new > error ** 2 * sigma_0):
        j = 0
        sigma_d = np.dot(d.transpose(), d)
        alfa = optimize.line_search(my_function, gradf, x, r)[0]
        x = x + alfa * d
        r = -gradf(x)
        sigma_old = sigma_new
        sigma_new = np.dot(r.transpose(), r)
        beta = sigma_new / sigma_old
        d = r + np.dot(beta, d)
        k += 1

        results.append(x)
        if k == iterations or np.dot(r.transpose(), d) <= 0:
            d = r
            k = 0
        i = i + 1


results = []

try:
    conj_grad(my_function, gradf, [0.7, 0.7], 10, 0.01, results)
    print("number of iterations: " + str(len(results)))
    for result in results:
        print("x= " + str(result))
        print("function =" + str(my_function(result)))
    func_eval = []
    for result in results:
        func_eval.append(my_function(result))
    x = []
    y = []
    for item in results:
        x.append(item[0])
        y.append(item[1])
    ax.scatter(x, y, func_eval, s=50)
    plt.show()

except OverflowError:
    print("overflow of gradient")
except RuntimeWarning:
    print("OverFlow")
