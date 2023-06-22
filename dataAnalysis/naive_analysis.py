import math as mt
import random as rd
import numpy as np
import matplotlib.pyplot as plt


g_BM = np.loadtxt("Magn_L10_B0-416_01.txt", dtype=float)
N_resamples = 10


def avg(array):
    sum = 0.0
    for i in range(len(array)):
        sum += array[i]
    return sum/float(len(array))


def x4_avg(array):
    sum = 0.0
    for i in range(len(array)):
        sum += array[i]**4
    return sum/float(len(array))


def x2_avg(array):
    sum = 0.0
    for i in range(len(array)):
        sum += array[i]**2
    return sum/float(len(array))


def E(array):
    val = x4_avg(array)/(3*(x2_avg(array))**2)
    return val


def resample(array):
    ln = len(array)
    q = np.empty(ln)
    for i in range(ln):
        i_ran = mt.floor(rd.random()*ln)
        q[i] = array[i_ran]
    return q


def sigma(array):
    if len(array) <= 1:
        return 0
    else:
        val = x2_avg(array)-(avg(array))**2
    return np.sqrt(val)


x = np.arange(0, N_resamples)
sigm = np.empty(N_resamples)
for count in range(1, N_resamples+1):
    E_array = np.empty(count)
    for i in range(count):
        E_array[i] = E(resample(g_BM))
    sigm[count-1] = sigma(E_array)


plt.plot(x, sigm, "x", markersize=3)
plt.show()