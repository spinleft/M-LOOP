import numpy as np
import matplotlib.pyplot as plt


def SigmoidLike(startpoint, endpoint, xmax, sample_rate, params):
    amplitude = endpoint - startpoint
    x_step = 1.0 / sample_rate
    coef = np.array(params)
    t = np.array(np.arange(0, xmax, x_step)) / xmax
    params_num = len(params)
    ft = coef[0] * t
    for i in range(1, params_num):
        ft = (coef[i] + ft) * t
    wave = amplitude * (1 - np.exp(-ft)) / (1 + np.exp(-ft)) + startpoint
    wave = np.append(wave, endpoint)
    return wave


def TridiagSolve(a, b, c, f):
    # a = [a_1, ..., a_n]
    # b = [b_1, ..., b_{n-1}, 0]
    # c = [0, c_2, ..., c_n]
    # f = [f_1, ..., f_n]
    n = len(a)
    v = 0
    v = np.zeros(n+1)
    y = np.zeros(n+1)
    x = np.zeros(n+1)
    for k in range(0, n):
        u = a[k] - c[k] * v[k]
        v[k+1] = b[k] / u
        y[k+1] = (f[k] -c[k] * y[k]) / u
    for k in range(n, 0, -1):
        x[k-1] = y[k] - v[k] * x[k]
    return x[0:n]


def CubicSplineInterpolation(startpoint, endpoint, xmax, sample_rate, params):
    n = len(params) + 1
    h = xmax / float(n)
    y = np.append(startpoint, params)
    y = np.append(y, endpoint)
    A = np.diag([2.0] * (n+1))
    A[0, 1] = 1.0
    A[n, n-1] = 1.0
    for i in range(1, n):
        A[i, i+1] = 0.5
        A[i, i-1] = 0.5
    A = np.mat(A)
    d = np.zeros(n+1)
    d[0] = 6 * (y[1] - y[0]) / h**2
    d[n] = 6 * (y[n-1] - y[n]) / h**2
    for i in range(1, n):
        d[i] = 3 * (y[i-1] + y[i+1] - 2 * y[i]) / h**2
    M = np.array(np.dot(d,A.I.T))
    M = M.reshape(n+1)

    x_step = 1.0 / sample_rate
    x_samples = np.arange(0, xmax, x_step)
    sample_num = len(x_samples)
    x_i = np.append(np.array([i * h for i in range(0, n)]), xmax)
    index = 0
    wave = np.zeros(sample_num+1)
    for i in range(sample_num):
        if x_samples[i] >= x_i[index+1]:
            index += 1
        wave[i] = ((x_i[index+1] - x_samples[i])**3 * M[index] + (x_samples[i] - x_i[index])**3 * M[index+1]) / (6 * h)
        wave[i] += ((x_i[index+1] - x_samples[i]) * y[index] + (x_samples[i] - x_i[index]) * y[index+1]) / h
        wave[i] -= h * ((x_i[index+1] - x_samples[i]) * M[index] + (x_samples[i] - x_i[index]) * M[index+1]) / 6
    wave[0]= startpoint
    wave[sample_num] = endpoint
    return wave



if __name__ == '__main__':
    params = [1, 2, 3]
    startpoint = 0.0
    endpoint = 10.0
    t = 15.71
    sample_rate = 100
    tsp = np.append(np.arange(0.0, t, 1.0 / sample_rate), t)
    wave = CubicSplineInterpolation(startpoint, endpoint, t, sample_rate, params)
    plt.plot(tsp, wave)
    x = np.append(np.arange(0, t, t/(len(params))), t)
    y = np.append(startpoint, np.append(params[1:], endpoint))
    plt.plot(x, y, 'bo-')
    plt.show()

