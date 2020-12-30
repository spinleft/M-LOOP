import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def save_params_to_file(filename, params):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, 'w') as out_file:
        for param in params:
            out_file.write('%.5f\n' % param)
        out_file.close()


def get_result_from_file(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    result = []
    while True:
        if os.path.exists(filename):
            break
        else:
            time.sleep(1)
    with open(filename, 'r') as in_file:
        for line in in_file:
            temp = line.strip('\n')
            if temp != '':
                result.append(float(temp))
        in_file.close()
    result = np.array(result)
    return result

def waveform(startpoint, endpoint, tf, sample_rate, params):
    num_params = len(params)
    a_1 = endpoint / startpoint - 1 - np.sum(params)
    coef = np.hstack((a_1, params))
    l = int((num_params + 1) / 2)
    t_step = 1.0 / sample_rate
    t = np.array(np.arange(0, tf, t_step)) / tf
    wave = np.ones(len(t))
    for i in range(l):
        wave = wave + coef[i] * np.power(t, i+1) + \
            coef[l+i] * np.log2(1 + (np.power(2, 2*i+3) - 1) * t) / (2 * i + 3)
    wave = startpoint * wave
    return wave

def waveform_linear(starpoint, endpoint, tf, sample_rate):
    t = np.arange(0, tf, 1 / sample_rate) / tf
    wave = starpoint + (endpoint - starpoint) * t
    return wave

def wave_interpolate(wave_old, tf, sample_rate_new):
    t_old = np.linspace(0, tf, len(wave_old))
    f = interpolate.interp1d(t_old, wave_old, kind='quadratic')
    t_step_new = 1. / sample_rate_new
    t_new = np.arange(0, tf, t_step_new)
    wave_new = f(t_new)
    return wave_new

def waveform_bezier(startpoint, endpoint, tf, sample_rate, t_scales, params):
    if len(params) != len(t_scales):
        print("len(params) != len(t_scales)")
        raise ValueError
    num_samples = len(params)
    # t_scales = np.cumprod(1. - params[:num_samples])
    # t_samples = (np.concatenate(([1], t_scales[:-1])) - t_scales).cumsum() * tf
    t_samples = t_scales * tf
    wave_samples = np.cumprod(params) * (startpoint - endpoint) + endpoint

    t_nodes = np.zeros(shape=(2 * num_samples + 1, ))
    wave_nodes = np.zeros(shape=(2 * num_samples + 1, ))
    t_nodes[0] = 0.
    t_nodes[-1] = tf
    wave_nodes[0] = startpoint
    wave_nodes[-1] = endpoint
    for i in range(num_samples - 1):
        t_nodes[2*i+1] = t_samples[i]
        t_nodes[2*i+2] = (t_samples[i] + t_samples[i+1]) / 2
        wave_nodes[2*i+1] = wave_samples[i]
        wave_nodes[2*i+2] = (wave_samples[i] + wave_samples[i+1]) / 2
    t_nodes[-2] = t_samples[-1]
    wave_nodes[-2] = wave_samples[-1]

    t_step = 1. / sample_rate
    t_fit = np.arange(0, tf, t_step)
    wave_fit = np.zeros(shape=(len(t_fit), ))
    index = np.zeros(shape=(num_samples + 1, ), dtype=int)
    for i in range(1, num_samples + 1):
        temp = int(t_nodes[2*i] // t_step)
        index[i] = temp + 1 if temp * t_step < t_nodes[2*i] else temp
    
    for i in range(num_samples):
        a = t_nodes[2*i] - 2 * t_nodes[2*i+1] + t_nodes[2*i+2]
        b = 2 * (t_nodes[2*i+1] - t_nodes[2*i])
        c = t_nodes[2*i] - t_fit[index[i]: index[i+1]]
        if abs(a) < 1e-15:
            temp = - c / b
        else:
            temp = (np.sqrt(b**2 - 4 * a * c) - b) / (2 * a)
        wave_fit[index[i]: index[i+1]] = (wave_nodes[2*i] - 2 * wave_nodes[2*i+1] + wave_nodes[2*i+2]) * temp**2 + 2 * (wave_nodes[2*i+1] - wave_nodes[2*i]) * temp + wave_nodes[2*i]
    return wave_fit

def plot_wave(startpoint, endpoint, tf, sample_rate, params):
    num_params = len(params)
    a_1 = endpoint / startpoint - 1 - np.sum(params)
    coef = np.hstack((a_1, params))
    l = int((num_params + 1) / 2)
    t_step = 1.0 / sample_rate
    t = np.array(np.arange(0, tf, t_step)) / tf
    wave = np.ones(len(t))
    for i in range(l):
        wave = wave + coef[i] * np.power(t, i+1) + \
            coef[l+i] * np.log2(1 + (np.power(2, 2*i+3) - 1) * t) / (2 * i + 3)
    wave = startpoint * wave

    plt.xlabel("t")
    plt.ylabel("wave")
    plt.plot(t * tf, wave)
    plt.show()