import numpy as np
import torch


def frequency_generator(f, nf, min_sep, dist_distribution):
    if dist_distribution == 'random':
        random_freq(f, nf, min_sep)
    elif dist_distribution == 'jittered':
        jittered_freq(f, nf, min_sep)
    elif dist_distribution == 'normal':
        normal_freq(f, nf, min_sep)


def random_freq(f, nf, min_sep):
    """
    Generate frequencies uniformly.
    """
    for i in range(nf):
        f_new = np.random.rand() - 1 / 2
        condition = True
        while condition:
            f_new = np.random.rand() - 1 / 2
            condition = (np.min(np.abs(f - f_new)) < min_sep) or \
                        (np.min(np.abs((f - 1) - f_new)) < min_sep) or \
                        (np.min(np.abs((f + 1) - f_new)) < min_sep)
        f[i] = f_new


def jittered_freq(f, nf, min_sep, jit=1):
    """
    Generate jittered frequencies.
    """
    l, r = -0.5, 0.5 - nf * min_sep * (1 + jit)
    s = l + np.random.rand() * (r - l)
    c = np.cumsum(min_sep * (np.ones(nf) + np.random.rand(nf) * jit))
    f[:nf] = (s + c - min_sep + 0.5) % 1 - 0.5


def normal_freq(f, nf, min_sep, scale=0.05):
    """
    Distance between two frequencies follows a normal distribution
    """
    f[0] = np.random.uniform() - 0.5
    for i in range(1, nf):
        condition = True
        while condition:
            d = np.random.normal(scale=scale)
            f_new = (d + np.sign(d) * min_sep + f[i - 1] + 0.5) % 1 - 0.5
            condition = (np.min(np.abs(f - f_new)) < min_sep) or \
                        (np.min(np.abs((f - 1) - f_new)) < min_sep) or \
                        (np.min(np.abs((f + 1) - f_new)) < min_sep)
        f[i] = f_new


def amplitude_generation(dim, amplitude, floor_amplitude=0.1):
    """
    Generate the amplitude associated with each frequency.
    """
    if amplitude == 'uniform':
        return np.random.rand(*dim) * (15 - floor_amplitude) + floor_amplitude
    elif amplitude == 'normal':
        return np.abs(np.random.randn(*dim))
    elif amplitude == 'normal_floor':
        return 1.5*np.abs(np.random.randn(*dim)) + floor_amplitude
    elif amplitude == 'alternating':
        return np.random.rand(*dim) * 0.5 + 20 * np.random.rand(*dim) * np.random.randint(0, 2, size=dim)

def amplitude_freq_generation(dim,floor_amplitude=0.1):
    dotm=np.random.rand(*dim) * (6 - 0.01) + 0.01
    f1=(np.random.rand(*dim)-0.5) * 8
    f2=(np.random.rand(*dim)-0.5) * 2
    amp=dotm/f1
    return f1,f2,amp



def gen_signal(num_samples, signal_dim, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False):
    s = np.zeros((num_samples, 2, signal_dim))

    xgrid=np.linspace(0,1,signal_dim,endpoint=False)[:, None]
    f1,f2,r = amplitude_freq_generation((num_samples, num_freq), floor_amplitude)

    amp = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)

    theta = np.random.rand(num_samples, num_freq) * 2 * np.pi

    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq
    for n in range(num_samples):
        if n%100==0:
            print(n)
        # x1 = exp(2j * pi * amp1 * cos(2 * pi * (w1 * t1 + w11 * t1. ^ 2) + theta1))
        for i in range(nfreq[n]):
            sin=amp[n,i]*np.exp(2j*np.pi*r[n,i]*np.cos(2*np.pi*(f1[n,i]*xgrid.T+f2[n,i]*np.power(xgrid.T,2))+theta[n,i]))
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag
        s[n] = s[n] / (np.sqrt(np.mean(np.power(s[n], 2)))+1e-13)

    return s.astype('float32'), f1.astype('float32'), f2.astype('float32'),nfreq,r,theta.astype('float32')

