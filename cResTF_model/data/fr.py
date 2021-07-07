import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from sympy import *

def freq2fr(f1,f2, xgrid, kernel_type='gaussian', param=None, r=None,nfreq=None,sig_dim=None,theta=None):
    if kernel_type == 'gaussian':
        return gaussian_kernel(f1,f2, xgrid, param, r,nfreq,sig_dim,theta)


def gaussian_kernel(f1,f2, xgrid, sigma, r,nfreq,sig_dim,theta):

    t=Symbol('t')
    t1=np.linspace(0,1,sig_dim,endpoint=False)
    fr = np.zeros((f1.shape[0], sig_dim,xgrid.shape[0]))
    for n in range(fr.shape[0]):
        for i in range(nfreq[n]):
            x11 =  r[n,i] * cos(2 * np.pi * (f1[n,i] * t + f2[n,i] * t**2) + theta[n,i])
            ph = diff(x11,t)
            func=lambdify(t,ph,'numpy')
            ph1=func(t1)
            flag = 1
            while flag:
                flag = 0
                if np.sum((ph1 > sig_dim / 2)) > 0:
                    flag = 1
                    ph1[ph1 > sig_dim / 2]= ph1[ph1 > sig_dim / 2] - sig_dim

                if np.sum((ph1 < -sig_dim / 2)) > 0:
                    flag = 1
                    ph1[ph1 < -sig_dim / 2] = ph1[ph1 < -sig_dim / 2] + sig_dim


            idx1 = np.round((ph1 + sig_dim / 2) / (sig_dim / xgrid.shape[0])).astype('int')
            idx1[idx1 == xgrid.shape[0]] = 0
            fr[n,range(sig_dim), idx1] +=  1
    fr[fr>1]=1
    return fr


def triangle(f, xgrid, slope):
    """
    Create a frequency representation with a triangle kernel.
    """
    fr = np.zeros((f.shape[0], xgrid.shape[0]))
    for i in range(f.shape[1]):
        dist = np.abs(xgrid[None, :] - f[:, i][:, None])
        rdist = np.abs(xgrid[None, :] - (f[:, i][:, None] + 1))
        ldist = np.abs(xgrid[None, :] - (f[:, i][:, None] - 1))
        dist = np.minimum(dist, rdist, ldist)
        fr += np.clip(1 - slope * dist, 0, 1)
    return fr



