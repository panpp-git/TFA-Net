import numpy as np
from sympy import *

def freq2fr(f1,f2, xgrid, kernel_type='gaussian', param=None, r=None,nfreq=None,sig_dim=None,theta=None,amp=None,sigpos=None,siglen=None,f3=None):
    if kernel_type == 'gaussian':
        return gaussian_kernel_simplify(f1,f2, xgrid, param, r,nfreq,sig_dim,theta,amp,sigpos,siglen,f3)



def gaussian_kernel_simplify(f1,f2, xgrid, sigma, r,nfreq,sig_dim,theta,amp,sigpos,siglen,f3):
    t=Symbol('t')
    t1=np.linspace(0,1,sig_dim,endpoint=False)
    fr = np.zeros((f1.shape[0], sig_dim,xgrid.shape[0]))
    wgt_factor = np.ones((f1.shape[0], sig_dim, xgrid.shape[0]))

    abs_max=-1
    for n in range(fr.shape[0]):
        tmp_max=np.max(amp[n,:nfreq[n]])
        if tmp_max>abs_max:
            abs_max=tmp_max
        abs_max=20 * np.log10(10*abs_max + 1)
    for n in range(fr.shape[0]):

        amp[n, :nfreq[n]] = 20 * np.log10(10*amp[n, :nfreq[n]] + 1)
        for i in range(nfreq[n]):
            x11 =  r[n,i] * cos(2 * np.pi * (f1[n,i] * t + f2[n,i] * t**2) + theta[n,i])+f3[n,i]*t
            ph = diff(x11,t)
            func=lambdify(t,ph,'numpy')
            ph1=func(t1)
            ph1=ph1-np.floor(ph1/sig_dim)*sig_dim
            idx1=np.floor(ph1).astype('int')


            fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
                n, i]]+1,256)] = np.maximum(amp[n, i]/6,fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
                n, i]]+1,256)])
            fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
                n, i]]-1,256)] = np.maximum(amp[n, i]/6,fr[n, range(sig_dim)[sigpos[n, i]:sigpos[n, i] + siglen[n, i]], np.mod(idx1[sigpos[n, i]:sigpos[n, i] + siglen[
                n, i]]-1,256)])


            fr[n,range(sig_dim)[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],idx1[sigpos[n,i]:sigpos[n,i]+siglen[n,i]]] =np.maximum(fr[n,range(sig_dim)[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],idx1[sigpos[n,i]:sigpos[n,i]+siglen[n,i]]],amp[n,i])
            wgt_factor[n,range(sig_dim)[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],idx1[sigpos[n,i]:sigpos[n,i]+siglen[n,i]]]=np.maximum(wgt_factor[n,range(sig_dim)[sigpos[n,i]:sigpos[n,i]+siglen[n,i]],idx1[sigpos[n,i]:sigpos[n,i]+siglen[n,i]]],np.power(abs_max/amp[n,i],2))


    return fr,wgt_factor







