import numpy as np



def amplitude_generation(dim, amplitude, floor_amplitude=0.1):
    """
    Generate the amplitude associated with each frequency.
    """
    if amplitude == 'uniform':
        return np.random.rand(*dim) * (15 - floor_amplitude) + floor_amplitude
    elif amplitude == 'normal':
        return np.abs(np.random.randn(*dim))
    elif amplitude == 'normal_floor':
        return 5*np.abs(np.random.randn(*dim)) + floor_amplitude
    elif amplitude == 'alternating':
        return np.random.rand(*dim) * 0.5 + 20 * np.random.rand(*dim) * np.random.randint(0, 2, size=dim)

def amplitude_freq_generation(dim,floor_amplitude=0.1):

    dotm=np.random.rand(*dim) * (6 - 0.01) + 0.01
    f1=(np.random.rand(*dim)-0.5) * 12
    f2=(np.random.rand(*dim)-0.5) * 8
    amp=dotm/f1
    f3 = (np.random.rand(*dim) - 0.5) * 300
    return f1,f2,amp,f3


def gen_signal(num_samples, signal_dim, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False):
    s = np.zeros((num_samples, 2, signal_dim))

    xgrid=np.linspace(0,1,signal_dim,endpoint=False)[:, None]
    f1,f2,r,f3 = amplitude_freq_generation((num_samples, num_freq), floor_amplitude)

    amp = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)

    theta = np.random.rand(num_samples, num_freq) * 2 * np.pi
    start_pos = np.floor(np.random.rand(num_samples, num_freq) * (signal_dim-256)).astype(int)
    sig_len_max = signal_dim - start_pos
    sig_len = np.random.randint(256, sig_len_max+1)


    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype='int') * num_freq
    for n in range(num_samples):
        if n%100==0:
            print(n)
        # if n>=100 and n<120:
        #     nfreq[n]=3
        #     r[n,0:3]=1
        for i in range(nfreq[n]):

            xgrid_t = np.zeros((signal_dim, 1)).astype('complex128')
            xgrid_t[start_pos[n, i]:start_pos[n, i] + sig_len[n, i], 0] = xgrid[start_pos[n, i]:start_pos[n, i] + sig_len[n, i], 0]
            sin=amp[n,i]*np.exp(2j*np.pi*r[n,i]*np.cos(2*np.pi*(f1[n,i]*xgrid_t.T+f2[n,i]*np.power(xgrid_t.T,2))+theta[n,i])+2j*np.pi*f3[n,i]*xgrid_t.T)
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag

        s[n] = s[n] / (np.sqrt(np.mean(np.power(s[n], 2)))+1e-13)

    return s.astype('float32'), f1.astype('float32'), f2.astype('float32'),nfreq,r,theta.astype('float32'),amp.astype('float32'),start_pos.astype('int'),sig_len.astype('int'),f3.astype('float32')




