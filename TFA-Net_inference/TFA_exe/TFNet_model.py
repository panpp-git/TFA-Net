import torch
import TFNet_util as util
import numpy as np
import h5py
import os
import scipy.io as sio

class Trainer:

    def __init__(self,data_path):
        skip_path = os.path.join('model', 'TFA-Net.pth')
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.skip_module, _, _, _, _ = util.load(skip_path, 'layer1',device)
        self.skip_module.cpu()
        self.skip_module.eval()
        self.data_path=data_path


    def inference(self):
        path = os.path.join(self.data_path, 'matlab_real2.h5')
        f = h5py.File(path, 'r')
        real_data2 = f['matlab_real2'][:]
        f.close()
        path = os.path.join(self.data_path, 'matlab_imag2.h5')
        f = h5py.File(path, 'r')
        imag_data2 = f['matlab_imag2'][:]
        f.close()
        path = os.path.join(self.data_path, 'bz.h5')
        f = h5py.File(path, 'r')
        bz = f['bz'][:]
        f.close()

        N=256
        signal_50dB2 = np.zeros([int(bz), 2, N]).astype(np.float32)
        signal_50dB2[:, 0,:] = (real_data2.astype(np.float32)).T
        signal_50dB2[:, 1,:] = (imag_data2.astype(np.float32)).T


        with torch.no_grad():
            fr_50dB2 = self.skip_module(torch.tensor(signal_50dB2))
            fr_50dB2 = fr_50dB2.cpu().data.numpy()
            path = os.path.join(self.data_path, 'data1_resfreq.mat')
            sio.savemat(path, {'data1_resfreq':fr_50dB2})






