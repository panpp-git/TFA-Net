import torch.nn as nn
import torch
from complexLayers import ComplexLinear,ComplexReLU,ComplexConv1d,ComplexConvTranspose1d,ComplexConv2d
import matplotlib.pyplot as plt


def set_layer1_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_layer1_ori(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)

    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net


class FrequencyRepresentationModule_layer1_ori(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=3):
        super().__init__()

        self.n_filters=n_filters
        self.inner=inner_dim
        self.n_layers=n_layers
        self.in_layer = ComplexConv1d(1,  inner_dim*n_filters, kernel_size=(1,31), padding=(0,31 // 2),
                                       bias=False)
        mod=[]
        for i in range(self.n_layers):
            tmp = []
            tmp += [
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size), padding=(kernel_size //2), bias=False,
                          padding_mode='circular'),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size), padding=(kernel_size // 2), bias=False,
                          padding_mode='circular'),
                nn.BatchNorm2d(n_filters),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(self.n_layers):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, (3,1), stride=(upsampling,1),
                                            padding=(1,0), output_padding=(1,0), bias=False)


    def forward(self, x):
        bsz = x.size(0)
        inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)
        inp=inp.view(bsz,1,1,-1)
        x=self.in_layer(inp).view(bsz, self.n_filters,self.inner, -1)

        # for i in range(0,16,1):
        #     plt.figure()
        #     plt.ion()
        #     plt.imshow(x[0,i,:,:].abs()/torch.max(x[0,i,:,:].abs()))
        #     plt.xticks([])
        #     plt.yticks([])

        x=x.abs()
        for i in range(self.n_layers):
            res_x = self.mod[i](x)
            x = res_x + x
            x = self.activate_layer[i](x)

        # for i in range(16):
        #     plt.figure()
        #     plt.ion()
        #     plt.imshow(x[0,i,:,:].abs())
        x = self.out_layer(x).squeeze(-3).transpose(1,2)
        return x