import torch.nn as nn
import torch
from complexLayers import ComplexLinear,ComplexReLU,ComplexConv1d,ComplexConvTranspose1d,ComplexConv2d,bmrelu
from rbn import RepresentativeBatchNorm2d
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

def set_skip_module(args):
    """
    Create a frequency-representation module
    """
    net = None

    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_skiplayer32(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)
    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()

    return net





class FrequencyRepresentationModule_skiplayer32(nn.Module):
    # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_160-skip-conn.pth'
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=25):
        super().__init__()

        self.n_filters=n_filters
        self.inner=inner_dim
        self.n_layers=n_layers
        self.in_layer = ComplexConv2d(1,  inner_dim*8, kernel_size=(1, 15), padding=(0, 15 // 2),
                                       bias=False)
        self.in_layer2=ComplexConv2d(8, n_filters, kernel_size=(kernel_size,1), padding=(kernel_size //2,0), bias=False)
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
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, (kernel_size,1), stride=(upsampling,1),
                                            padding=(upsampling//2,0), output_padding=(1,0), bias=False)


    def forward(self, x):
        bsz = x.size(0)
        inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)
        inp=inp.view(bsz,1,1,-1)
        x=self.in_layer(inp).view(bsz, 8,self.inner, -1)
        x=self.in_layer2(x)
        x=x.abs()

        # for i in range(16):
        #     plt.figure()
        #     plt.ion()
        #     plt.imshow(x[0,i])
            # plt.show()
            # plt.pause(1)
        for i in range(self.n_layers):
            res_x = self.mod[i](x)
            x = res_x + x
            x = self.activate_layer[i](x)

        # for i in range(16):
        #     plt.figure()
        #     plt.ion()
        #     plt.imshow(x[0,i])
            # plt.show()
            # plt.pause(1)
        x = self.out_layer(x).squeeze(-3).transpose(1,2)
        return x





class FrequencyRepresentationModule_layer1_ori(nn.Module):
    # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_160-skip-conn.pth'
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=25):
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


class FrequencyRepresentationModule_layer0(nn.Module):
    # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_160-skip-conn.pth'
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=25):
        super().__init__()

        self.n_filters=n_filters
        self.inner=inner_dim
        self.n_layers=n_layers
        self.in_layer = ComplexConv1d(1,  inner_dim*n_filters, kernel_size=(1,31), padding=(0,31 // 2),
                                       bias=False)
        self.in_layer2 = ComplexConv2d(n_filters, n_filters, kernel_size=(kernel_size, 1),
                                       padding=(kernel_size // 2, 0),
                                       bias=False)
        mod=[]
        for i in range(self.n_layers):
            tmp = []
            tmp += [
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size,1), padding=(kernel_size //2,0), bias=False,
                          padding_mode='circular'),
                nn.Conv2d(n_filters, n_filters, kernel_size=(1, kernel_size), padding=(0, kernel_size //2), bias=False,
                          padding_mode='circular'),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size,1), padding=(kernel_size // 2,0), bias=False,
                          padding_mode='circular'),
                nn.Conv2d(n_filters, n_filters, kernel_size=(1,kernel_size), padding=(0,kernel_size // 2), bias=False,
                          padding_mode='circular'),
                nn.BatchNorm2d(n_filters),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(self.n_layers):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, (kernel_out,1), stride=(upsampling,1),
                                            padding=(1,0), output_padding=(1,0), bias=False)


    def forward(self, x):
        bsz = x.size(0)
        inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)
        inp=inp.view(bsz,1,1,-1)
        x=self.in_layer(inp).view(bsz, self.n_filters,self.inner, -1)
        x=self.in_layer2(x)
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

class FrequencyRepresentationModule_layer1(nn.Module):
    # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_160-skip-conn.pth'
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=25):
        super().__init__()

        self.n_filters=n_filters
        self.inner=inner_dim
        self.n_layers=n_layers
        self.in_layer = ComplexConv1d(1,  inner_dim*n_filters, kernel_size=(1,31), padding=(0,31 // 2),
                                       bias=False)
        self.in_layer2 = ComplexConv2d(n_filters, n_filters, kernel_size=(kernel_size, 1),
                                       padding=(kernel_size // 2, 0),
                                       bias=False)
        mod=[]
        for i in range(self.n_layers):
            tmp = []
            tmp += [
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size,1), padding=(kernel_size //2,0), bias=False,
                          padding_mode='circular'),
                nn.Conv2d(n_filters, n_filters, kernel_size=(1, kernel_size), padding=(0, kernel_size //2), bias=False,
                          padding_mode='circular'),

                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size,1), padding=(kernel_size // 2,0), bias=False,
                          padding_mode='circular'),
                nn.Conv2d(n_filters, n_filters, kernel_size=(1,kernel_size), padding=(0,kernel_size // 2), bias=False,
                          padding_mode='circular'),

                nn.BatchNorm2d(n_filters),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(self.n_layers):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, (kernel_out,1), stride=(upsampling,1),
                                            padding=(1,0), output_padding=(1,0), bias=False)


    def forward(self, x):
        bsz = x.size(0)
        inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)
        inp=inp.view(bsz,1,1,-1)
        x=self.in_layer(inp).view(bsz, self.n_filters,self.inner, -1)
        x=self.in_layer2(x)
        # for i in range(0,16,1):
        #     plt.figure()
        #     plt.ion()
        #     plt.imshow(x[0,i,:,:].abs()/torch.max(x[0,i,:,:].abs()))
        #     plt.xticks([])
        #     plt.yticks([])

        x=x.abs()
        x=20 * torch.log(x+1).to(x.device) / torch.log(torch.Tensor([10])).to(x.device)
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

class FrequencyRepresentationModule_layer2(nn.Module):
    # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_160-skip-conn.pth'
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=25):
        super().__init__()

        self.n_filters=n_filters
        self.inner=inner_dim
        self.n_layers=n_layers
        self.in_layer = ComplexConv2d(1,  inner_dim*n_filters, kernel_size=(1, 31), padding=(0, 31 // 2),
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
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, (kernel_out,1), stride=(upsampling,1),
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

class FrequencyRepresentationModule_layer3(nn.Module):
    # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_160-skip-conn.pth'
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=25):
        super().__init__()


        self.n_filters=n_filters
        self.inner=inner_dim
        self.n_layers=n_layers
        self.in_layer = ComplexConv1d(1,  inner_dim*n_filters, kernel_size=(1, 31), padding=(0, 31 // 2),
                                       bias=False)
        # self.in_layer2 = ComplexConv2d(n_filters, n_filters, kernel_size=(31, 1),
        #                                padding=(31 // 2, 0),
        #                                bias=False)
        mod=[]
        for i in range(self.n_layers):
            tmp = []
            tmp += [
                # nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size), padding=(kernel_size //2), bias=False,
                #           padding_mode='circular'),
                nn.BatchNorm2d(n_filters),
                bmrelu(n_filters),
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size), padding=(kernel_size // 2), bias=False,
                          padding_mode='circular'),
                nn.BatchNorm2d(n_filters),
                bmrelu(n_filters),
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size), padding=(kernel_size // 2), bias=False,
                          padding_mode='circular'),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        self.bn_layer= nn.BatchNorm2d(n_filters)
        self.activate_layer=bmrelu(n_filters)
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, (kernel_out,1), stride=(upsampling,1),
                                            padding=(1,0), output_padding=(1,0), bias=False)


    def forward(self, x):
        bsz = x.size(0)
        inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)
        inp=inp.view(bsz,1,1,-1)
        x=self.in_layer(inp).view(bsz, self.n_filters,self.inner, -1)
        # x=self.in_layer2(x)

        # for i in range(0,20,1):
        #     plt.figure()
        #     plt.ion()
        #     plt.imshow(x[0,i,:,:].abs()/torch.max(x[0,i,:,:].abs()))
        #     plt.xticks([])
        #     plt.yticks([])

        x=x.abs()
        for i in range(self.n_layers):
            res_x = self.mod[i](x)
            x = res_x + x
        x=self.bn_layer(x)
        x=self.activate_layer(x)

        # for i in range(20):
        #     plt.figure()
        #     plt.ion()
        #     plt.imshow(x[0,i,:,:].abs())
        x = self.out_layer(x).squeeze(-3).transpose(1,2)
        return x

class FrequencyRepresentationModule_layer4(nn.Module):
    # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_160-skip-conn.pth'
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=25):
        super().__init__()


        self.n_filters=n_filters
        self.inner=inner_dim
        self.n_layers=n_layers
        self.in_layer = ComplexConv1d(1,  inner_dim*8, kernel_size=(1, 15), padding=(0, 15 // 2),
                                       bias=False)
        self.in_layer2 =ComplexConv2d(8, n_filters, kernel_size=(3, 1),
                                       padding=(3 // 2, 0),
                                       bias=False)
        mod=[]
        for i in range(self.n_layers):
            tmp = []
            tmp += [
                # nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size), padding=(kernel_size //2), bias=False,
                #           padding_mode='circular'),
                RepresentativeBatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size), padding=(kernel_size // 2), bias=False,
                          padding_mode='circular'),
                RepresentativeBatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size), padding=(kernel_size // 2), bias=False,
                          padding_mode='circular'),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        self.bn_layer= RepresentativeBatchNorm2d(n_filters)
        self.activate_layer=nn.ReLU()
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, (kernel_size,1), stride=(upsampling,1),
                                            padding=(1,0), output_padding=(1,0), bias=False)


    def forward(self, x):
        bsz = x.size(0)
        inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)
        inp=inp.view(bsz,1,1,-1)
        x=self.in_layer(inp).view(bsz, 8,self.inner, -1)


        # for i in range(0,16,1):
        #     plt.figure()
        #     plt.ion()
        #     plt.imshow(x[0,i,:,:].abs()/torch.max(x[0,i,:,:].abs()))
        #     plt.xticks([])
        #     plt.yticks([])
        x = self.in_layer2(x)
        x=x.abs()

        for i in range(self.n_layers):
            res_x = self.mod[i](x)
            x = res_x + x
        x=self.bn_layer(x)
        x=self.activate_layer(x)

        # for i in range(16):
        #     plt.figure()
        #     plt.ion()
        #     plt.imshow(x[0,i,:,:].abs())
        x = self.out_layer(x).squeeze(-3).transpose(1,2)
        return x

class FrequencyRepresentationModule_layer6(nn.Module):
    # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_160-skip-conn.pth'
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=25):
        super().__init__()

        self.n_filters=n_filters
        self.inner=inner_dim
        self.n_layers=n_layers
        self.in_layer = ComplexConv1d(1,  inner_dim*n_filters, kernel_size=(1,15), padding=(0,15 // 2),
                                       bias=False)
        # self.in_layer2 = ComplexConv2d(n_filters, n_filters, kernel_size=(kernel_size, 1),
        #                                padding=(kernel_size // 2, 0),
        #                                bias=False)
        mod=[]
        for i in range(self.n_layers):
            tmp = []
            tmp += [
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size,1), padding=(kernel_size //2,0), bias=False,
                          padding_mode='circular'),
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size,1), padding=(kernel_size //2,0), bias=False,
                          padding_mode='circular'),

                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size,kernel_size), padding=(kernel_size // 2,kernel_size // 2), bias=False,
                          padding_mode='circular'),
                nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size, kernel_size),
                          padding=(kernel_size // 2, kernel_size // 2), bias=False,
                          padding_mode='circular'),

                nn.BatchNorm2d(n_filters),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(self.n_layers):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, (kernel_size,1), stride=(upsampling,1),
                                            padding=(1,0), output_padding=(1,0), bias=False)


    def forward(self, x):
        bsz = x.size(0)
        inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)
        inp=inp.view(bsz,1,1,-1)
        x=self.in_layer(inp).view(bsz, self.n_filters,self.inner, -1)
        # x=self.in_layer2(x)
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

class FrequencyRepresentationModule_layer6(nn.Module):
    # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_160-skip-conn.pth'
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=25):
        super().__init__()

        self.n_filters=n_filters
        self.inner=inner_dim
        self.n_layers=n_layers
        self.n_filters2=n_filters*3
        self.in_layer1 = ComplexConv1d(1,  inner_dim*n_filters, kernel_size=(1,15), padding=(0,15 // 2),
                                       bias=False)
        # self.in_layer2 = ComplexConv2d(n_filters, n_filters, kernel_size=(kernel_size, 1),
        #                                padding=(kernel_size // 2, 0),
        #                                bias=False)
        self.in_layer2 = ComplexConv1d(1,  inner_dim*n_filters, kernel_size=(1,7), padding=(0,7 // 2),
                                       bias=False)
        self.in_layer3 = ComplexConv1d(1,  inner_dim*n_filters, kernel_size=(1,31), padding=(0,31 // 2),
                                       bias=False)
        mod=[]
        for i in range(self.n_layers):
            tmp = []
            tmp += [
                nn.Conv2d(self.n_filters2, self.n_filters2, kernel_size=(kernel_size,kernel_size), padding=(kernel_size // 2,kernel_size // 2), bias=False,
                          padding_mode='circular'),

                nn.BatchNorm2d(self.n_filters2),
                nn.ReLU(),
                nn.Conv2d(self.n_filters2, self.n_filters2, kernel_size=(kernel_size,kernel_size), padding=(kernel_size // 2,kernel_size // 2), bias=False,
                          padding_mode='circular'),


                nn.BatchNorm2d(self.n_filters2),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(self.n_layers):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)
        self.out_layer = nn.ConvTranspose2d(self.n_filters2, 1, (kernel_size,1), stride=(upsampling,1),
                                            padding=(1,0), output_padding=(1,0), bias=False)


    def forward(self, x):
        bsz = x.size(0)
        inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)
        inp=inp.view(bsz,1,1,-1)
        x1=self.in_layer1(inp).view(bsz, self.n_filters,self.inner, -1)
        x2 = self.in_layer2(inp).view(bsz, self.n_filters, self.inner, -1)
        x3 = self.in_layer3(inp).view(bsz, self.n_filters, self.inner, -1)
        x=torch.cat((x1,x2,x3),1)
        # x=self.in_layer2(x)
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