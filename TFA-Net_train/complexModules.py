import torch.nn as nn
import torch
from complexLayers import ComplexConv1d
import matplotlib.pyplot as plt


def set_layer1_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_TFA_Net(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)

        #########uncomment for RED-Net training###############################################################
        # net = FrequencyRepresentationModule_RED_Net(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
        #                                     inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
        #                                     upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
        #                                     kernel_out=args.fr_kernel_out)


    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net



import math
class REDNet30(nn.Module):
    def __init__(self, num_layers=15, num_features=8):
        super(REDNet30, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features*2, num_features, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x
class REDNet30_stft(nn.Module):
    def __init__(self, num_layers=15, num_features=8):
        super(REDNet30_stft, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features*2, num_features*2, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features*2, num_features, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x

class FrequencyRepresentationModule_TFA_Net(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=3):
        super().__init__()

        self.n_filters = n_filters
        self.inner = inner_dim
        self.n_layers = n_layers

        self.in_layer1 = ComplexConv1d(1, inner_dim * n_filters, kernel_size=(1, 31), padding=(0, 31 // 2),
                                       bias=False)

        self.rednet=REDNet30(self.n_layers,num_features=n_filters)
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, (3, 1), stride=(upsampling, 1),
                                            padding=(1, 0), output_padding=(1, 0), bias=False)


    def forward(self, x):
        bsz = x.size(0)
        inp_real = x[:, 0, :].view(bsz, 1, 1, -1)
        inp_imag = x[:, 1, :].view(bsz, 1, 1, -1)
        inp = torch.cat((inp_real, inp_imag), 1)

        x1 = self.in_layer1(inp)
        xreal,ximag=torch.chunk(x1,2,1)
        xreal = xreal.view(bsz, self.n_filters, self.inner, -1)
        ximag = ximag.view(bsz, self.n_filters, self.inner, -1)

        x=torch.sqrt(torch.pow(xreal, 2) + torch.pow(ximag, 2))

        # for i in range(0,16,1):
        #     plt.figure()
        #     plt.ion()
        #     plt.imshow(x[0,i,:,:].abs()/torch.max(x[0,i,:,:].abs()))
        #     plt.xticks([])
        #     plt.yticks([])

        x=self.rednet(x)
        x = self.out_layer(x).squeeze(-3).transpose(1, 2)
        return x
class FrequencyRepresentationModule_RED_Net(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=2, kernel_out=3):
        super().__init__()

        self.n_filters = n_filters
        self.inner = inner_dim
        self.n_layers = n_layers
        self.in_layer=nn.Conv2d(4, n_filters, kernel_size=1)
        self.rednet=REDNet30_stft(self.n_layers,num_features=n_filters)
        self.out_layer = nn.ConvTranspose2d(n_filters, 1, (3, 1), stride=(upsampling, 1),
                                            padding=(1, 0), output_padding=(1, 0), bias=False)


    def forward(self, x):
        bsz = x.size(0)
        xreal = x[:, 0, :].view(bsz, 1, 1, -1)
        ximag = x[:, 1, :].view(bsz, 1, 1, -1)

        xreal2 = x[:, 2, :].view(bsz, 1, 1, -1)
        ximag2 = x[:, 3, :].view(bsz, 1, 1, -1)

        xreal3 = x[:, 4, :].view(bsz, 1, 1, -1)
        ximag3 = x[:, 5, :].view(bsz, 1, 1, -1)

        xreal4 = x[:, 6, :].view(bsz, 1, 1, -1)
        ximag4 = x[:, 7, :].view(bsz, 1, 1, -1)

        xreal = xreal.view(bsz, 1, self.inner, -1)
        ximag = ximag.view(bsz, 1, self.inner, -1)

        xreal2 = xreal2.view(bsz, 1, self.inner, -1)
        ximag2 = ximag2.view(bsz, 1, self.inner, -1)

        xreal3 = xreal3.view(bsz, 1, self.inner, -1)
        ximag3 = ximag3.view(bsz, 1, self.inner, -1)

        xreal4 = xreal4.view(bsz, 1, self.inner, -1)
        ximag4 = ximag4.view(bsz, 1, self.inner, -1)

        x=torch.sqrt(torch.pow(xreal, 2) + torch.pow(ximag, 2))
        x2 = torch.sqrt(torch.pow(xreal2, 2) + torch.pow(ximag2, 2))
        x3 = torch.sqrt(torch.pow(xreal3, 2) + torch.pow(ximag3, 2))
        x4 = torch.sqrt(torch.pow(xreal4, 2) + torch.pow(ximag4, 2))
        x=torch.cat((x,x2,x3,x4),1)
        x=self.in_layer(x)
        x=self.rednet(x)
        x = self.out_layer(x).squeeze(-3).transpose(1, 2)
        return x

