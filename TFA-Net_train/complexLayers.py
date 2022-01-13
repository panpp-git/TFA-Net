#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:30:02 2019
@author: Sebastien M. Popoff
Based on https://openreview.net/forum?id=H1T2hmZAb
"""

import torch
from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d,Conv1d,ReLU
from torch.nn import ConvTranspose2d,ConvTranspose1d
from complexFunctions import complex_relu, complex_max_pool2d, complex_avg_pool2d
from complexFunctions import complex_dropout, complex_dropout2d
import torch.nn as nn
import torch



def apply_complex(fr, fi, input):
    return (fr(input.real) - fi(input.imag)).type(torch.complex64) \
           + 1j * (fr(input.imag) + fi(input.real)).type(torch.complex64)

def apply_complex2(fr, fi, input):

    inp_real,inp_imag=torch.chunk(input,2,1)
    ret_real=(fr(inp_real) - fi(inp_imag))
    ret_imag=(fr(inp_imag) + fi(inp_real))
    return ret_real,ret_imag

class ComplexDropout(Module):
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            return complex_dropout(input, self.p)
        else:
            return input


class ComplexDropout2d(Module):
    def __init__(self, p=0.5):
        super(ComplexDropout2d, self).__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            return complex_dropout2d(input, self.p)
        else:
            return input

class Shrink(Module):
    def __init__(self,in_channel,out_channel):
        super(Shrink,self).__init__()
        self.l1=Linear(in_channel,out_channel//4)
        self.l2=BatchNorm1d(out_channel//4)
        self.l3=ReLU()
        self.l4=Linear(out_channel//4,out_channel)
        self.l5=nn.Sigmoid()

    def forward(self,residual):
        abs_mean=torch.mean(residual,dim=[2,3])
        scales=self.l1(abs_mean)
        scales=self.l2(scales)
        scales=self.l3(scales)
        scales=self.l4(scales)
        scales=self.l5(scales)
        scales=torch.unsqueeze(scales,-1)
        scales=torch.unsqueeze(scales,-1)
        abs_mean=torch.unsqueeze(abs_mean,-1)
        abs_mean = torch.unsqueeze(abs_mean, -1)
        thres=abs_mean*scales
        res_sign=torch.sign(residual)
        residual_max=torch.maximum(torch.abs(residual)-thres,torch.zeros_like(residual))
        residual=res_sign*residual_max
        return residual

class ComplexMaxPool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input):
        return complex_max_pool2d(input, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, ceil_mode=self.ceil_mode,
                                  return_indices=self.return_indices)



class ComplexAvgPool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super(ComplexAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input):
        return complex_avg_pool2d(input, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, ceil_mode=self.ceil_mode,
                                  return_indices=self.return_indices)







class ComplexConvTranspose2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ComplexConvTranspose2d, self).__init__()

        self.conv_tran_r = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)

    def forward(self, input):
        return apply_complex4(self.conv_tran_r, self.conv_tran_i, input)

class ComplexConvTranspose1d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ComplexConvTranspose1d, self).__init__()

        self.conv_tran_r = ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)

    def forward(self, input):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, input)





def apply_complex4(fr, fi, input):

    inp_real,inp_imag=torch.chunk(input,2,1)
    ret_real=(fr(inp_real) - fi(inp_imag))
    ret_imag=(fr(inp_imag) + fi(inp_real))
    ret=torch.cat((ret_real,ret_imag),1)
    return ret




class ComplexConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=1, padding=(0,0),
                 dilation=1, groups=1, bias=True,padding_mode='zero'):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,padding_mode=padding_mode)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,padding_mode=padding_mode)

    def forward(self, input):
        return apply_complex4(self.conv_r, self.conv_i, input)




class ComplexConv1d(Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super(ComplexConv1d, self).__init__()
        self.conv_r = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return apply_complex4(self.conv_r, self.conv_i, input)




class ComplexLinear(Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features, bias=False)
        self.fc_i = Linear(in_features, out_features, bias=False)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)


class NaiveComplexBatchNorm1d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, \
                 track_running_stats=True):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        return self.bn_r(input.real).type(torch.complex64) + 1j * self.bn_i(input.imag).type(torch.complex64)


class NaiveComplexBatchNorm2d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(NaiveComplexBatchNorm2d, self).__init__()
        self.bn_r = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        input_r,input_i=torch.chunk(input,2,1)
        ret_r=self.bn_r(input_r)
        ret_i=self.bn_i(input_i)
        ret=torch.cat((ret_r,ret_i),1)
        return ret
        # return self.bn_r(input.real).type(torch.complex64) + 1j * self.bn_i(input.imag).type(torch.complex64)


class _ComplexBatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            # self.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.complex64))
            self.register_buffer('running_mean_r', torch.zeros(num_features, dtype=torch.float))
            self.register_buffer('running_mean_i', torch.zeros(num_features, dtype=torch.float))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean_r', None)
            self.register_parameter('running_mean_i', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean_r.zero_()
            self.running_mean_r.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input):
        exponential_average_factor = 0.0
        input_r,input_i=torch.chunk(input,2,1)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            # mean_r = input.real.mean([0, 2, 3]).type(torch.complex64)
            # mean_i = input.imag.mean([0, 2, 3]).type(torch.complex64)
            # mean = mean_r + 1j * mean_i

            mean_r = input_r.mean([0, 2, 3]).type(torch.float)
            mean_i = input_i.mean([0, 2, 3]).type(torch.float)


        else:
            mean_r = self.running_mean_r
            mean_i=self.running_mean_i

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                # self.running_mean = exponential_average_factor * mean \
                #                     + (1 - exponential_average_factor) * self.running_mean
                self.running_mean_r=exponential_average_factor * mean_r \
                                     + (1 - exponential_average_factor) * self.running_mean_r
                self.running_mean_i=exponential_average_factor * mean_i \
                                     + (1 - exponential_average_factor) * self.running_mean_i

        # input = input - mean[None, :, None, None]
        input_r = input_r - mean_r[None, :, None, None]
        input_i = input_i - mean_i[None, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = 1. / n * input_r.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input_i.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        # input = (Rrr[None, :, None, None] * input.real + Rri[None, :, None, None] * input.imag).type(torch.complex64) \
        #         + 1j * (Rii[None, :, None, None] * input.imag + Rri[None, :, None, None] * input.real).type(
        #     torch.complex64)

        input_r=(Rrr[None, :, None, None] * input_r + Rri[None, :, None, None] * input_i)
        input_i = (Rii[None, :, None, None] * input_i + Rri[None, :, None, None] * input_r)

        if self.affine:
            # input = (self.weight[None, :, 0, None, None] * input.real + self.weight[None, :, 2, None,
            #                                                             None] * input.imag + self.bias[None, :, 0, None, None]).type(torch.complex64) \
            #         + 1j * (self.weight[None, :, 2, None, None] * input.real + self.weight[None, :, 1, None,
            #                                                                    None] * input.imag + self.bias[None, :, 1, None, None]).type(torch.complex64)
            input_r=(self.weight[None, :, 0, None, None] * input_r + self.weight[None, :, 2, None,
                                                                        None] * input_i + self.bias[None, :, 0, None, None])
            input_i=(self.weight[None, :, 2, None, None] * input_r + self.weight[None, :, 1, None,
                                                                               None] * input_i + self.bias[None, :, 1, None, None])

        return torch.cat((input_r,input_i),1)


class ComplexBatchNorm1d(_ComplexBatchNorm):

    def forward(self, input):

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            mean_r = input.real.mean(dim=0).type(torch.complex64)
            mean_i = input.imag.mean(dim=0).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, ...]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = input.real.var(dim=0, unbiased=False) + self.eps
            Cii = input.imag.var(dim=0, unbiased=False) + self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=0)
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_covar[:, 0]

            self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_covar[:, 1]

            self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                       + (1 - exponential_average_factor) * self.running_covar[:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None, :] * input.real + Rri[None, :] * input.imag).type(torch.complex64) \
                + 1j * (Rii[None, :] * input.imag + Rri[None, :] * input.real).type(torch.complex64)

        if self.affine:
            input = (self.weight[None, :, 0] * input.real + self.weight[None, :, 2] * input.imag + self.bias[None, :, 0]).type(torch.complex64) \
                    + 1j * (self.weight[None, :, 2] * input.real + self.weight[None, :, 1] * input.imag + self.bias[None, :, 1]).type(torch.complex64)

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return input
