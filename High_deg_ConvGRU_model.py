import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class W_ConvGRU(nn.Module):
    def __init__(self, hidden=2, conv_kernel_size=3, input_channle=1, R=True):
        super(W_ConvGRU, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.hidden_channel = hidden
        self.R = R
        self.conv_kernel_size = conv_kernel_size
        self.input_channel = input_channle
        self.padding = math.floor(conv_kernel_size / 2)
        self.build_model()

    def get_parameter(self, shape, init_method='xavier'):
        param = Parameter(torch.Tensor(*shape).cuda())
        if init_method == 'xavier':
            nn.init.xavier_uniform_(param)
        elif init_method == 'zero':
            nn.init.constant_(param, 0)
        else:
            raise Exception('init method error')
        return param

    def build_model(self):
        input_to_state_shape = [
            self.hidden_channel,
            self.input_channel,
            self.conv_kernel_size,
            self.conv_kernel_size
        ]
        state_to_state_shape = [
            self.hidden_channel,
            self.hidden_channel,
            self.conv_kernel_size,
            self.conv_kernel_size
        ]
        state_bias_shape = [
            1, self.hidden_channel, 1, 1
        ]

        self.w_xz = self.get_parameter(input_to_state_shape)
        self.w_hz = self.get_parameter(state_to_state_shape)
        self.w_xr = self.get_parameter(input_to_state_shape)
        self.w_hr = self.get_parameter(state_to_state_shape)
        self.w_xh = self.get_parameter(input_to_state_shape)
        self.w_hh = self.get_parameter(state_to_state_shape)

        self.b_z = self.get_parameter(state_bias_shape, 'zero')
        self.b_r = self.get_parameter(state_bias_shape, 'zero')
        self.b_h_ = self.get_parameter(state_bias_shape, 'zero')

    def forward(self, x_t, hidden):
        h_tm1 = hidden
        Z = self.sigmoid(
            F.conv2d(x_t, self.w_xz, bias=None, padding=self.padding)
            + F.conv2d(h_tm1, self.w_hz, bias=None, padding=self.padding)
            + self.b_z
        )
        if not self.R:
            R = 1
        else:
            R = self.sigmoid(
                F.conv2d(x_t, self.w_xr, bias=None, padding=self.padding)
                + F.conv2d(h_tm1, self.w_hr, bias=None, padding=self.padding)
                + self.b_r
            )
        H_ = self.tanh(
            F.conv2d(x_t, self.w_xh, bias=None, padding=self.padding)
            + R * F.conv2d(h_tm1, self.w_hh, bias=None, padding=self.padding)
            + self.b_h_
        )
        H = (1-Z)*H_ + Z*h_tm1

        return H
