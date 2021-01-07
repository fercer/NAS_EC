import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class IdentityModule(nn.Conv2d):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, **kwargs):
        super(IdentityModule, self).__init__(output_channels, output_channels, kernel_size=stride, stride=stride, padding=padding, groups=output_channels)
        self.requires_grad_(False)
        self.offset = int((output_channels - input_channels) // 2)
        self.weight.data[:, :, :kernel_size, :kernel_size] = 1

    def forward(self, x):
        z_pad = torch.zeros([x.size(0), self.offset, x.size(2), x.size(3)], device=x.device, requires_grad=False)
        x = torch.cat((z_pad, x, z_pad), dim=1)
        return super(IdentityModule, self).forward(x)


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, intermediate_channels=None, activation_functions=nn.ReLU, kernel_sizes=3, strides=1, paddings=1, bias=False, batch_normalization=False, dropout=0.0):
        super(ConvBlock, self).__init__()

        if intermediate_channels is None:
            intermediate_channels = []

        intermediate_channels.append(output_channels)

        # Number of convolution layers
        n_convlayers = len(intermediate_channels)

        if not isinstance(activation_functions, list):
            activation_functions = [activation_functions] * n_convlayers

        if isinstance(kernel_sizes, int):
            kernel_sizes = [(kernel_sizes, kernel_sizes)] * n_convlayers

        elif isinstance(kernel_sizes, tuple):
            kernel_sizes = [kernel_sizes] * n_convlayers

        if isinstance(strides, int):
            strides = [(strides, strides)] * n_convlayers

        elif isinstance(strides, tuple):
            strides = [strides] * n_convlayers

        if isinstance(paddings, int):
            paddings = [(paddings, paddings)] * n_convlayers

        elif isinstance(paddings, tuple):
            paddings = [paddings] * n_convlayers

        if isinstance(bias, bool):
            bias = [bias] * n_convlayers

        if isinstance(batch_normalization, bool):
            batch_normalization = [batch_normalization] * n_convlayers

        if isinstance(dropout, float):
            dropout = [dropout] * n_convlayers

        layers = OrderedDict([])
        input_features = input_channels
        for ci, (ic, af, ks, st, pd, bi, bn, do) in enumerate(zip(intermediate_channels, activation_functions, kernel_sizes, strides, paddings, bias, batch_normalization, dropout)):
            layers['Conv_' + str(ci+1)] = nn.Conv2d(input_features, ic, kernel_size=ks, stride=st, padding=pd, bias=bi)
            if bn:
                layers['BatchNorm_' + str(ci+1)] = nn.BatchNorm2d(ic)

            if af is not None:
                try: # Try to use inplace if available
                    layers['ActivationFun_' + str(ci + 1)] = af(inplace=True)

                except TypeError:
                    layers['ActivationFun_' + str(ci + 1)] = af()

            if do is not None and do > 1e-6 and ci < (n_convlayers - 1):
                layers['Dropout_' + str(ci + 1)] = nn.Dropout2d(do)

            input_features = ic

        self._layers = nn.Sequential(layers)

    def forward(self, x):
        fx = self._layers(x)
        return fx


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, intermediate_channels=None, downsampling_functions=None, projection_functions=None, activation_functions=nn.ReLU, kernel_sizes=3, strides=1, paddings=1, bias=False, batch_normalization=False, dropout=0.0):
        super(ResBlock, self).__init__()

        if intermediate_channels is None:
            intermediate_channels = []

        intermediate_channels.append(output_channels)

        # Number of convolution layers
        n_convlayers = len(intermediate_channels)

        if not isinstance(downsampling_functions, list):
            downsampling_functions = [downsampling_functions] * n_convlayers

        if not isinstance(projection_functions, list):
            projection_functions = [projection_functions] * n_convlayers

        if not isinstance(activation_functions, list):
            activation_functions = [activation_functions] * n_convlayers

        if isinstance(kernel_sizes, int):
            kernel_sizes = [(kernel_sizes, kernel_sizes)] * n_convlayers

        elif isinstance(kernel_sizes, tuple):
            kernel_sizes = [kernel_sizes] * n_convlayers

        if isinstance(strides, int):
            strides = [(strides, strides)] * n_convlayers

        elif isinstance(strides, tuple):
            strides = [strides] * n_convlayers

        if isinstance(paddings, int):
            paddings = [(paddings, paddings)] * n_convlayers

        elif isinstance(paddings, tuple):
            paddings = [paddings] * n_convlayers

        if isinstance(bias, bool):
            bias = [bias] * n_convlayers

        if isinstance(batch_normalization, bool):
            batch_normalization = [batch_normalization] * n_convlayers

        if isinstance(dropout, float):
            dropout = [dropout] * n_convlayers

        projection_layers = OrderedDict([])
        layers = OrderedDict([])
        input_features = input_channels
        for ci, (ic, df, pf, af, ks, st, pd, bi, bn, do) in enumerate(zip(intermediate_channels, downsampling_functions, projection_functions, activation_functions, kernel_sizes, strides, paddings, bias, batch_normalization, dropout)):
            layers['Conv_' + str(ci+1)] = nn.Conv2d(input_features, ic, kernel_size=ks, stride=st, padding=pd, bias=bi)
            if bn:
                layers['BatchNorm_' + str(ci+1)] = nn.BatchNorm2d(ic)

            if af is not None:
                try: # Try to use inplace if available
                    layers['ActivationFun_' + str(ci + 1)] = af(inplace=True)

                except TypeError:
                    layers['ActivationFun_' + str(ci + 1)] = af()

            if df is not None:
                layers['DownsamplingFun_' + str(ci + 1)] = df(kernel_size=2, stride=2)
                projection_layers['ProjectionFun_' + str(ci + 1)] = pf(input_channels, output_channels, kernel_size=1, stride=2, padding=0, bias=bi)

            elif pf is not None:
                projection_layers['ProjectionFun_' + str(ci + 1)] = pf(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=bi)

            if do is not None and do > 1e-6 and ci < (n_convlayers - 1):
                layers['Dropout_' + str(ci + 1)] = nn.Dropout2d(do)

            input_features = ic

        self._layers = nn.Sequential(layers)
        if len(projection_layers) > 0:
            self._projection_layers = nn.Sequential(projection_layers)
        else:
            self._projection_layers = None

    def forward(self, x):
        if self._projection_layers is not None:
            x_prim = self._projection_layers(x)
        else:
            x_prim = x.clone()

        fx = self._layers(x)
        return fx + x_prim


class DownsamplingBlock(nn.Module):
    def __init__(self, reduction_function=nn.MaxPool2d, kernel_size=2, stride=2, padding=1):
        super(DownsamplingBlock, self).__init__()

        self._layers = reduction_function(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        fx = self._layers(x)
        return fx


class UpsamplingBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, bias=False):
        super(UpsamplingBlock, self).__init__()

        self._layers = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        fx = self._layers(x)
        return fx


class FullyConnectedBlock(nn.Module):
    def __init__(self, input_channels, output_channels, intermediate_channels=None, activation_functions=nn.ReLU, bias=False, batch_normalization=False, dropout=0.0):
        super(FullyConnectedBlock, self).__init__()

        if intermediate_channels is None:
            intermediate_channels = []

        intermediate_channels.append(output_channels)

        # Number of convolution layers
        n_layers = len(intermediate_channels)

        if not isinstance(activation_functions, list):
            activation_functions = [activation_functions] * n_layers

        if isinstance(bias, bool):
            bias = [bias] * n_layers

        if isinstance(batch_normalization, bool):
            batch_normalization = [batch_normalization] * n_layers

        if isinstance(dropout, float):
            dropout = [dropout] * n_layers

        layers = OrderedDict([])
        input_features = input_channels
        for ci, (ic, af, bi, bn, do) in enumerate(zip(intermediate_channels, activation_functions, bias, batch_normalization, dropout)):
            layers['Linear_' + str(ci + 1)] = nn.Linear(input_features, ic, bias=bi)
            if bn:
                layers['BatchNorm_' + str(ci + 1)] = nn.BatchNorm1d(ic)

            if af is not None:
                try:  # Try to use inplace if available
                    layers['ActivationFun_' + str(ci + 1)] = af(inplace=True)

                except TypeError:
                    layers['ActivationFun_' + str(ci + 1)] = af()

            if do is not None and ci < (n_layers - 1):
                layers['Dropout_' + str(ci + 1)] = nn.Dropout(do)

            input_features = ic

        self._layers = nn.Sequential(layers)

    def forward(self, x):
        if x.dim() == 4:
            fx = self._layers(x.view(-1, x.size(1) * x.size(2) * x.size(3)))

        else:
            fx = self._layers(x)

        return fx


layers_dict = {'conv': ConvBlock, 'downsample': DownsamplingBlock, 'upsample':UpsamplingBlock, 'fullyconnected': FullyConnectedBlock, 'resblock': ResBlock}
