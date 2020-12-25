import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


def _weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, list):
        for m_i in m:
            if isinstance(m_i, nn.Conv2d) or isinstance(m_i, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


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


class CNNBase(nn.Module):
    def __init__(self, input_features, output_classes, architecture):
        super(CNNBase, self).__init__()

        self.config = {'name': 'cnn_base', 'input_features': input_features, 'output_channels': output_classes, 'architecture': architecture}

        self._input_features = input_features
        self._output_classes = output_classes

        self._layers = []
        self._defineArchitecture()

        self.apply(_weight_init)

    def _defineArchitecture(self):
        model_list = []
        for block, params in self.config['architecture']:
            model_list.append(layers_dict[block](*params))

        self._layers = nn.ModuleList(model_list)

    def forward(self, x):
        fx = x.clone()
        for layer in self._layers:
            fx = layer(fx)

        return fx


class YOLO(CNNBase):
    def __init__(self, input_features, output_classes, architecture=None):
        if architecture is None:
            architecture = [('conv', (input_features, 64, None, nn.ReLU, 7, 1, 0, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 2, 2, 0)),
                            ('conv', (64, 192, None, nn.ReLU, 3, 1, 0, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 2, 2, 0)),
                            ('conv', (192, 512, [128, 256, 256], nn.ReLU, [1, 3, 1, 3], 1, 0, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 2, 2, 0)),
                            ('conv', (512, 1024, [256, 512, 256, 512, 256, 512, 256, 512, 512], nn.ReLU, [1, 3, 1, 3, 1, 3, 1, 3, 1, 3], 1, 0, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 2, 2, 0)),
                            ('conv', (1024, 1024, [512, 1024, 512, 1024, 1024], nn.ReLU, [1, 3, 1, 3, 3, 3], [1, 1, 1, 1, 1, 2], 0, True, True, 0.0)),
                            ('conv', (1024, 1024, [1024], nn.ReLU, [3, 3], 1, 1, True, True, 0.0)),
                            ('fullyconnected', (7*7*1024, output_classes, [4096], nn.ReLU, True, True, 0.5))]

        super(YOLO, self).__init__(input_features, output_classes, architecture)
        self.config['name'] = 'YOLO'


class UNet(CNNBase):
    def __init__(self, input_features, output_classes, architecture=None):
        if architecture is None:
            architecture = [('conv', (input_features, 64, [64], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 2, 2, 0)),
                            ('conv', (64, 128, [128], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 2, 2, 0)),
                            ('conv', (128, 256, [256], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 2, 2, 0)),
                            ('conv', (256, 512, [512], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 2, 2, 0)),
                            ('conv', (512, 1024, [1024], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('upsample', (1024, 512, 2, 2, 0, True)),
                            ('conv', (1024, 512, [512], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('upsample', (512, 256, 2, 2, 0, True)),
                            ('conv', (512, 256, [256], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('upsample', (256, 128, 2, 2, 0, True)),
                            ('conv', (256, 128, [128], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('upsample', (128, 64, 2, 2, 0, True)),
                            ('conv', (128, output_classes, [64, 64], nn.ReLU, 3, 1, 1, True, True, 0.5))]

        super(UNet, self).__init__(input_features, output_classes, architecture)
        self.config['name'] = 'UNet'

    def forward(self, x):
        bridge_tensors = []
        for layer, (layer_type, _) in zip(self._layers, self.config['architecture']):
            if layer_type == 'downsample':
                bridge_tensors.append(x.clone())
                x = layer(x)

            elif layer_type == 'upsample':
                fx = layer(x)
                x = torch.cat((bridge_tensors.pop(), fx), dim=1)

            else:
                x = layer(x)

        return x


class VGG(CNNBase):
    def __init__(self, input_features, output_classes, architecture=None):
        if architecture is None:
            architecture = [('conv', (input_features, 64, [64], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 2, 2, 0)),
                            ('conv', (64, 128, [128], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 2, 2, 0)),
                            ('conv', (128, 256, [256], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 2, 2, 0)),
                            ('conv', (256, 512, [512], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 2, 2, 0)),
                            ('conv', (512, 512, [512], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('fullyconnected', (4*512, output_classes, [4096], nn.ReLU, True, True, 0.5))]

        super(VGG, self).__init__(input_features, output_classes, architecture)
        self.config['name'] = 'VGG'


class ResNet(CNNBase):
    def __init__(self, input_features, output_classes, architecture=None):
        if architecture is None:
            architecture = [('conv', (input_features, 64, [64], nn.ReLU, 7, 2, 0, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 3, 2, 1))]
            architecture += [('resblock', (64, 256, [64, 64], None, [IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))]
            architecture += [('resblock', (256, 256, [64, 64], None, [IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))] * 2
            architecture += [('resblock', (256, 512, [128, 128], [nn.MaxPool2d, None, None], [IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))]
            architecture += [('resblock', (512, 512, [128, 128], None, [IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))]*7
            architecture += [('resblock', (512, 1024, [256, 256], [nn.MaxPool2d, None, None], [IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))]
            architecture += [('resblock', (1024, 1024, [256, 256], None, [IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))] * 35
            architecture += [('resblock', (1024, 2048, [512, 512], [nn.MaxPool2d, None, None], [IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))]
            architecture += [('resblock', (2048, 2048, [512, 512], None, [IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))] * 2
            architecture += [('downsample', (nn.AvgPool2d, 3, 3, 0)), ('fullyconnected', (2048, output_classes, [1000], nn.ReLU, True, True, 0.5))]

        super(ResNet, self).__init__(input_features, output_classes, architecture)
        self.config['name'] = 'ResNet152'


if __name__ == '__main__':
    net = YOLO(3, 7*7*30)
    criterion = nn.MSELoss()

    # net.cuda()
    # criterion.cuda()

    test_input = torch.randn([2, 3, 448, 448])
    target = torch.randn([2, 7*7*30])

    # test_input = test_input.cuda()
    # target = target.cuda()

    with torch.no_grad():
        test_output = net(test_input)

    print('Net output:', test_output.size())
    print('Target:', target.size(), target.min(), target.max())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for e in range(50):
        optimizer.zero_grad()
        output = net(test_input)
        # output = F.log_softmax(output, dim=1)
        loss = criterion(output, target)
        print('[{}] Loss: {}'.format(e, loss.item()))
        loss.backward()
        optimizer.step()

    print('Training finished !!')