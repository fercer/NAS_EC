import torch
import torch.nn as nn
import torch.nn.functional as F

import blocks as bl

import pytorch_lightning as pl



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


class CNNBase(pl.LightningModule):
    def __init__(self, input_features, output_classes, architecture, lr=1e-3):
        super(CNNBase, self).__init__()
        
        self.lr = lr

        self.config = {'name': 'cnn_base', 'input_features': input_features, 'output_channels': output_classes, 'architecture': architecture}

        self._input_features = input_features
        self._output_classes = output_classes

        self._layers = []
        self._defineArchitecture()

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.apply(_weight_init)

    def _defineArchitecture(self):
        model_list = []
        for block, params in self.config['architecture']:
            model_list.append(bl.layers_dict[block](*params))

        self._layers = nn.ModuleList(model_list)

    def forward(self, x):
        fx = x.clone()
        for layer in self._layers:
            fx = layer(fx)

        return fx

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        # loss = F.mse_loss(y_hat, y)

        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.binary_cross_entropy_with_logits(y_hat, y)
        # val_loss = F.mse_loss(y_hat, y)

        self.valid_acc(y_hat, y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        # loss = F.mse_loss(y_hat, y)

        self.test_acc(y_hat, y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)

        return loss


class YOLO(CNNBase):
    def __init__(self, input_features, output_classes, architecture=None, **kwargs):
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

        super(YOLO, self).__init__(input_features, output_classes, architecture, **kwargs)
        self.config['name'] = 'YOLO'


class UNet(CNNBase):
    def __init__(self, input_features, output_classes, architecture=None, **kwargs):
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
    def __init__(self, input_features, output_classes, architecture=None, **kwargs):
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
                            ('downsample', (nn.MaxPool2d, 2, 2, 0)),
                            ('conv', (512, 512, [512], nn.ReLU, 3, 1, 1, True, True, 0.0)),
                            ('fullyconnected', (7*7*512, output_classes, [4096], nn.ReLU, True, True, 0.5))]

        super(VGG, self).__init__(input_features, output_classes, architecture)
        self.config['name'] = 'VGG'


class ResNet(CNNBase):
    def __init__(self, input_features, output_classes, architecture=None, **kwargs):
        if architecture is None:
            architecture = [('conv', (input_features, 64, [64], nn.ReLU, 7, 2, 0, True, True, 0.0)),
                            ('downsample', (nn.MaxPool2d, 3, 2, 1))]
            architecture += [('resblock', (64, 256, [64, 64], None, [bl.IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))]
            architecture += [('resblock', (256, 256, [64, 64], None, [bl.IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))] * 2
            architecture += [('resblock', (256, 512, [128, 128], [nn.MaxPool2d, None, None], [bl.IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))]
            architecture += [('resblock', (512, 512, [128, 128], None, [bl.IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))]*7
            architecture += [('resblock', (512, 1024, [256, 256], [nn.MaxPool2d, None, None], [bl.IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))]
            architecture += [('resblock', (1024, 1024, [256, 256], None, [bl.IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))] * 35
            architecture += [('resblock', (1024, 2048, [512, 512], [nn.MaxPool2d, None, None], [bl.IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))]
            architecture += [('resblock', (2048, 2048, [512, 512], None, [bl.IdentityModule, None, None], nn.ReLU, [1, 3, 1], 1, [0, 1, 0], True, True, 0.0))] * 2
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