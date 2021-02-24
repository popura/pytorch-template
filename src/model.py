import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchsummary import summary
import deepy.nn.functional


class BaselineCNN(nn.Module):
    """
    """

    def __init__(self, num_classes, feature_dim):
        super(BaselineCNN, self).__init__()

        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=32,
                                  kernel_size=(feature_dim, 5), stride=1,
                                  padding=0, bias=False),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),

                        nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5)),

                        nn.Conv2d(in_channels=32, out_channels=64,
                                  kernel_size=(1, 3), stride=1,
                                  padding=0, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),

                        nn.AdaptiveAvgPool2d((1, 1))
                    )
        self.linear = nn.Sequential(
                        nn.Linear(64, 64),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.2),
                        nn.Linear(64, num_classes),
                    )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class VGG1d(nn.Module):
    """ U-Net
    """
    class double_conv(nn.Module):
        '''(conv => BN => ReLU) * 2'''
        def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU):
            super(VGG1d.double_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                nn.BatchNorm1d(out_channels),
                activation(),
                nn.Conv1d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                nn.BatchNorm1d(out_channels),
                activation()
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class inconv(nn.Module):
        def __init__(self, in_channels, out_channels, activation=nn.ReLU):
            super(VGG1d.inconv, self).__init__()
            self.conv = VGG1d.double_conv(in_channels, out_channels, activation)

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_channels, out_channels,
                     down_conv_layer=nn.Conv1d, activation=nn.ReLU):
            super(VGG1d.down, self).__init__()
            self.mpconv = nn.Sequential(
                down_conv_layer(in_channels=in_channels, out_channels=in_channels,
                                padding=1, kernel_size=3,
                                stride=2, bias=False),
                VGG1d.double_conv(in_channels, out_channels, activation=activation)
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x

    def __init__(self, in_channels: int, num_classes: int,
                 base_channels: int, depth: int,
                 max_channels: int=512,
                 down_conv_layer=nn.Conv1d,
                 activation=nn.ReLU):
        super(VGG1d, self).__init__()
        self.depth = depth
        self.inc = VGG1d.inconv(in_channels=in_channels,
                               out_channels=base_channels,
                               activation=activation)
        self.down_blocks = nn.ModuleList(
            [
                VGG1d.down(
                    in_channels=min(base_channels*(2**i), max_channels),
                    out_channels=min(base_channels*(2**(i+1)), max_channels),
                    down_conv_layer=down_conv_layer,
                    activation=activation
                )
                for i in range(depth)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Sequential(
                        nn.Linear(min(base_channels*(2**depth), max_channels), 64),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.2),
                        nn.Linear(64, num_classes),
                    )

    def forward(self, x):
        skip_connections = []
        x = self.inc(x)
        for i, l in enumerate(self.down_blocks):
            skip_connections.append(x)
            x = l(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MultiVGG1d(nn.Module):
    def __init__(self, in_channels: int, num_inputs: int,
                 num_classes: int,
                 base_channels: int, depth: int,
                 max_channels: int=512,
                 down_conv_layer=nn.Conv1d,
                 activation=nn.ReLU):
        super(MultiVGG1d, self).__init__()
        self.vgg = VGG1d(in_channels=in_channels*num_inputs,
                         num_classes=num_classes,
                         base_channels=base_channels,
                         depth=depth,
                         max_channels=max_channels,
                         down_conv_layer=nn.Conv1d,
                         activation=nn.ReLU)
    
    def forward(self, *inputs):
        y = list(inputs)
        y = torch.cat(y, dim=1)
        y = self.vgg(y)
        return y


class VGG2d(nn.Module):
    """ U-Net
    """
    class double_conv(nn.Module):
        '''(conv => BN => ReLU) * 2'''
        def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                activation(),
                nn.Conv2d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                activation()
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class inconv(nn.Module):
        def __init__(self, in_channels, out_channels, activation=nn.ReLU):
            super().__init__()
            self.conv = VGG2d.double_conv(in_channels, out_channels, activation)

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_channels, out_channels,
                     down_conv_layer=nn.Conv2d, activation=nn.ReLU):
            super().__init__()
            self.mpconv = nn.Sequential(
                down_conv_layer(in_channels=in_channels, out_channels=in_channels,
                                padding=1, kernel_size=3,
                                stride=2, bias=False),
                VGG2d.double_conv(in_channels, out_channels, activation=activation)
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x

    def __init__(self, in_channels: int, num_classes: int,
                 base_channels: int, depth: int,
                 max_channels: int=512,
                 down_conv_layer=nn.Conv2d,
                 activation=nn.ReLU):
        super().__init__()
        self.depth = depth
        self.inc = VGG2d.inconv(in_channels=in_channels,
                               out_channels=base_channels,
                               activation=activation)
        self.down_blocks = nn.ModuleList(
            [
                VGG2d.down(
                    in_channels=min(base_channels*(2**i), max_channels),
                    out_channels=min(base_channels*(2**(i+1)), max_channels),
                    down_conv_layer=down_conv_layer,
                    activation=activation
                )
                for i in range(depth)
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
                        nn.Linear(min(base_channels*(2**depth), max_channels), 64),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.2),
                        nn.Linear(64, num_classes),
                    )

    def forward(self, x):
        skip_connections = []
        x = self.inc(x)
        for i, l in enumerate(self.down_blocks):
            skip_connections.append(x)
            x = l(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MultiVGG2d(nn.Module):
    def __init__(self, in_channels: int, num_inputs: int,
                 num_classes: int,
                 base_channels: int, depth: int,
                 max_channels: int=512,
                 down_conv_layer=nn.Conv2d,
                 activation=nn.ReLU):
        super(MultiVGG2d, self).__init__()
        self.vgg = VGG2d(in_channels=in_channels*num_inputs,
                         num_classes=num_classes,
                         base_channels=base_channels,
                         depth=depth,
                         max_channels=max_channels,
                         down_conv_layer=nn.Conv2d,
                         activation=nn.ReLU)
    
    def forward(self, *inputs):
        y = list(inputs)
        y = torch.cat(y, dim=1)
        y = self.vgg(y)
        return y


class BlinkyVGG1d(nn.Module):
    def __init__(self, in_channels: int, num_classes: int,
                 base_channels: int, depth: int,
                 sample_rate: int,
                 max_channels: int=512,
                 down_conv_layer=nn.Conv1d,
                 activation=nn.ReLU,
                 audio_buffer_size: int = 64,
                 frame_rate: int = 30,
                 temperature: float = 0.1):
        super(BlinkyVGG1d, self).__init__()
        self.vgg = VGG1d(in_channels=in_channels,
                         num_classes=num_classes,
                         base_channels=base_channels,
                         depth=depth,
                         max_channels=max_channels,
                         down_conv_layer=nn.Conv1d,
                         activation=nn.ReLU)
        self.blinky = Blinky(audio_buffer_size=audio_buffer_size)
        self.camera = CameraResponse(sample_rate=sample_rate//audio_buffer_size,
                                     frame_rate=frame_rate,
                                     temperature=temperature)
    
    def forward(self, x):
        x = self.blinky(x)
        x = self.camera(x)
        x = self.vgg(x)
        return x


class MultiBlinkyVGG1d(nn.Module):
    def __init__(self, in_channels: int, num_inputs: int,
                 num_classes: int,
                 base_channels: int, depth: int,
                 sample_rate: int,
                 distances: List[float], biases: List[float],
                 stds: List[float],
                 max_channels: int=512,
                 down_conv_layer=nn.Conv1d,
                 activation=nn.ReLU,
                 audio_buffer_size: int = 64,
                 frame_rate: int = 30,
                 temperature: float = 0.1):
        super(MultiBlinkyVGG1d, self).__init__()
        self.vgg = VGG1d(in_channels=in_channels*num_inputs,
                         num_classes=num_classes,
                         base_channels=base_channels,
                         depth=depth,
                         max_channels=max_channels,
                         down_conv_layer=nn.Conv1d,
                         activation=nn.ReLU)
        self.blinkies = nn.ModuleList([Blinky(audio_buffer_size=audio_buffer_size)
                                       for i in range(num_inputs)])
        self.lights = nn.ModuleList([LightPropagation(distance=d, bias=b, std=s)
                                     for d, b, s in zip(distances, biases, stds)])
        self.camera = CameraResponse(sample_rate=sample_rate//audio_buffer_size,
                                     frame_rate=frame_rate,
                                     temperature=temperature)
    
    def forward(self, *inputs):
        y = []
        for i, x in enumerate(inputs):
            tmp_y = self.blinkies[i](x) 
            tmp_y = self.lights[i](tmp_y)
            tmp_y = self.camera(tmp_y)
            y.append(tmp_y)
        y = torch.cat(y, dim=1)
        y = self.vgg(y)
        return y


class UNetVGG1d(nn.Module):
    def __init__(self, in_channels: int,
                 num_classes: int,
                 base_channels: int, depth: int,
                 sample_rate: int,
                 max_channels: int=512,
                 up_conv_layer=nn.ConvTranspose1d,
                 down_conv_layer=nn.Conv1d,
                 activation=nn.ReLU,
                 final_activation=nn.Identity,
                 audio_buffer_size: int = 64,
                 frame_rate: int = 30,
                 temperature: float = 0.1):
        super(UNetVGG1d, self).__init__()
        self.vgg = VGG1d(in_channels=in_channels,
                         num_classes=num_classes,
                         base_channels=base_channels,
                         depth=depth,
                         max_channels=max_channels,
                         down_conv_layer=down_conv_layer,
                         activation=activation)
        self.camera = CameraResponse(sample_rate=sample_rate,
                                     frame_rate=frame_rate,
                                     temperature=temperature)
        self.unet = UNet1d(in_channels=in_channels,
                           out_channels=in_channels,
                           base_channels=base_channels,
                           depth=depth,
                           max_channels=max_channels,
                           up_conv_layer=up_conv_layer,
                           down_conv_layer=down_conv_layer,
                           activation=activation,
                           final_activation=final_activation)
    
    def forward(self, x):
        x = self.unet(x)
        x = self.normalization(x)
        x = self.camera(x)
        x = self.vgg(x)
        return x

    def normalization(self, x):
        x_min, _ = torch.min(x, dim=-1, keepdim=True)
        x_max, _ = torch.max(x, dim=-1, keepdim=True)
        return (x - x_min) / (x_max - x_min)


class MultiUNetVGG1d(nn.Module):
    def __init__(self, in_channels: int, num_inputs: int,
                 num_classes: int,
                 base_channels: int, depth: int,
                 sample_rate: int,
                 distances: List[float], biases: List[float],
                 stds: List[float],
                 max_channels: int=512,
                 up_conv_layer=nn.ConvTranspose1d,
                 down_conv_layer=nn.Conv1d,
                 activation=nn.ReLU,
                 final_activation=nn.Identity,
                 audio_buffer_size: int = 64,
                 frame_rate: int = 30,
                 temperature: float = 0.1):
        super(MultiUNetVGG1d, self).__init__()
        self.vgg = VGG1d(in_channels=in_channels*num_inputs,
                         num_classes=num_classes,
                         base_channels=base_channels,
                         depth=depth,
                         max_channels=max_channels,
                         down_conv_layer=down_conv_layer,
                         activation=activation)
        self.camera = CameraResponse(sample_rate=sample_rate,
                                     frame_rate=frame_rate,
                                     temperature=temperature)
        self.lights = nn.ModuleList([LightPropagation(distance=d, bias=b, std=s)
                                     for d, b, s in zip(distances, biases, stds)])
        self.unets = nn.ModuleList([UNet1d(in_channels=in_channels,
                                           out_channels=in_channels,
                                           base_channels=base_channels,
                                           depth=depth,
                                           max_channels=max_channels,
                                           up_conv_layer=up_conv_layer,
                                           down_conv_layer=down_conv_layer,
                                           activation=activation,
                                           final_activation=final_activation)
                                    for i in range(num_inputs)])
    
    def forward(self, *inputs):
        y = []
        for i, x in enumerate(inputs):
            tmp_y = self.unets[i](x) 
            tmp_y = self.normalization(tmp_y)
            tmp_y = self.lights[i](tmp_y)
            tmp_y = self.camera(tmp_y)
            y.append(tmp_y)
        y = torch.cat(y, dim=1)
        y = self.vgg(y)
        return y

    def normalization(self, x):
        x_min, _ = torch.min(x, dim=-1, keepdim=True)
        x_max, _ = torch.max(x, dim=-1, keepdim=True)
        return (x - x_min) / (x_max - x_min)


class UNet1d(nn.Module):
    """ U-Net
    """
    class double_conv(nn.Module):
        '''(conv => BN => ReLU) * 2'''
        def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU):
            super(UNet1d.double_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                nn.BatchNorm1d(out_channels),
                activation(),
                nn.Conv1d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                nn.BatchNorm1d(out_channels),
                activation()
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class inconv(nn.Module):
        def __init__(self, in_channels, out_channels, activation=nn.ReLU):
            super(UNet1d.inconv, self).__init__()
            self.conv = UNet1d.double_conv(in_channels, out_channels, activation)

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_channels, out_channels,
                     down_conv_layer=nn.Conv1d, activation=nn.ReLU):
            super(UNet1d.down, self).__init__()
            self.mpconv = nn.Sequential(
                down_conv_layer(in_channels=in_channels, out_channels=in_channels,
                                padding=1, kernel_size=3,
                                stride=2, bias=False),
                UNet1d.double_conv(in_channels, out_channels, activation=activation)
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x

    class up(nn.Module):
        def __init__(self, in_channels, mid_channels, out_channels,
                     up_conv_layer=nn.ConvTranspose1d, activation=nn.ReLU):
            super(UNet1d.up, self).__init__()
            self.upconv = up_conv_layer(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            self.conv = UNet1d.double_conv(mid_channels, out_channels, activation=activation)

        def forward(self, x1, x2):
            x1 = self.upconv(x1)
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x

    class outconv(nn.Module):
        def __init__(self, in_channels, out_channels, activation=nn.Identity):
            super(UNet1d.outconv, self).__init__()
            self.conv = nn.Sequential(
                    nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              padding=0,
                              bias=True),
                    activation(),
                )

        def forward(self, x):
            x = self.conv(x)
            return x
    
    def __init__(self, in_channels: int, out_channels: int,
                 base_channels: int, depth: int,
                 max_channels: int=512,
                 up_conv_layer=nn.ConvTranspose1d,
                 down_conv_layer=nn.Conv1d,
                 activation=nn.ReLU,
                 final_activation=nn.Identity):
        super(UNet1d, self).__init__()
        self.depth = depth
        self.inc = UNet1d.inconv(in_channels=in_channels,
                                 out_channels=base_channels,
                                 activation=activation)
        self.down_blocks = nn.ModuleList(
            [
                UNet1d.down(
                    in_channels=min(base_channels*(2**i), max_channels),
                    out_channels=min(base_channels*(2**(i+1)), max_channels),
                    down_conv_layer=down_conv_layer,
                    activation=activation
                )
                for i in range(depth)
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                UNet1d.up(
                    in_channels=min(base_channels*(2**(i+1)), max_channels),
                    mid_channels=min(base_channels*(2**i), max_channels)*2,
                    out_channels=max(min(base_channels*(2**i), max_channels), base_channels),
                    up_conv_layer=up_conv_layer,
                    activation=activation
                )
                for i in reversed(range(depth))
            ]
        )
        self.outc = UNet1d.outconv(in_channels=base_channels,
                                   out_channels=out_channels,
                                   activation=final_activation)

    def forward(self, x):
        skip_connections = []
        x = self.inc(x)
        for l in self.down_blocks:
            skip_connections.append(x)
            x = l(x)
        for l in self.up_blocks:
            x = l(x, skip_connections.pop())
        x = self.outc(x)
        return x


class LightPropagation(nn.Module):
    def __init__(self, distance: float, bias: float, std: float):
        super(LightPropagation, self).__init__()
        self.distance = distance
        self.bias = bias
        self.std = std

    def forward(self, x):
        attenuation = 1 / (self.distance**2)
        noise = self.std * torch.randn(x.shape).to(x.device) + self.bias
        x = attenuation * x + noise
        return x


class CameraResponse(nn.Module):
    def __init__(self, sample_rate: int,
                 levels=None, frame_rate: int = 30,
                 temperature: float = 0.1):
        super(CameraResponse, self).__init__()
        self.frame_rate = frame_rate
        self.temperature = temperature
        #self.levels = levels
        self.resample = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=frame_rate,
            resampling_method='sinc_interpolation')
    
    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x = self.resample(x)
        # x = deepy.nn.functional.softstaircase(x, self.levels, self.temperature)
        return x


class Blinky(nn.Module):
    def __init__(self, audio_buffer_size: int = 64):
        super(Blinky, self).__init__()
        self.audio_buffer_size = audio_buffer_size
        self.pool = nn.AvgPool1d(kernel_size=audio_buffer_size)

    def forward(self, x):
        x = x ** 2
        x = self.pool(x)
        x = self.normalization(x)
        x = self.nonlinearity(x)
        return x
    
    def nonlinearity(self, x):
        # This is a polynomial approximation of the original nonlinear function
        coef = [-40.0177, 154.9749, -224.3099, 148.0482,
                -44.6547,   6.8411,    0.1164,   0.0044]
        y = torch.zeros_like(x)
        for i in coef:
            y = y*x + i
        
        return  y
    
    def normalization(self, x):
        x_min, _ = torch.min(x, dim=-1, keepdim=True)
        x_max, _ = torch.max(x, dim=-1, keepdim=True)
        return (x - x_min) / (x_max - x_min)


class UNetEncoder1d(nn.Module):
    """ U-Net
    """
    class double_conv(nn.Module):
        '''(conv => BN => ReLU) * 2'''
        def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                nn.BatchNorm1d(out_channels),
                activation(),
                nn.Conv1d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          padding=1,
                          bias=False),
                nn.BatchNorm1d(out_channels),
                activation()
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class inconv(nn.Module):
        def __init__(self, in_channels, out_channels, activation=nn.ReLU):
            super().__init__()
            self.conv = UNetEncoder1d.double_conv(in_channels, out_channels, activation)

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_channels, out_channels,
                     down_conv_layer=nn.Conv1d, activation=nn.ReLU):
            super().__init__()
            self.mpconv = nn.Sequential(
                down_conv_layer(in_channels=in_channels, out_channels=in_channels,
                                padding=1, kernel_size=3,
                                stride=2, bias=False),
                UNetEncoder1d.double_conv(in_channels, out_channels, activation=activation)
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x

    class outconv(nn.Module):
        def __init__(self, in_channels, out_channels, activation=nn.Identity):
            super().__init__()
            self.conv = nn.Sequential(
                    nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              padding=0,
                              bias=True),
                    activation(),
                )

        def forward(self, x):
            x = self.conv(x)
            return x
    
    def __init__(self, in_channels: int, out_channels: int,
                 base_channels: int, depth: int,
                 max_channels: int=512,
                 down_conv_layer=nn.Conv1d,
                 activation=nn.ReLU,
                 final_activation=nn.Identity):
        super().__init__()
        self.depth = depth
        self.inc = UNetEncoder1d.inconv(in_channels=in_channels,
                                        out_channels=base_channels,
                                        activation=activation)
        self.down_blocks = nn.ModuleList(
            [
                UNetEncoder1d.down(
                    in_channels=min(base_channels*(2**i), max_channels),
                    out_channels=min(base_channels*(2**(i+1)), max_channels),
                    down_conv_layer=down_conv_layer,
                    activation=activation
                )
                for i in range(depth)
            ]
        )
        self.outc = UNetEncoder1d.outconv(in_channels=min(base_channels*(2**depth), max_channels),
                                          out_channels=out_channels,
                                          activation=final_activation)

    def forward(self, x):
        x = self.inc(x)
        for l in self.down_blocks:
            x = l(x)
        x = self.outc(x)
        return x


class MultiEncoderVGG1d(nn.Module):
    def __init__(self, in_channels: int, num_inputs: int,
                 num_classes: int,
                 base_channels: int, depth: int,
                 sample_rate: int,
                 distances: List[float], biases: List[float],
                 stds: List[float],
                 max_channels: int=512,
                 down_conv_layer=nn.Conv1d,
                 activation=nn.ReLU,
                 final_activation=nn.Identity,
                 audio_buffer_size: int = 64,
                 frame_rate: int = 30,
                 temperature: float = 0.1):
        super().__init__()
        self.vgg = VGG1d(in_channels=in_channels*num_inputs,
                         num_classes=num_classes,
                         base_channels=base_channels,
                         depth=depth,
                         max_channels=max_channels,
                         down_conv_layer=down_conv_layer,
                         activation=activation)
        self.camera = CameraResponse(sample_rate=sample_rate//audio_buffer_size,
                                     frame_rate=frame_rate,
                                     temperature=temperature)
        self.lights = nn.ModuleList([LightPropagation(distance=d, bias=b, std=s)
                                     for d, b, s in zip(distances, biases, stds)])
        self.unets = nn.ModuleList([UNetEncoder1d(in_channels=in_channels,
                                                  out_channels=in_channels,
                                                  base_channels=base_channels,
                                                  depth=int(math.log2(audio_buffer_size)),
                                                  max_channels=max_channels,
                                                  down_conv_layer=down_conv_layer,
                                                  activation=activation,
                                                  final_activation=final_activation)
                                    for i in range(num_inputs)])
    
    def forward(self, *inputs):
        y = []
        for i, x in enumerate(inputs):
            tmp_y = self.unets[i](x) 
            tmp_y = self.normalization(tmp_y)
            tmp_y = self.lights[i](tmp_y)
            tmp_y = self.camera(tmp_y)
            y.append(tmp_y)
        y = torch.cat(y, dim=1)
        y = self.vgg(y)
        return y

    def normalization(self, x):
        x_min, _ = torch.min(x, dim=-1, keepdim=True)
        x_max, _ = torch.max(x, dim=-1, keepdim=True)
        return (x - x_min) / (x_max - x_min)
