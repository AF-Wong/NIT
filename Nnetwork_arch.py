from collections import OrderedDict
import torch
from torch import nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import torch.fft

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class PFFN(nn.Module):
    def __init__(self, dim):
        super(PFFN,self).__init__()
        self.conva = nn.Conv2d(dim,dim * 2,1,1,0)
        self.conv1 = nn.Conv2d(dim, dim ,1,1,0)
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.dim = dim
        self.act =nn.GELU()
        self.convb = nn.Conv2d(dim * 2, dim, 1, 1, 0)

    def forward(self, x):

        x = self.act(self.conva(x))
        x1, x2 = torch.split(x,[self.dim,self.dim],dim=1)
        x1 = self.act(self.conv1(x1))
        x2f = torch.fft.fftn(x2,dim=(2,3))
        x2r = self.act(self.conv2(x2f.real))
        x2i = self.act(self.conv3(x2f.imag))
        x2 = torch.complex(x2r,x2i)
        x2 = torch.abs(torch.fft.ifftn(x2,dim=(2,3)))
        x = self.convb(torch.cat([x1,x2], dim=1))
        return x


class LEM(nn.Module):
    def __init__(self, planes : int, mix_margin: int = 1) -> None:
        super(LEM, self).__init__()

        assert planes % 5 == 0

        self.planes = planes
        #self.mix_margin = nn.Parameter(torch.tensor(mix_margin, dtype=torch.float), requires_grad=True)
        self.mix_margin = mix_margin
        self.mask = nn.Parameter(torch.zeros((self.planes, 1, mix_margin*2+1, mix_margin*2+1)), requires_grad=False)

        #self.eval_conv = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1, stride=1, bias=True)
        #self.eval_conv.weight.requires_grad = False
        #self.eval_conv.bias.requires_grad = False

        #self.mask[3::4, 0, 0, mix_margin] = 1. #从右往左
        #self.mask[2::4, 0, -1, mix_margin] = 1. #从左往右
        #self.mask[1::4, 0, mix_margin, 0] = 1. #从下往上
        #self.mask[0::4, 0, mix_margin, -1] = 1. #从上往下

        #self.mask[4::8, 0, 0, 2] = 1. #左斜下
        #self.mask[5::8, 0, 2, 0] = 1. #右斜上
        #self.mask[6::8, 0, 2, 2] = 1. #左斜上
        #self.mask[7::8, 0, 0, 0] = 1. #右斜下

        self.mask[3::5, 0, 0, mix_margin] = 1.
        self.mask[2::5, 0, -1, mix_margin] = 1.
        self.mask[1::5, 0, mix_margin, 0] = 1.
        self.mask[0::5, 0, mix_margin, -1] = 1.
        self.mask[4::5, 0, mix_margin, mix_margin] = 1.


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #m = int(self.mix_margin.item())
        m = self.mix_margin
        x = F.conv2d(input=F.pad(x, pad=(m, m, m, m), mode='circular'),
        weight=self.mask, bias=None, stride=(1, 1), padding=(0, 0),
        dilation=(1, 1), groups=self.planes)
        #if self.training:
            #m = int(self.mix_margin.item())
            #x = F.conv2d(input=F.pad(x, pad=(m, m, m, m), mode='circular'),
                        #weight=self.mask, bias=None, stride=(1, 1), padding=(0, 0),
                        #dilation=(m, m), groups=self.planes)
        #else :
            #x = self.eval_conv(x)

        return x


class Ablock(nn.Module):
    def __init__(self, dim, paradigm='spatial'):
        super(Ablock, self).__init__()
        self.lem0 = LEM(planes=dim)
        self.lem1 = LEM(planes=dim)
        self.lem2 = LEM(planes=dim)
        self.lem3 = LEM(planes=dim)
        self.lem4 = LEM(planes=dim)
        self.lem5 = LEM(planes=dim)
        self.act = nn.SiLU(inplace=True)
        self.paradigm = paradigm

    def forward(self, x):
        if self.paradigm == 'spatial':
            f = self.lem0(x)
            f = self.act(f)
            fatt = self.lem1(torch.sigmoid(f) - 0.5)
            f = (f + x) * fatt
        else:
            f = torch.fft.fftn(x, dim=(2, 3))

            fr = self.lem2(f.real)
            fr = self.act(fr)
            fi = self.lem3(f.imag)
            fi = self.act(fi)

            attr = self.lem4(torch.sigmoid(fr) - 0.5)
            atti = self.lem5(torch.sigmoid(fi) - 0.5)

            f = torch.complex(fr, fi)
            fatt = torch.complex(attr, atti)
            f = torch.abs(torch.fft.ifftn(f, dim=(2, 3)))
            #fatt = self.lem(torch.sigmoid(f) - 0.5)
            fatt = torch.abs(torch.fft.ifftn(fatt, dim=(2, 3)))
            f = (f + x) * fatt

        return f

class AAblock(nn.Module):
    def __init__(self, dim):
        super(AAblock, self).__init__()
        self.SAblock = Ablock(dim = dim,paradigm="spatial" )
        self.FAblock = Ablock(dim = dim,paradigm="frequency")
        self.ffn = PFFN(dim)
    def forward(self, x):

        out0 = self.SAblock(x)+x+self.FAblock(x)
        out = self.ffn(out0)+out0

        return out

@ARCH_REGISTRY.register()
class Nnetwork(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 feature_channels=40,
                 upscale=4,
                 bias=True,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 task='sr'
                 ):
        super(Nnetwork, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv0 = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1, bias=bias)
        self.AAblock0= AAblock(dim=feature_channels)
        self.AAblock1 = AAblock(dim=feature_channels)
        self.AAblock2 = AAblock(dim=feature_channels)
        self.AAblock3 = AAblock(dim=feature_channels)
        self.AAblock4 = AAblock(dim=feature_channels)
        self.AAblock5 = AAblock(dim=feature_channels)

        #self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        #self.conv1 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x0 = self.conv0(x)

        #out_b1, _, att1 = self.block_1(out_feature)
        #out_b2, _, att2 = self.block_2(out_b1)
        #out_b3, _, att3 = self.block_3(out_b2)

        #out_b4, _, att4 = self.block_4(out_b3)
        #out_b5, _, att5 = self.block_5(out_b4)
        #out_b6, out_b5_2, att6 = self.block_6(out_b5)

        #out_b6 = self.conv_2(out_b6)
        #out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        out0 = self.AAblock0(x0)
        out1 = self.AAblock1(out0)
        out2 = self.AAblock2(out1)
        out3 = self.AAblock3(out2)
        out4 = self.AAblock4(out3)
        out5 = self.AAblock5(out4)

        out = out5+x0

        output = self.upsampler(out)

        return output

if __name__ == "__main__":
    # from fvcore.nn import FlopCountAnalysis, flop_count_table
    # import time
    model = Nnetwork(3, 3, upscale=4, feature_channels=40)
    # model.eval()
    inputs = (torch.rand(1, 3, 256, 256),)
    print(model(*inputs).shape)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(flop_count_table(FlopCountAnalysis(model, inputs)))
    print(count_parameters(model))