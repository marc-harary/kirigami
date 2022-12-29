from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, dilations):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[0],
            padding=self.get_padding(dilations[0], kernel_sizes[0]),
            dilation=dilations[0], bias=False)
        # self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=.5)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_sizes[1],
            padding=self.get_padding(dilations[1], kernel_sizes[1]),
            dilation=dilations[1], bias=False)
        # self.norm2 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, ipt):
        opt = ipt
        opt = self.conv1(opt)
        if opt.shape[-1] > 1:
            opt = self.norm1(opt)
        opt = self.act1(opt)
        opt = self.drop(opt)
        opt = self.conv2(opt)
        if opt.shape[-1] > 1:
            opt = self.norm2(opt)
        opt = self.act2(opt)
        return opt

    @staticmethod
    def get_padding(dilation: int, kernel_size: int) -> int:
        return round((dilation * (kernel_size - 1)) / 2)
    


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, dilations):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv_block1 = ConvBlock(in_channels, out_channels, kernel_sizes, dilations)
        self.conv_block2 = ConvBlock(out_channels, out_channels, kernel_sizes, dilations)
        self.conv_block3 = ConvBlock(out_channels, out_channels, kernel_sizes, dilations)

    def forward(self, ipt):
        opt = ipt
        if opt.shape[-1] > 1:
            opt = self.pool(opt)
        opt = self.conv_block1(opt)
        opt = self.conv_block2(opt)
        opt = self.conv_block3(opt)
        return opt


# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv_block = ConvBlock(in_channels, out_channels)
# 
#     def forward(self, ipt1, ipt2):
#         ipt1 = self.deconv(ipt1)
#         diffY = ipt2.size()[2] - ipt1.size()[2]
#         diffX = ipt2.size()[3] - ipt1.size()[3]
#         ipt1 = F.pad(ipt1, [diffX // 2, diffX - diffX // 2,
#                             diffY // 2, diffY - diffY // 2])
#         opt = torch.cat([ipt2, ipt1], dim=1)
#         opt = self.conv_block(opt)
#         return opt


class DecoderBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, kernel_sizes, dilations):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels1, out_channels, kernel_size=2, stride=3)
        self.conv_block1 = ConvBlock(in_channels2 + out_channels, out_channels, kernel_sizes, dilations)
        self.conv_block2 = ConvBlock(out_channels, out_channels, kernel_sizes, dilations)
        self.conv_block3 = ConvBlock(out_channels, out_channels, kernel_sizes, dilations)

    def forward(self, ipt1, ipt2):
        ipt1 = self.deconv(ipt1)
        diffY = ipt2.shape[2] - ipt1.shape[2]
        diffX = ipt2.shape[3] - ipt1.shape[3]
        ipt1 = F.pad(ipt1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        opt = torch.cat([ipt2, ipt1], dim=1)
        opt = self.conv_block1(opt)
        opt = self.conv_block2(opt)
        opt = self.conv_block3(opt)
        return opt


class UNet(nn.Module):
    def __init__(self, channels: List[int], kernel_sizes=(3, 5), dilations=None):
        super(UNet, self).__init__()
        if dilations is None:
            dilations = (len(channels) // 2) * [1, 2, 4, 8, 16]
        self.ipt_conv = ConvBlock(channels[0], channels[1], kernel_sizes=kernel_sizes, dilations=dilations[0:2])
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(2, len(channels)):
            self.encoders.append(EncoderBlock(channels[i-1], channels[i], kernel_sizes=kernel_sizes, dilations=dilations[2*i:2*(i+1)]))
        for i in range(len(channels)-1, 0, -1):
            # self.decoders.append(DecoderBlock(channels[i], channels[i-1]))
            self.decoders.append(DecoderBlock(channels[i], channels[i-1], channels[i-1], kernel_sizes=kernel_sizes, dilations=dilations[2*i:2*(i+1)]))
        self.opt_conv = nn.Conv2d(channels[1], 1, kernel_size=1)

    def forward(self, ipt):
        opt = self.ipt_conv(ipt)
        inters = [opt]
        for encoder in self.encoders:
            inter = encoder(inters[-1])
            inters.append(inter)
        inters.reverse()
        opt = inters[0]
        for inter, decoder in zip(inters[1:], self.decoders):
            opt = decoder(opt, inter)
        logits = self.opt_conv(opt)
        logits = logits.sigmoid()
        opt = dict(con=logits)
        opt["dists"] = {}
        return opt


# def main():
#     B, C, L, L = 1, 1, 100, 100
#     ipt = torch.rand(B, C, L, L)
# 
#     # unet = UNet([1, 32, 64, 128, 256, 512])
#     unet = UNet([1, 64, 20, 250, 100, 20])
#     opt = unet(ipt)
#     print(unet)
#     print(opt.shape)
# 
# 
# if __name__ == "__main__":
#     main()


