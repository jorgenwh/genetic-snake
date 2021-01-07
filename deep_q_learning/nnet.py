import torch
import torch.nn as nn

class Neural_Network(nn.Module):
    def __init__(self, args):
        super(Neural_Network, self).__init__()
        self.args = args

        self.conv_block1 = self.conv_block(3, 256, kernel_size=3, stride=1, padding=self.same_padding(3))
        self.conv_block2 = self.conv_block(256, 256, kernel_size=3, stride=1, padding=self.same_padding(3))
        self.conv_block3 = self.conv_block(256, 256, kernel_size=3, stride=1, padding=self.same_padding(3))

        self.fc1 = self.linear_block(256 * self.args.size ** 2, 512)
        self.out = nn.Linear(512, 4)

    def forward(self, s):
        s = s.reshape(s.shape[0], 3, self.args.size, self.args.size)
        r = self.conv_block1(s)
        r = self.conv_block2(r)
        r = self.conv_block3(r)
        r = r.reshape(r.shape[0], -1)
        r = self.fc1(r)
        out = self.out(r)

        return out

    def same_padding(self, kernel_size):
        return kernel_size // 2

    def conv_size_out(self, size, kernel_size, stride, padding=0):
        size += padding*2
        return (size - (kernel_size - 1) - 1) // stride + 1

    def conv_block(self, in_channels, out_channels, *args, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def linear_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )