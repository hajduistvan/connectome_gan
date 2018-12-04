"""
@author Muhammed Kocabas
github: https://github.com/mkocabas/CoordConv-pytorch
paper: https://arxiv.org/pdf/1807.03247.pdf

Modified so that it is compatible with connectivity matrices. Removed y coords
"""
import torch
import torch.nn as nn
class AddCoords(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / max((x_dim - 1), 1)
        yy_channel = yy_channel.float() / max((y_dim - 1), 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)


        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_coord=True, **kwargs):
        super().__init__()
        self.use_coord = use_coord
        if self.use_coord:
            self.addcoords = AddCoords()
            in_channels+=2
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        if self.use_coord:
            x = self.addcoords(x)
        x = self.conv(x)
        return x


class CoordConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, use_coord=True, **kwargs):
        super().__init__()
        self.use_coord = use_coord
        if self.use_coord:
            self.addcoords = AddCoords()
            in_channels+=2
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        if self.use_coord:
            x = self.addcoords(x)
        x = self.conv(x)
        return x

def pad_kernel_init(kernel, value=1e-10):
    shape = kernel.shape
    coord_shape = [shape[0], 2, shape[2], shape[3]]
    # coord_weights = torch.full(coord_shape, value, device=kernel.device, dtype=kernel.dtype)
    coord_weights = torch.randn(coord_shape).type(kernel.dtype).cuda()
    new_weight = torch.cat([kernel, coord_weights], 1)

    return new_weight


