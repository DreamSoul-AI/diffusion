import torch
import torch.nn as nn
import math


class Activation(nn.Module):
    def __init__(self, activation='relu', inplace=False):
        super().__init__()
        if activation == 'relu':
            activation = nn.ReLU(inplace=inplace)
        elif activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'silu':
            activation = nn.SiLU()
        elif activation == 'elu':
            activation = nn.ELU()
        elif activation == 'gelu':
            activation = nn.GELU()
        elif activation == 'none':
            activation = nn.Identity()
        else:
            raise ValueError('Not valid activation')
        self.activation = activation

    def forward(self, x):
        return self.activation(x)


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True) if not is_last else nn.Identity(),
            nn.ReLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)


class SelfAttention2d(nn.Module):
    def __init__(self, c_in, n_head=1, dropout_rate=0.1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True)

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3] ** -0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.dropout(self.out_proj(y))


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


def expand_to_planes(input, shape, repeat_batch=False):
    """
    Expand input to match the spatial dimensions of shape.
    Optionally repeat across the batch dimension if repeat_batch is True.
    Handles cases where input does not have spatial dimensions.
    """
    # If the batch dimension needs to be repeated to match the target batch size
    if repeat_batch and input.shape[0] == 1:
        # Expand the batch dimension without extra repetitions
        input = input.expand(shape[0], -1)

    # Add spatial dimensions and repeat as necessary to match `shape`
    if input.dim() == 2:  # Assuming input is [batch_size, channels]
        input = input[:, :, None, None]  # Add spatial dimensions: [batch_size, channels, 1, 1]

    # Repeat spatial dimensions to match the target shape (height and width)
    output = input.expand(-1, -1, shape[2], shape[3])
    return output
