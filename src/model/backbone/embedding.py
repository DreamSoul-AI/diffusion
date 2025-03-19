import torch
import torch.nn as nn
import math


class IdentityEmbedding(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        shape = list(input.shape)
        shape[-1] = self.output_size
        output = input.expand(*shape)
        return output


class FourierEmbedding(nn.Module):
    def __init__(self, input_size, output_size, std=1.):
        super().__init__()
        assert output_size % 2 == 0
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.randn([output_size // 2, input_size]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_size, embedding_mode):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding_mode = embedding_mode
        if embedding_mode == 'identity':
            self.time_embedding = IdentityEmbedding(self.embedding_size)
        elif embedding_mode == 'linear':
            self.time_embedding = FourierEmbedding(1, self.embedding_size)
        else:
            raise ValueError('Embedding mode {} not supported'.format(embedding_mode))

    def forward(self, input):
        if input.dim() == 1:
            input = input.unsqueeze(-1)
        output = self.time_embedding(input)
        return output
