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


# class FourierEmbedding(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         assert output_size % 2 == 0
#         self.input_size = input_size
#         self.output_size = output_size
#         self.weight = nn.Parameter(torch.randn([output_size // 2, input_size]))
#
#     def forward(self, input):
#         f = 2 * math.pi * input @ self.weight.T
#         return torch.cat([f.cos(), f.sin()], dim=-1)

class FourierEmbedding(nn.Module):
    def __init__(self, input_size, output_size, max_period=10000.):
        super().__init__()
        assert output_size % 2 == 0, "Output size must be even"
        self.input_size = input_size
        self.output_size = output_size
        self.max_period = max_period

        half_dim = output_size // 2
        freq_indices = torch.arange(half_dim, dtype=torch.float32) / half_dim
        self.register_buffer('freqs', torch.exp(-math.log(max_period) * freq_indices))

    def forward(self, input):
        args = input @ self.freqs.unsqueeze(0) * 2 * math.pi
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_mode, embedding_size):
        super().__init__()
        self.embedding_mode = embedding_mode
        self.embedding_size = embedding_size
        if self.embedding_size > 0:
            if embedding_mode == 'identity':
                self.time_embedding = IdentityEmbedding(self.embedding_size)
                self.is_time = False
            elif embedding_mode == 'fourier':
                self.time_embedding = FourierEmbedding(1, self.embedding_size)
                self.is_time = True
            else:
                raise ValueError('Embedding mode {} not supported'.format(embedding_mode))
        else:
            self.time_embedding = None

    def forward(self, input):
        if self.time_embedding is not None:
            if input.dim() == 1:
                input = input.unsqueeze(-1)
            embedding = self.time_embedding(input)
        else:
            embedding = None
        return embedding


class ConditionEmbedding(nn.Module):
    def __init__(self, num_embedding, embedding_size):
        super().__init__()
        self.num_embedding = num_embedding
        self.embedding_size = embedding_size
        if embedding_size > 0:
            self.cond_embedding = nn.Embedding(num_embedding, embedding_size)
            self.is_cond = True
        else:
            self.cond_embedding = None
            self.is_cond = False

    def forward(self, cond):
        if self.cond_embedding is not None and cond is not None:
            embedding = self.cond_embedding(cond + 1)
        else:
            embedding = None
        return embedding
