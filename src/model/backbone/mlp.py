import math
from .layers import *


class MLP(nn.Module):
    def __init__(self, data_size, hidden_size, timestep_embedding_size, cond_embedding_size):
        super().__init__()
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.timestep_embedding_size = timestep_embedding_size
        self.cond_embedding_size = cond_embedding_size
        #
        # ins = [in_dim + 2 * n_frequencies] + h_dims
        # outs = h_dims + [out_dim]
        # self.n_frequencies = n_frequencies
        #
        # self.layers = nn.ModuleList([
        #     nn.Sequential(nn.Linear(in_d, out_d), nn.LeakyReLU()) for in_d, out_d in zip(ins, outs)
        # ])
        # self.top = nn.Sequential(nn.Linear(out_dim, out_dim))

        input_size = math.prod(data_size) + timestep_embedding_size + cond_embedding_size
        blocks = []
        for i in range(len(hidden_size)):
            blocks.append(nn.Linear(input_size, hidden_size[i]))
            blocks.append(nn.ReLU())
            input_size = hidden_size[i]
        self.blocks = nn.Sequential(*blocks)
        self.output_proj = nn.Linear(input_size, math.prod(data_size))

    def feature(self, x, timestep_embedding, cond_embedding):
        size = x.size()
        x = x.reshape(x.size(0), -1)
        x = torch.cat([x, timestep_embedding, cond_embedding], dim=-1)
        x = self.blocks(x)
        return x, size

    def output(self, x, size):
        x = self.output_proj(x)
        x = x.view(size)
        return x

    def forward(self, x, timestep_embedding=None, cond_embedding=None):
        x, size = self.feature(x, timestep_embedding, cond_embedding)
        x = self.output(x, size)
        return x


def mlp(cfg):
    data_size = cfg['data_size']
    hidden_size = cfg['mlp']['hidden_size']
    timestep_embedding_size = cfg['timestep_embedding_size']
    cond_embedding_size = cfg['cond_embedding_size']
    model = MLP(data_size, hidden_size, timestep_embedding_size, cond_embedding_size)
    return model
