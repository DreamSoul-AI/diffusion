from .layers import *
from ..model import init_param
from .embedding import DataEmbedding
from .patch import Patchify, Reconstruct


class HyperAttention(nn.Module):
    def __init__(self, time_embedding_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_proj = nn.Linear(time_embedding_size, hidden_size)

    def forward(self, time_embedding, x):
        time_embedding = time_embedding.unsqueeze(1)
        time_embedding = self.time_proj(time_embedding)
        x = x.matmul(time_embedding.transpose(-2, -1))
        attn = x.sigmoid()
        x = torch.matmul(attn, time_embedding)
        return x


class HyperMLP(nn.Module):
    def __init__(self, data_size, hidden_size, activation, time_embedding_size, cond_embedding_size):
        super().__init__()
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.time_embedding_size = time_embedding_size
        self.cond_embedding_size = cond_embedding_size

        input_size = math.prod(data_size)
        num_layers = 2
        patch_size = (1, 4, 4)
        dim_index = [1, 2, 3]
        self.patchify = Patchify(patch_size=patch_size, dim_index=dim_index)
        self.reconstruct = Reconstruct(data_size=[1] + list(data_size), dim_index=dim_index)
        self.data_embedding = DataEmbedding(math.prod(patch_size), hidden_size)
        self.attention = HyperAttention(time_embedding_size, hidden_size)
        blocks = []
        for i in range(num_layers):
            blocks.append(nn.Linear(hidden_size, hidden_size))
            blocks.append(Activation(activation))
        self.blocks = nn.Sequential(*blocks)
        self.output_proj = nn.Linear(hidden_size, 16)

    def forward(self, x, time_embedding=None, cond_embedding=None):
        x = self.patchify(x)
        patch_size = x.size()
        x = self.data_embedding(x)
        x = self.attention(time_embedding, x)
        x = self.output_proj(x)
        x = x.view(patch_size)
        x = self.reconstruct(x)
        return x


def hypermlp(cfg):
    data_size = cfg['data_size']
    hidden_size = cfg['hypermlp']['hidden_size']
    activation = cfg['hypermlp']['activation']
    time_embedding_size = cfg['time_embedding_size']
    cond_embedding_size = cfg['cond_embedding_size']
    model = HyperMLP(data_size, hidden_size, activation, time_embedding_size, cond_embedding_size)
    model.apply(init_param)
    return model
