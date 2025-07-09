from .layers import *
from ..model import init_param
from .embedding import DataEmbedding
from .patch import Patchify, Reconstruct


class HyperNetwork(nn.Module):
    def __init__(self, time_embedding_size, hidden_size):
        super().__init__()
        self.time_embedding_size = time_embedding_size
        self.hidden_size = hidden_size

        self.initial_proj = nn.Linear(time_embedding_size, hidden_size * hidden_size)

        # Layers operating on (B, d, d)
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.LayerNorm([d, d]),
                nn.Linear(d, d),  # Applies to last dim
                nn.GELU(),
                nn.Linear(d, d)
            ) for _ in range(n_layers)
        ])

    def forward(self, time_embedding):
        time_embedding = time_embedding
        B = time_embed.size(0)
        x = self.initial_proj(time_embed).view(B, self.d, self.d)  # (B, d, d)
        x = self.layers(x)  # Structured processing
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
        patch_size = (4, 4)
        dim_index = 1
        self.patchify = Patchify(patch_size=patch_size, dim_index=dim_index)
        self.reconstruct = Reconstruct(data_size=[1] + list(data_size), dim_index=dim_index)
        self.data_embedding = DataEmbedding(math.prod(patch_size), hidden_size)

    def forward(self, x):
        print(x.size())
        x = self.patchify(x)
        x = self.data_embedding(x)
        print(x.size())
        exit()

        x = torch.bmm(hyper_weight, x.unsqueeze(-1)).squeeze(-1)
        x = self.activation(x)
        x = self.mlp(x)
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
