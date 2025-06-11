from .layers import *
from ..model import init_param


class MLP(nn.Module):
    def __init__(self, data_size, hidden_size, activation, time_embedding_size, cond_embedding_size):
        super().__init__()
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.time_embedding_size = time_embedding_size
        self.cond_embedding_size = cond_embedding_size

        input_size = math.prod(data_size) + time_embedding_size + cond_embedding_size
        blocks = []
        for i in range(len(hidden_size)):
            blocks.append(nn.Linear(input_size, hidden_size[i]))
            blocks.append(Activation(activation))
            input_size = hidden_size[i]
        self.blocks = nn.Sequential(*blocks)
        self.output_proj = nn.Linear(input_size, math.prod(data_size))

    def feature(self, x, time_embedding=None, cond_embedding=None):
        size = x.size()
        x = x.reshape(x.size(0), -1)
        if time_embedding is not None:
            x = torch.cat([x, time_embedding], dim=1)
        if cond_embedding is not None:
            cond_embedding = expand_to_planes(cond_embedding, x.shape)
            x = torch.cat([x, cond_embedding], dim=1)
        x = self.blocks(x)
        return x, size

    def output(self, x, size):
        x = self.output_proj(x)
        x = x.view(size)
        return x

    def forward(self, x, time_embedding=None, cond_embedding=None):
        x, size = self.feature(x, time_embedding, cond_embedding)
        x = self.output(x, size)
        return x


def mlp(cfg):
    data_size = cfg['data_size']
    hidden_size = cfg['mlp']['hidden_size']
    activation = cfg['mlp']['activation']
    time_embedding_size = cfg['time_embedding_size']
    cond_embedding_size = cfg['cond_embedding_size']
    model = MLP(data_size, hidden_size, activation, time_embedding_size, cond_embedding_size)
    model.apply(init_param)
    return model
