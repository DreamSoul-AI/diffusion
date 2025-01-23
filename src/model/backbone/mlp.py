from .layers import *


class MLP(nn.Module):
    # def __init__(self, in_dim, out_dim, h_dims, n_frequencies):
    def __init__(self, data_shape, hidden_size, n_frequencies):
        super().__init__()
        self.data_shape = data_shape
        self.hidden_size = hidden_size

        ins = [in_dim + 2 * n_frequencies] + h_dims
        outs = h_dims + [out_dim]
        self.n_frequencies = n_frequencies

        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(in_d, out_d), nn.LeakyReLU()) for in_d, out_d in zip(ins, outs)
        ])
        self.top = nn.Sequential(nn.Linear(out_dim, out_dim))

    # # TODO: improve the way we do it sinusoidal position embedding (check the fourier embedding)
    # def time_encoder(self, t):
    #     freq = 2 * torch.arange(self.n_frequencies, device=t.device) * torch.pi
    #     t = freq * t[..., None]
    #     return torch.cat((t.cos(), t.sin()), dim=-1)

    def forward(self, x):
        # t = self.time_encoder(t)
        # x = torch.cat((x, t), dim=-1)
        for l in self.layers:
            x = l(x)
        x = self.top(x)
        return x


def mlp(cfg):
    data_shape = cfg['data_shape']
    hidden_size = cfg['diffusion']['hidden_size']
    model = MLP(data_shape, hidden_size)
    return model
