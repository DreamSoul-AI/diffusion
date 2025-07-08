from .layers import *
from ..model import init_param


class StructuredHyperNetwork(nn.Module):
    def __init__(self, time_embed_dim, d, n_layers=2):
        super().__init__()
        self.d = d
        self.time_embed_dim = time_embed_dim

        # Project time embedding to (B, d, d)
        self.initial_proj = nn.Linear(time_embed_dim, d * d)

        # Layers operating on (B, d, d)
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.LayerNorm([d, d]),
                nn.Linear(d, d),  # Applies to last dim
                nn.GELU(),
                nn.Linear(d, d)
            ) for _ in range(n_layers)
        ])

    def forward(self, time_embed):
        """
        Args:
            time_embed: Tensor of shape (B, d')
        Returns:
            generated_weight: Tensor of shape (B, d, d)
        """
        B = time_embed.size(0)
        x = self.initial_proj(time_embed).view(B, self.d, self.d)  # (B, d, d)
        x = self.layers(x)  # Structured processing
        return x  # This will be used as a generated weight matrix


class HyperMLP(nn.Module):
    def __init__(self, image_input_size, d, hidden_size, activation):
        super().__init__()
        self.d = d
        self.image_embed = nn.Linear(image_input_size, d)
        self.activation = Activation(activation)

        # MLP block after applying hyper weight
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden_size),
            Activation(activation),
            nn.Linear(hidden_size, image_input_size)
        )

    def forward(self, x, hyper_weight):
        """
        Args:
            x: image tensor of shape (B, k)
            hyper_weight: tensor of shape (B, d, d)
        """
        x = self.image_embed(x)  # (B, d)
        x = torch.bmm(hyper_weight, x.unsqueeze(-1)).squeeze(-1)  # (B, d)
        x = self.activation(x)
        x = self.mlp(x)  # (B, k)
        return x


def hypermlp(cfg):
    data_size = cfg['data_size']
    hidden_size = cfg['mlp']['hidden_size']
    activation = cfg['mlp']['activation']
    time_embedding_size = cfg['time_embedding_size']
    cond_embedding_size = cfg['cond_embedding_size']
    model = HyperMLP(data_size, hidden_size, activation, time_embedding_size, cond_embedding_size)
    model.apply(init_param)
    return model
