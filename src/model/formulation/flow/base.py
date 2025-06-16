import math
from model.model import *
from model.backbone import TimeEmbedding, ConditionEmbedding, expand_shape


class Base(nn.Module):
    def __init__(self, backbone, target_size, class_dropout, time_embedding_mode, time_embedding_size,
                 cond_embedding_size):
        super().__init__()
        self.backbone = backbone
        self.target_size = target_size
        self.class_dropout = class_dropout
        self.time_embedding = TimeEmbedding(time_embedding_mode, time_embedding_size)
        self.cond_embedding = ConditionEmbedding(self.target_size + 1, cond_embedding_size)

    @property
    def is_time(self):
        return self.time_embedding.is_time

    @property
    def is_cond(self):
        return self.cond_embedding.is_cond

    def forward_diffusion_pass(self, z, t, cond=None):
        time_embedding = self.time_embedding(t[:, None])  # TODO: make this adapt
        if self.is_cond and cond is not None:
            cond_embedding = self.cond_embedding(cond + 1)
        else:
            cond_embedding = None
        pred = self.backbone(z, time_embedding, cond_embedding)
        return pred

    def make_x0(self, x):
        x_0 = torch.randn_like(x)
        return x_0

    def make_z(self, x_0, x_1, t):
        t = expand_shape(t, x_0.size())
        z = self.sigma(t) * x_0 + self.alpha(t) * x_1
        return z

    def make_cond(self, classes):
        if self.is_cond:
            # Drop out the class of the examples
            to_drop = torch.rand(classes.shape, device=classes.device).le(self.class_dropout)
            cond = torch.where(to_drop, -torch.ones_like(classes), classes)
        else:
            cond = None
        return cond

    def forward(self, z, t, cond, training=True):
        if training:
            x_1 = z
            x_0 = self.make_x0(x_1)
            z = self.make_z(x_0, x_1, t)
            v = self.make_v(x_0, x_1, t)
            cond = self.make_cond(cond)
            pred_v = self.forward_diffusion_pass(z, t, cond)
            loss = F.mse_loss(pred_v, v)
        else:
            pred_v = self.forward_diffusion_pass(z, t, cond)
            loss = 0
        return pred_v, loss


class OptimalTransport(Base):
    def __init__(self, backbone, target_size, class_dropout, time_embedding_size, time_embedding_mode,
                 cond_embedding_size):
        super().__init__(backbone, target_size, class_dropout, time_embedding_size, time_embedding_mode,
                         cond_embedding_size)

    def alpha(self, t):
        return t

    def sigma(self, t):
        return 1 - t

    def make_v(self, x_0, x_1, t):
        targets = x_1 - x_0
        return targets

    def predict_x0(self, z, v, t):
        t = expand_shape(t, z.size())
        x_0 = z - t * v
        return x_0

    def predict_x1(self, z, v, t):
        t = expand_shape(t, z.size())
        x_1 = z + (1 - t) * v
        return x_1


class VariancePreserve(Base):

    def __init__(self, backbone, target_size, class_dropout, time_embedding_size, time_embedding_mode,
                 cond_embedding_size):
        super().__init__(backbone, target_size, class_dropout, time_embedding_size, time_embedding_mode,
                         cond_embedding_size)

    def alpha(self, t):
        return torch.sin(t * math.pi / 2)

    def sigma(self, t):
        return torch.cos(t * math.pi / 2)

    def make_v(self, x_0, x_1, t):
        t = expand_shape(t, x_0.size())
        targets = math.pi / 2 * (self.sigma(t) * x_1 - self.alpha(t) * x_0)
        return targets

    def predict_x0(self, z, v, t):
        t = expand_shape(t, z.size())
        x_0 = self.sigma(t) * z - 2 / math.pi * self.alpha(t) * v
        return x_0

    def predict_x1(self, z, v, t):
        t = expand_shape(t, z.size())
        x_1 = self.alpha(t) * z + 2 / math.pi * self.sigma(t) * v
        return x_1


def ot(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['flow']['class_dropout']
    time_embedding_mode = cfg['time_embedding_mode']
    time_embedding_size = cfg['time_embedding_size']
    cond_embedding_size = cfg['cond_embedding_size']
    model = OptimalTransport(backbone, target_size, class_dropout, time_embedding_mode, time_embedding_size,
                             cond_embedding_size)
    return model


def vp(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['flow']['class_dropout']
    time_embedding_mode = cfg['time_embedding_mode']
    time_embedding_size = cfg['time_embedding_size']
    cond_embedding_size = cfg['cond_embedding_size']
    model = VariancePreserve(backbone, target_size, class_dropout, time_embedding_mode, time_embedding_size,
                             cond_embedding_size)
    return model
