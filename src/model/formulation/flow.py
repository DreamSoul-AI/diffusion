import math
from model.model import *
from model.backbone import TimeEmbedding, ConditionEmbedding, expand_shape


class Flow(nn.Module):
    def __init__(self, backbone, target_size, class_dropout, time_embedding_mode, time_embedding_size,
                 cond_embedding_size, regularization):
        super().__init__()
        self.backbone = backbone
        self.target_size = target_size
        self.class_dropout = class_dropout
        self.time_embedding = TimeEmbedding(time_embedding_mode, time_embedding_size)
        self.cond_embedding = ConditionEmbedding(self.target_size + 1, cond_embedding_size, offset=1)
        self.regularization = regularization
        self.step_size = 1e-2

    @property
    def is_time(self):
        return self.time_embedding.is_time

    @property
    def is_cond(self):
        return self.cond_embedding.is_cond

    def forward_diffusion_pass(self, z, t, cond=None):
        time_embedding = self.time_embedding(t)
        cond_embedding = self.cond_embedding(cond)
        pred = self.backbone(z, time_embedding, cond_embedding)
        return pred

    def make_x0(self, x):
        x0 = torch.randn_like(x)
        return x0

    def make_z(self, x0, x1, t):
        t = expand_shape(t, x0.size())
        z = self.sigma(t) * x0 + self.alpha(t) * x1
        return z

    def make_cond(self, classes):
        if self.is_cond:
            # Drop out the class of the examples
            to_drop = torch.rand(classes.shape, device=classes.device).le(self.class_dropout)
            cond = torch.where(to_drop, -torch.ones_like(classes), classes)
        else:
            cond = None
        return cond

    def add_noise(self, z, t):
        z += z + self.sigma(t) * self.make_x0(z)
        return z

    def forward(self, z, t, cond, training=True):
        if training:
            x1 = z
            x0 = self.make_x0(x1)
            z = self.make_z(x0, x1, t)
            v = self.make_v(x0, x1, t)
            cond = self.make_cond(cond)
            pred_v = self.forward_diffusion_pass(z, t, cond)
            loss_v = F.mse_loss(pred_v, v)

            pred_x0 = self.predict_x0(z, pred_v, t)
            loss_x0 = F.mse_loss(pred_x0, x0)

            pred_x1 = self.predict_x1(z, pred_v, t)
            loss_x1 = F.mse_loss(pred_x1, x1)

            loss = z.new_zeros(())
            if self.regularization['v'] > 0:
                loss += self.regularization['v'] * loss_v
            if self.regularization['x0'] > 0:
                loss += self.regularization['x0'] * loss_x0
            if self.regularization['x1'] > 0:
                loss += self.regularization['x1'] * loss_x1
            if self.regularization['consistency'] > 0:
                with torch.no_grad():
                    t_consistency = (t - self.step_size).clamp(min=0, max=1)
                    z_consistency = self.make_z(pred_x0, pred_x1, t_consistency)
                    pred_v_consistency = self.forward_diffusion_pass(z_consistency, t_consistency, cond)
                    pred_x1_consistency = self.predict_x1(z_consistency, pred_v_consistency, t_consistency)
                    pred_x1_consistency = pred_x1_consistency.detach()
                pred_x1 = self.predict_x1(z, pred_v, t)
                loss_consistency = F.mse_loss(pred_x1, pred_x1_consistency)
                loss += self.regularization['consistency'] * loss_consistency
            else:
                loss_consistency = torch.tensor([0])
        else:
            pred_v = self.forward_diffusion_pass(z, t, cond)
            loss, loss_v, loss_x0, loss_x1, loss_consistency = (torch.tensor([0]), torch.tensor([0]), torch.tensor([0]),
                                                                torch.tensor([0]), torch.tensor([0]))
        return pred_v, loss, loss_v, loss_x0, loss_x1, loss_consistency


class OptimalTransport(Flow):
    def __init__(self, backbone, target_size, class_dropout, time_embedding_size, time_embedding_mode,
                 cond_embedding_size, regularization):
        super().__init__(backbone, target_size, class_dropout, time_embedding_size, time_embedding_mode,
                         cond_embedding_size, regularization)

    def alpha(self, t):
        return t

    def sigma(self, t):
        return 1 - t

    def make_v(self, x0, x1, t):
        v = x1 - x0
        return v

    def predict_x0(self, z, v, t):
        t = expand_shape(t, z.size())
        x0 = z - t * v
        return x0

    def predict_x1(self, z, v, t):
        t = expand_shape(t, z.size())
        x1 = z + (1 - t) * v
        return x1


class AngularTransport(Flow):
    def __init__(self, backbone, target_size, class_dropout, time_embedding_size, time_embedding_mode,
                 cond_embedding_size, regularization):
        super().__init__(backbone, target_size, class_dropout, time_embedding_size, time_embedding_mode,
                         cond_embedding_size, regularization)

    def alpha(self, t):
        return torch.sin(t * math.pi / 2)

    def sigma(self, t):
        return torch.cos(t * math.pi / 2)

    def make_v(self, x0, x1, t):
        t = expand_shape(t, x0.size())
        v = math.pi / 2 * (self.sigma(t) * x1 - self.alpha(t) * x0)
        return v

    def predict_x0(self, z, v, t):
        t = expand_shape(t, z.size())
        x0 = self.sigma(t) * z - 2 / math.pi * self.alpha(t) * v
        return x0

    def predict_x1(self, z, v, t):
        t = expand_shape(t, z.size())
        x1 = self.alpha(t) * z + 2 / math.pi * self.sigma(t) * v
        return x1


class GaussianTransport(Flow):
    def __init__(self, backbone, target_size, class_dropout, time_embedding_size, time_embedding_mode,
                 cond_embedding_size, regularization):
        super().__init__(backbone, target_size, class_dropout, time_embedding_size, time_embedding_mode,
                         cond_embedding_size, regularization)

    def alpha(self, t):
        return torch.sqrt(t)

    def sigma(self, t):
        return torch.sqrt(1 - t)

    def make_v(self, x0, x1, t):
        t = expand_shape(t, x0.size())
        v = 1 / 2 * (1 / self.alpha(t) * x1 - 1 / self.sigma(t) * x0)
        return v

    def predict_x0(self, z, v, t):
        t = expand_shape(t, z.size())
        x0 = self.sigma(t) * (z - 2 * (self.alpha(t) ** 2) * v)
        return x0

    def predict_x1(self, z, v, t):
        t = expand_shape(t, z.size())
        x1 = self.alpha(t) * (z + 2 * (self.sigma(t) ** 2) * v)
        return x1


def ot(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['flow']['class_dropout']
    regularization = cfg['flow']['regularization']
    time_embedding_mode = cfg['time_embedding_mode']
    time_embedding_size = cfg['time_embedding_size']
    cond_embedding_size = cfg['cond_embedding_size']
    model = OptimalTransport(backbone, target_size, class_dropout, time_embedding_mode, time_embedding_size,
                             cond_embedding_size, regularization)
    return model


def at(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['flow']['class_dropout']
    regularization = cfg['flow']['regularization']
    time_embedding_mode = cfg['time_embedding_mode']
    time_embedding_size = cfg['time_embedding_size']
    cond_embedding_size = cfg['cond_embedding_size']
    model = AngularTransport(backbone, target_size, class_dropout, time_embedding_mode, time_embedding_size,
                             cond_embedding_size, regularization)
    return model


def gt(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['flow']['class_dropout']
    regularization = cfg['flow']['regularization']
    time_embedding_mode = cfg['time_embedding_mode']
    time_embedding_size = cfg['time_embedding_size']
    cond_embedding_size = cfg['cond_embedding_size']
    model = GaussianTransport(backbone, target_size, class_dropout, time_embedding_mode, time_embedding_size,
                              cond_embedding_size, regularization)
    return model
