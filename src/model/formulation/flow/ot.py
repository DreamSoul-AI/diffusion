from zuko.utils import odeint
from model.model import *
from model.backbone import FourierFeatures
# from ..utils import expand_to_planes


class OptimalTransport(nn.Module):
    def __init__(self, backbone, target_size, class_dropout, sig_min=1e-3, n_steps=100):
        super().__init__()
        self.backbone = backbone
        self.target_size = target_size
        self.class_dropout = class_dropout
        # TODO: need unify time embedding
        self.timestep_embed = FourierFeatures(1, 16)
        self.class_embed = nn.Embedding(self.target_size + 1, 4)
        self.sig_min = 1e-3

    def psi_t(self, x, x_1, t, sig_min=1e-3):
        """ Conditional Flow
        """
        return (1 - (1 - sig_min) * t) * x + t * x_1

    def wrapper(self, x, t):  # TODO: merge with forward
        t = t * torch.ones(len(x), device=x.device)
        return self(x, t)

    # def decode_t0_t1(self, x_0, t0, t1):  # TODO: merge with decode
    #     return odeint(self.wrapper, x_0, t0, t1, self.parameters())

    def encode(self, x_1, t0=1., t1=0.):  # TODO: not used, add t0, t1 option
        return odeint(self.wrapper, x_1, t0, t1, self.parameters())

    def decode(self, x_0, t0=0., t1=1.):
        return odeint(self.wrapper, x_0, t0, t1, self.parameters())

    def forward_diffusion_pass(self, x_0, t, cond):  # TODO:rename function
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x_0.shape)
        class_embed = expand_to_planes(self.class_embed(cond + 1), x_0.shape)
        pred = self.backbone(torch.cat([x_0, class_embed, timestep_embed], dim=1))
        return pred

    def forward_diffusion_sample(self, x_1, t, classes):  # TODO: inverse x_0 and x_1 from diffusion
        # Calculate the noise schedule parameters for those timesteps
        # alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        # alphas = alphas[:, None, None, None]
        # sigmas = sigmas[:, None, None, None]
        t = t[:, None, None, None]
        x_0 = torch.randn_like(x_1)
        noised_reals = self.psi_t(x_0, x_1, t, self.sig_min)
        targets = x_1 - (1 - self.sig_min) * x_0

        # Drop out the class of the examples
        to_drop = torch.rand(classes.shape, device=classes.device).le(self.class_dropout)
        classes_drop = torch.where(to_drop, -torch.ones_like(classes), classes)
        return noised_reals, targets, classes_drop

    def forward(self, x_0, t, cond, training=True):
        # x_0 = torch.randn_like(x_1)
        # v_psi = v_t(t[:, 0], self.psi_t(x_0, x_1, t))
        # d_psi = x_1 - (1 - self.sig_min) * x_0
        # loss = torch.mean((v_psi - d_psi) ** 2)  # TODO: improve mse loss

        if training:
            noised_reals, targets, classes_drop = self.forward_diffusion_sample(x_0, t, cond)
            predicted_v = self.forward_diffusion_pass(noised_reals, t, classes_drop)
            output_target = predicted_v
            loss = F.mse_loss(output_target, targets)
        else:
            predicted_v = self.forward_diffusion_pass(x_0, t, cond)
            output_target = predicted_v
            loss = 0
        return output_target, loss


def ot(backbone, cfg):
    target_size = cfg['target_size']
    class_dropout = cfg['flow']['class_dropout']
    model = OptimalTransport(backbone, target_size, class_dropout)
    return model

# class Net(nn.Module):
#     def __init__(self, in_dim, out_dim, h_dims, n_frequencies):
#         super().__init__()
#
#         ins = [in_dim + 2 * n_frequencies] + h_dims
#         outs = h_dims + [out_dim]
#         self.n_frequencies = n_frequencies
#
#         self.layers = nn.ModuleList([
#             nn.Sequential(nn.Linear(in_d, out_d), nn.LeakyReLU()) for in_d, out_d in zip(ins, outs)
#         ])
#         self.top = nn.Sequential(nn.Linear(out_dim, out_dim))
#
#     # TODO: improve the way we do it sinusoidal position embedding (check the fourier embedding)
#     def time_encoder(self, t):
#         freq = 2 * torch.arange(self.n_frequencies, device=t.device) * torch.pi
#         t = freq * t[..., None]
#         return torch.cat((t.cos(), t.sin()), dim=-1)
#
#     def forward(self, x, t):
#         t = self.time_encoder(t)
#         x = torch.cat((x, t), dim=-1)
#
#         for l in self.layers:
#             x = l(x)
#         x = self.top(x)
#         return x


# def flow(core, cfg):
#     model = Flow(core)
#     # model.apply(init_param)
#     return model

## Training
# def get_model(name: str):
#     if name == "vp":
#         return VPDiffusionFlowMatching()
#     elif name == "ve":
#         return VEDiffusionFlowMatching()
#     if name == "ot":
#         return OTFlowMatching()
#
#
# MODEL = "ot"
# model = get_model(MODEL)
# net = Net(2, 2, [512] * 5, 10).to(device)
# v_t = CondVF(net)
#
# losses = []
# # configure optimizer
# optimizer = torch.optim.Adam(v_t.parameters(), lr=1e-3)
# n_epochs = 5000
#
# for epoch in tqdm(range(n_epochs), ncols=88):
#     for batch in dataloader:
#         x_1 = batch[0]
#         # compute loss
#         loss = model.loss(v_t, x_1)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         losses += [loss.detach()]


# Sampling
# N_SAMPLES = 10_000
# N_STEPS = 100
# t_steps = torch.linspace(0, 1, N_STEPS, device=device)
# with torch.no_grad():
#     x_t = [torch.randn(n_samples, 2, device=device)]
#     for t in range(len(t_steps)-1):
#       x_t += [v_t.decode_t0_t1(x_t[-1], t_steps[t], t_steps[t+1])]
#
# # pad predictions
# x_t = [x_t[0]]*10 + x_t + [x_t[-1]] * 10
#
# x_t_numpy = np.array([x.detach().cpu().numpy() for x in x_t])
# filename = f"{DATASET}_{MODEL}_{N_SAMPLES}_{N_STEPS}.npy"
# np.save(filename, x_t_numpy)
