import math
from model.model import *
from zuko.utils import odeint


class Flow:

    def __init__(self, sig_min=0.001):
        super().__init__()
        self.sig_min = sig_min
        self.eps = 1e-5

    def psi_t(self, x, x_1, t):
        """ Conditional Flow
        """
        return (1 - (1 - self.sig_min) * t) * x + t * x_1

    def loss(self, v_t, x_1):
        """ Compute loss
        """
        # t ~ Unif([0, 1])
        # TODO: why add? different samples for each data points, similar to torch.quasirandom.SobolEngine
        t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
        t = t[:, None].expand(x_1.shape)
        # x ~ p_t(x_0)
        x_0 = torch.randn_like(x_1)
        v_psi = v_t(t[:, 0], self.psi_t(x_0, x_1, t))
        d_psi = x_1 - (1 - self.sig_min) * x_0
        loss = torch.mean((v_psi - d_psi) ** 2)  # TODO: improve mse loss
        return loss

    # def forward(self, input):
    #     x_0 = input['data']
    #     cond = input['target']
    #     if 'training' in input:
    #         training = input['training']
    #     else:
    #         training = self.training
    #     if training:
    #         t = self.rng.draw(x_0.shape[0])[:, 0].to(x_0.device)
    #     else:
    #         t = input['t']
    #     output = {}
    #     output['data'], output['loss'] = self.core(x_0, t, cond, training)
    #     return output


class CondVF(nn.Module):
    def __init__(self, net, n_steps=100):
        super().__init__()
        self.net = net

    def forward(self, t, x):
        return self.net(t, x)

    def wrapper(self, t, x):  # TODO: merge with forward
        t = t * torch.ones(len(x), device=x.device)
        return self(t, x)

    def decode_t0_t1(self, x_0, t0, t1):  # TODO: merge with decode
        return odeint(self.wrapper, x_0, t0, t1, self.parameters())

    def encode(self, x_1):  # TODO: not used, add t0, t1 option
        return odeint(self.wrapper, x_1, 1., 0., self.parameters())

    def decode(self, x_0):
        return odeint(self.wrapper, x_0, 0., 1., self.parameters())


class Net(nn.Module):
    def __init__(self, in_dim, out_dim, h_dims, n_frequencies):
        super().__init__()

        ins = [in_dim + 2 * n_frequencies] + h_dims
        outs = h_dims + [out_dim]
        self.n_frequencies = n_frequencies

        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(in_d, out_d), nn.LeakyReLU()) for in_d, out_d in zip(ins, outs)
        ])
        self.top = nn.Sequential(nn.Linear(out_dim, out_dim))

    # TODO: improve the way we do it sinusoidal position embedding (check the fourier embedding)
    def time_encoder(self, t):
        freq = 2 * torch.arange(self.n_frequencies, device=t.device) * torch.pi
        t = freq * t[..., None]
        return torch.cat((t.cos(), t.sin()), dim=-1)

    def forward(self, t, x):
        t = self.time_encoder(t)
        x = torch.cat((x, t), dim=-1)

        for l in self.layers:
            x = l(x)
        x = self.top(x)
        return x


def flow(core, cfg):
    model = Flow(core)
    # model.apply(init_param)
    return model

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
