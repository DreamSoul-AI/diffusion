import torch
from tqdm import tqdm
from model import OptimalTransport


class FlowSampler:
    def __init__(self, num_steps=100, guidance_scale=1.0, eta=0.0, normalize=False):
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.normalize = normalize

    def apply_normalize(self, data, low, high):
        data.clamp_(min=low, max=high)
        data.sub_(low).div_(max(high - low, 1e-5))
        return data

    def sample(self, noise, model, classes=None):
        if isinstance(model.core, OptimalTransport):
            samples = self._sample('ot', noise, model, classes)
        else:
            raise ValueError('Not valid model')
        if self.normalize:
            samples = self.apply_normalize(samples, -1, 1)
        return samples

    @torch.no_grad()
    def _sample(self, mode, z, model, classes=None):
        """Draws samples from a model given starting noise for the Epsilon objective."""
        model.train(False)
        ts = z.new_ones([z.shape[0]])

        # Define timesteps and compute alphas and sigmas based on the schedule
        if mode == 'ot':
            # t = torch.linspace(0, 1, self.num_steps + 1, device=z.device)
            t = torch.linspace(0, 1, self.num_steps + 1, device=z.device)
        else:
            raise ValueError('Not valid mode')

        for i in tqdm(range(self.num_steps)):
            # if model.core.is_cond and self.guidance_scale > 1 and classes is not None:
            #     x_0 = torch.cat([z, z])  # Duplicate input for unconditional and conditional
            #     cond = torch.cat([-torch.ones_like(classes), classes])  # Classifier-free guidance
            #     uncond, cond = model.core.decode(x_0, t[i], t[i + 1], cond=cond).float().chunk(2)
            #     pred = uncond + self.guidance_scale * (cond - uncond)
            # else:
            #     x_0 = z
            # cond = -z.new_ones((z.size(0),), dtype=torch.long)
            #     # pred = model(input)['data'].float()
            #     pred = model.core.decode(x_0, t[i], t[i + 1], cond=cond)
            # cond = torch.cat([-torch.ones_like(classes), classes])  # Classifier-free guidance

            if model.core.is_cond and self.guidance_scale > 1:
                x_0 = torch.cat([z, z])  # Duplicate input for unconditional and conditional
                ts = t[i].repeat(x_0.shape[0])
                cond = torch.cat([-torch.ones_like(classes), classes])  # Classifier-free guidance
                uncond, cond = model.core.forward_diffusion_pass(x_0, ts, cond=cond).float().chunk(2)
                pred = uncond + self.guidance_scale * (cond - uncond)
                z += pred * 1 / self.num_steps
            else:
                ts = t[i].repeat(z.shape[0])
                cond = -z.new_ones((z.size(0),), dtype=torch.long)
                pred = model.core.forward_diffusion_pass(z, ts, cond=cond)
                z += pred * 1 / self.num_steps

            # https://github.com/facebookresearch/flow_matching/blob/main/examples/standalone_flow_matching.ipynb
            # ODE solver
            # https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/ode_solver.py
            # https://github.com/rtqichen/torchdiffeq
            # mid-point
            # t_start = t[i]
            # t_end = t[i + 1]  # TODO: add to use a solver, and a unified api also for diffusion
            # z_1 = z + model.core.forward_diffusion_pass(z, t_start.repeat(z.shape[0]), cond=classes) * (
            #         t_end - t_start) / 2
            # z_2 = z + model.core.forward_diffusion_pass(z_1, (t_start + (t_end - t_start) / 2).repeat(z.shape[0]),
            #                                             cond=classes) * (t_end - t_start)
            # z = z_2

        # with torch.no_grad():
        #     x_t = [torch.randn(n_samples, 2, device=device)]
        #     for t in range(len(t_steps) - 1):
        #         x_t += [v_t.decode_t0_t1(x_t[-1], t_steps[t], t_steps[t + 1])]

        # pad predictions
        # x_t = [x_t[0]] * 10 + x_t + [x_t[-1]] * 10

        # # Sampling
        # N_SAMPLES = 10_000
        # N_STEPS = 100
        # t_steps = torch.linspace(0, 1, N_STEPS, device=device)
        # with torch.no_grad():
        #     x_t = [torch.randn(n_samples, 2, device=device)]
        #     for t in range(len(t_steps) - 1):
        #         x_t += [v_t.decode_t0_t1(x_t[-1], t_steps[t], t_steps[t + 1])]
        #
        # # pad predictions
        # x_t = [x_t[0]] * 10 + x_t + [x_t[-1]] * 10
        #
        # x_t_numpy = np.array([x.detach().cpu().numpy() for x in x_t])
        # filename = f"{DATASET}_{MODEL}_{N_SAMPLES}_{N_STEPS}.npy"
        # np.save(filename, x_t_numpy)
        return z
