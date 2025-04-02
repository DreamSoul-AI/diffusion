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
            t = torch.linspace(0, 1, self.num_steps + 1, device=z.device)
        else:
            raise ValueError('Not valid mode')

        for i in tqdm(range(self.num_steps)):
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
        return z
