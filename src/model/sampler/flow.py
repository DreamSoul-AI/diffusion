import torch
from tqdm import tqdm
from model import OptimalTransport, VariancePreserve, VarianceExplode


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
        elif isinstance(model.core, VariancePreserve):
            samples = self._sample('vp', noise, model, classes)
        elif isinstance(model.core, VarianceExplode):
            samples = self._sample('ve', noise, model, classes)
        else:
            raise ValueError('Not valid model')
        if self.normalize:
            samples = self.apply_normalize(samples, -1, 1)
        return samples

    @torch.no_grad()
    def _sample(self, mode, z, model, classes=None):
        """Draws samples from a model given starting noise for the Epsilon objective."""
        model.train(False)

        # Define timesteps and compute alphas and sigmas based on the schedule
        if mode in ['ot', 'vp', 've']:
            # t = torch.linspace(0, 1, self.num_steps + 1, device=z.device)
            t = torch.linspace(1, 0, self.num_steps, device=z.device)
        else:
            raise ValueError('Not valid mode')

        import math
        alphas, sigmas = torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
        ts = z.new_ones([z.shape[0]])

        x = None
        input = {}
        # The sampling loop
        for i in tqdm(range(self.num_steps)):
            if model.core.is_cond and self.guidance_scale > 1 and classes is not None:
                input['data'] = torch.cat([z, z])  # Duplicate input for unconditional and conditional
                input['target'] = torch.cat([-torch.ones_like(classes), classes])  # Classifier-free guidance
                input['t'] = torch.cat([ts, ts]) * t[i]
                uncond, cond = model(input)['data'].float().chunk(2)
                pred = uncond + self.guidance_scale * (cond - uncond)
            else:
                input['data'] = z
                input['target'] = -z.new_ones((z.size(0),), dtype=torch.long)
                input['t'] = ts * t[i]
                # pred = model(input)['data'].float()
                pred = model.core.forward_diffusion_pass(z, ts * t[i])
                # cond = -z.new_ones((z.size(0),), dtype=torch.long)
                # pred = model.core.forward_diffusion_pass(z, ts, cond=cond)

            # v = pred
            # x = z * alphas[i] - v * sigmas[i]
            # eps = z * sigmas[i] + v * alphas[i]
            v = pred
            x = z * alphas[i] - v * sigmas[i]
            eps = z * sigmas[i] + v * alphas[i]

            if i < self.num_steps - 1:
                ddim_sigma = self.eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt() * \
                             (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
                adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()
                z = x * alphas[i + 1] + eps * adjusted_sigma
                # Add noise if eta > 0
                if self.eta:
                    z += torch.randn_like(z) * ddim_sigma

        # x = None
        # for i in tqdm(range(self.num_steps)):
        #     if model.core.is_cond and self.guidance_scale > 1 and classes is not None:
        #         z_ = torch.cat([z, z])  # Duplicate input for unconditional and conditional
        #         ts = t[i].repeat(z_.shape[0])
        #         cond = torch.cat([-torch.ones_like(classes), classes])  # Classifier-free guidance
        #         uncond, cond = model.core.forward_diffusion_pass(z_, ts, cond=cond).float().chunk(2)
        #         pred = uncond + self.guidance_scale * (cond - uncond)
        #         z += pred * 1 / self.num_steps
        #     else:
        #         ts = ts * t[i]
        #         cond = -z.new_ones((z.size(0),), dtype=torch.long)
        #         pred = model.core.forward_diffusion_pass(z, ts, cond=cond)
        #         v = pred
        #         x = z * alphas[i] - v * sigmas[i]
        #         eps = z * sigmas[i] + v * alphas[i]
        #         if i < self.num_steps - 1:
        #             ddim_sigma = self.eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt() * \
        #                          (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
        #             adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()
        #             z = x * alphas[i + 1] + eps * adjusted_sigma
        #         # z += pred * 1 / self.num_steps
        #     print(x)
        return x
