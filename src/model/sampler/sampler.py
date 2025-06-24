import torch
from tqdm import tqdm
from model import OptimalTransport, AngularTransport, GaussianTransport
from model.backbone import expand_shape

class Sampler:
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
        elif isinstance(model.core, AngularTransport):
            samples = self._sample('at', noise, model, classes)
        elif isinstance(model.core, GaussianTransport):
            samples = self._sample('gt', noise, model, classes)
        else:
            raise ValueError('Not valid model')
        if self.normalize:
            samples = self.apply_normalize(samples, -1, 1)
        return samples

    @torch.no_grad()
    def _sample(self, mode, z, model, classes=None):
        model.train(False)
        t = torch.linspace(0, 1, self.num_steps + 1, device=z.device)
        ts = z.new_ones([z.shape[0]])

        input = {}
        for i in tqdm(range(self.num_steps)):
            if model.core.is_cond and self.guidance_scale > 1 and classes is not None:
                input['data'] = torch.cat([z, z])  # Duplicate input for unconditional and conditional
                input['target'] = torch.cat([-torch.ones_like(classes), classes])  # Classifier-free guidance
                input['t'] = torch.cat([ts, ts]) * t[i]
                uncond, cond = model(input)['data'].chunk(2)
                pred = uncond + self.guidance_scale * (cond - uncond)
            else:
                input['data'] = z
                input['target'] = -z.new_ones((z.size(0),), dtype=torch.long)
                input['t'] = ts * t[i]
                pred = model.core.forward_diffusion_pass(z, ts * t[i])

            v = pred
            x0 = model.core.predict_x0(z, v, t[i])
            x1 = model.core.predict_x1(z, v, t[i])

            if i < self.num_steps - 1:
                if self.eta > 0 and mode in ['at', 'gt']:
                    ddim_sigma = (model.core.sigma(t[i + 1]) ** 2 / model.core.sigma(t[i]) ** 2).sqrt() * \
                                 (1 - model.core.alpha(t[i]) ** 2 / model.core.alpha(t[i + 1]) ** 2).sqrt()
                    adjusted_sigma = (model.core.sigma(t[i + 1]) ** 2 - ddim_sigma ** 2).sqrt()

                    t = expand_shape(t, x0.size())
                    z = x1 * model.core.alpha(t[i + 1]) + x0 * adjusted_sigma
                    z += torch.randn_like(z) * ddim_sigma
                else:
                    z = model.core.make_z(x0, x1, t[i + 1])
        return x1
