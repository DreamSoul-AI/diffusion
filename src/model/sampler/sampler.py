import torch
from tqdm import tqdm
from model import get_alphas_sigmas, V


class Sampler:
    def __init__(self, num_steps=100, guidance_scale=1.0, eta=0.0):
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.eta = eta

    def sample(self, noise, model, classes=None):
        if isinstance(model.core, V):
            samples = self.sample_v(noise, model, classes)
        else:
            raise NotImplementedError
        return samples

    @torch.no_grad()
    def sample_v(self, x, model, classes=None):
        """Draws samples from a model given starting noise."""
        model.train(False)
        ts = x.new_ones([x.shape[0]])

        t = torch.linspace(1, 0, self.num_steps + 1)[:-1].to(x.device)
        alphas, sigmas = get_alphas_sigmas(t)

        input = {}
        pred = None
        # The sampling loop
        for i in tqdm(range(self.num_steps)):
            if self.guidance_scale > 1 and classes is not None:
                # with torch.amp.autocast(cfg['device']):
                x_in = torch.cat([x, x])
                ts_in = torch.cat([ts, ts])
                classes_in = torch.cat([-torch.ones_like(classes), classes])
                input['data'] = x_in
                input['target'] = classes_in
                input['t'] = ts_in * t[i]
                v_uncond, v_cond = model(input)['data'].float().chunk(2)
                v = v_uncond + self.guidance_scale * (v_cond - v_uncond)
            else:
                input['data'] = x
                input['target'] = -x.new_ones((x.size(0),), dtype=torch.long)
                input['t'] = ts * t[i]
                v = model(input)['data'].float()

            # Predict the noise and the denoised image
            pred = x * alphas[i] - v * sigmas[i]
            eps = x * sigmas[i] + v * alphas[i]

            # If we are not on the last timestep, compute the noisy image for the
            # next timestep.
            if i < self.num_steps - 1:
                # If eta > 0, adjust the scaling factor for the predicted noise
                # downward according to the amount of additional noise to add
                ddim_sigma = self.eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt() * \
                             (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
                adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()

                # Recombine the predicted noise and predicted denoised image in the
                # correct proportions for the next step
                x = pred * alphas[i + 1] + eps * adjusted_sigma  # ddim eq(12)

                # Add the correct amount of fresh noise
                if self.eta:
                    x += torch.randn_like(x) * ddim_sigma
        # If we are on the last timestep, output the denoised image
        return pred
