import torch
from tqdm import tqdm
from model import get_alphas_sigmas, X, Eps, V, Regularized


class DiffusionSampler:
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
        if isinstance(model.core, X):
            samples = self.sample_x(noise, model, classes)
        elif isinstance(model.core, Eps):
            samples = self.sample_eps(noise, model, classes)
        elif isinstance(model.core, (V, Regularized)):
            samples = self.sample_v(noise, model, classes)
        else:
            raise NotImplementedError
        if self.normalize:
            samples = self.apply_normalize(samples, -1, 1)
        return samples

    @torch.no_grad()
    def sample_x(self, z, model, classes=None):
        """Draws samples from a model given starting noise for the X_zero objective."""
        model.train(False)
        ts = z.new_ones([z.shape[0]])

        # Define timesteps and compute alphas and sigmas based on the schedule
        t = torch.linspace(1, 0, self.num_steps + 1)[:-1].to(z.device)
        alphas, sigmas = get_alphas_sigmas(t)

        input = {}
        x = None
        # The sampling loop
        for i in tqdm(range(self.num_steps)):
            if self.guidance_scale > 1 and classes is not None:
                x_in = torch.cat([z, z])  # Duplicate input for unconditional and conditional
                ts_in = torch.cat([ts, ts])
                classes_in = torch.cat([-torch.ones_like(classes), classes])  # Classifier-free guidance
                input['data'] = x_in
                input['target'] = classes_in
                input['t'] = ts_in * t[i]
                x_uncond, x_cond = model(input)['data'].float().chunk(2)
                x = x_uncond + self.guidance_scale * (x_cond - x_uncond)
            else:
                input['data'] = z
                input['target'] = -z.new_ones((z.size(0),), dtype=torch.long)
                input['t'] = ts * t[i]
                x = model(input)['data'].float()

            x = x
            eps = (z - x * alphas[i]) / sigmas[i]

            # If not on the last timestep, calculate the noisy image for the next timestep
            if i < self.num_steps - 1:
                ddim_sigma = self.eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt() * \
                             (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
                adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()

                # Recombine the predicted x_0 and the noise for the next step
                z = x * alphas[i + 1] + eps * adjusted_sigma

                # Add noise if eta > 0
                if self.eta:
                    z += torch.randn_like(z) * ddim_sigma

        # If on the last timestep, return the denoised image
        return x

    @torch.no_grad()
    def sample_eps(self, z, model, classes=None):
        """Draws samples from a model given starting noise for the Epsilon objective."""
        model.train(False)
        ts = z.new_ones([z.shape[0]])

        # Define timesteps and compute alphas and sigmas based on the schedule
        t = torch.linspace(1, 0, self.num_steps + 1)[1:].to(z.device)
        alphas, sigmas = get_alphas_sigmas(t)

        input = {}
        x = None
        # The sampling loop
        for i in tqdm(range(self.num_steps)):
            if self.guidance_scale > 1 and classes is not None:
                x_in = torch.cat([z, z])  # Duplicate input for unconditional and conditional
                ts_in = torch.cat([ts, ts])
                classes_in = torch.cat([-torch.ones_like(classes), classes])  # Classifier-free guidance
                input['data'] = x_in
                input['target'] = classes_in
                input['t'] = ts_in * t[i]
                eps_uncond, eps_cond = model(input)['data'].float().chunk(2)
                eps = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
            else:
                input['data'] = z
                input['target'] = -z.new_ones((z.size(0),), dtype=torch.long)
                input['t'] = ts * t[i]
                eps = model(input)['data'].float()

            x = (z - eps * sigmas[i]) / alphas[i]
            eps = eps

            # If not on the last timestep, calculate the noisy image for the next timestep
            if i < self.num_steps - 1:
                # if i > 0:
                ddim_sigma = self.eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt() * \
                             (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
                adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()

                # Recombine the denoised image and the noise for the next step
                z = x * alphas[i + 1] + eps * adjusted_sigma

                # Add noise if eta > 0
                if self.eta:
                    z += torch.randn_like(z) * ddim_sigma
        # If we are on the last timestep, output the denoised image
        return x

    @torch.no_grad()
    def sample_v(self, z, model, classes=None):
        """Draws samples from a model given starting noise."""
        model.train(False)
        ts = z.new_ones([z.shape[0]])

        # t = torch.linspace(1, 0, self.num_steps + 1)[:-1].to(z.device)
        t = torch.linspace(1, 0, self.num_steps + 1).to(z.device)
        alphas, sigmas = get_alphas_sigmas(t)

        input = {}
        x = None
        # The sampling loop
        for i in tqdm(range(self.num_steps)):
            if self.guidance_scale > 1 and classes is not None:
                z_in = torch.cat([z, z])  # Duplicate input for unconditional and conditional
                ts_in = torch.cat([ts, ts])
                classes_in = torch.cat([-torch.ones_like(classes), classes])
                input['data'] = z_in
                input['target'] = classes_in
                input['t'] = ts_in * t[i]
                v_uncond, v_cond = model(input)['data'].float().chunk(2)
                v = v_uncond + self.guidance_scale * (v_cond - v_uncond)
            else:
                input['data'] = z
                input['target'] = -z.new_ones((z.size(0),), dtype=torch.long)
                input['t'] = ts * t[i]
                v = model(input)['data'].float()

            x = z * alphas[i] - v * sigmas[i]
            eps = z * sigmas[i] + v * alphas[i]

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
                z = x * alphas[i + 1] + eps * adjusted_sigma  # ddim eq(12)

                # Add the correct amount of fresh noise
                if self.eta:
                    z += torch.randn_like(z) * ddim_sigma
        # If we are on the last timestep, output the denoised image
        return x

    # @torch.no_grad()
    # def sample_threelosses(self, z, model, classes=None):
    #     """
    #     Draws samples from the three-loss diffusion model.
    #
    #     Args:
    #         z: Initial noise tensor.
    #         model: The trained three-loss diffusion model.
    #         classes: Optional class labels for conditional generation.
    #
    #     Returns:
    #         The final generated sample.
    #     """
    #     model.eval()  # Set model to evaluation mode
    #     ts = z.new_ones([z.shape[0]])  # Timesteps tensor
    #     t = torch.linspace(1, 0, self.num_steps + 1).to(z.device)  # Diffusion schedule
    #     alphas, sigmas = get_alphas_sigmas(t)  # Compute alphas and sigmas
    #
    #     input = {}
    #     x = None  # Placeholder for the generated sample
    #
    #     for i in tqdm(range(self.num_steps)):
    #         # Handle classifier-free guidance if applicable
    #         if self.guidance_scale > 1 and classes is not None:
    #             z_in = torch.cat([z, z])  # Duplicate for unconditional and conditional inputs
    #             ts_in = torch.cat([ts, ts])
    #             classes_in = torch.cat([-torch.ones_like(classes), classes])
    #             input['data'] = z_in
    #             input['target'] = classes_in
    #             input['t'] = ts_in * t[i]
    #             outputs = model(input)['data']
    #             v_uncond, v_cond = outputs['v_predicted'].float().chunk(2)
    #             v = v_uncond + self.guidance_scale * (v_cond - v_uncond)
    #         else:
    #             input['data'] = z
    #             input['target'] = -z.new_ones((z.size(0),), dtype=torch.long)
    #             input['t'] = ts * t[i]
    #             v = model(input)['data'].float()
    #
    #         x = z * alphas[i] - v * sigmas[i]
    #         eps = z * sigmas[i] + v * alphas[i]
    #
    #         # If we are not on the last timestep, compute the noisy image for the
    #         # next timestep.
    #         if i < self.num_steps - 1:
    #             # If eta > 0, adjust the scaling factor for the predicted noise
    #             # downward according to the amount of additional noise to add
    #             ddim_sigma = self.eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt() * \
    #                          (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
    #             adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()
    #
    #             # Recombine the predicted noise and predicted denoised image in the
    #             # correct proportions for the next step
    #             z = x * alphas[i + 1] + eps * adjusted_sigma  # ddim eq(12)
    #
    #             # Add the correct amount of fresh noise
    #             if self.eta:
    #                 z += torch.randn_like(z) * ddim_sigma
    #     # If we are on the last timestep, output the denoised image
    #     return x
