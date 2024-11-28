import torch
from torch.nn import functional as F
import math
from tqdm.notebook import trange
from model import get_alphas_sigmas
from config import cfg


@torch.no_grad()
def ddim_sample_loop_Xzero(model, x, steps, eta, classes, guidance_scale=1.):
    """Draws samples from a model given starting noise for the X_zero objective."""
    ts = x.new_ones([x.shape[0]])

    # Define timesteps and compute alphas and sigmas based on the schedule
    t = torch.linspace(1, 0, steps + 1)[:-1].to(x.device)
    alphas, sigmas = get_alphas_sigmas(t)

    input = {}

    # The sampling loop
    for i in trange(steps):
        with torch.cuda.amp.autocast():
            x_in = torch.cat([x, x])  # Duplicate input for unconditional and conditional
            ts_in = torch.cat([ts, ts])
            classes_in = torch.cat([-torch.ones_like(classes), classes])  # Classifier-free guidance
            input['data'] = x_in
            input['target'] = classes_in
            input['t'] = ts_in * t[i]

            # Model directly outputs the predicted x_0
            x0_uncond, x0_cond = model(input)['data'].float().chunk(2)

        # Apply classifier-free guidance on x_0 prediction
        pred_x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)

        # If not on the last timestep, calculate the noisy image for the next timestep
        if i < steps - 1:
            ddim_sigma = eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt() * \
                         (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
            adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()

            # Recombine the predicted x_0 and the noise for the next step
            x = pred_x0 * alphas[i + 1] + (x - pred_x0 * alphas[i]) / sigmas[i] * adjusted_sigma

            # Add noise if eta > 0
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If on the last timestep, return the denoised image
    return pred_x0


@torch.no_grad()
def ddim_sample_loop_Epsilon(model, x, steps, eta, classes, guidance_scale=1.):
    """Draws samples from a model given starting noise for the Epsilon objective."""
    ts = x.new_ones([x.shape[0]])

    # Define timesteps and compute alphas and sigmas based on the schedule
    t = torch.linspace(1, 0, steps + 1)[:-1].to(x.device)
    alphas, sigmas = get_alphas_sigmas(t)

    input = {}

    # The sampling loop
    for i in trange(steps):
        with torch.cuda.amp.autocast():
            x_in = torch.cat([x, x])  # Duplicate input for unconditional and conditional
            ts_in = torch.cat([ts, ts])
            classes_in = torch.cat([-torch.ones_like(classes), classes])  # Classifier-free guidance
            input['data'] = x_in
            input['target'] = classes_in
            input['t'] = ts_in * t[i]

            # Model outputs the predicted noise for each input
            eps_uncond, eps_cond = model(input)['data'].float().chunk(2)

        # Apply classifier-free guidance on noise prediction
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # Predict the denoised image (epsilon objective)
        pred_x0 = (x - sigmas[i] * eps) / alphas[i]

        # If not on the last timestep, calculate the noisy image for the next timestep
        if i < steps - 1:
            ddim_sigma = eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt() * \
                         (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
            adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()

            # Recombine the denoised image and the noise for the next step
            x = pred_x0 * alphas[i + 1] + eps * adjusted_sigma

            # Add noise if eta > 0
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If on the last timestep, return the denoised image
    return pred_x0


@torch.no_grad()
def ddim_sample_loop_V(model, x, steps, eta, classes, guidance_scale=1.):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    t = torch.linspace(1, 0, steps + 1)[:-1].to(cfg['device'])
    alphas, sigmas = get_alphas_sigmas(t)

    input = {}

    # The sampling loop
    for i in trange(steps):
        with torch.cuda.amp.autocast():
            x_in = torch.cat([x, x])
            ts_in = torch.cat([ts, ts])
            classes_in = torch.cat([-torch.ones_like(classes), classes])
            input['data'] = x_in
            input['target'] = classes_in
            input['t'] = ts_in * t[i]
            v_uncond, v_cond = model(input)['data'].float().chunk(2)
        v = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt() * \
                         (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
            adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma  # ddim eq(12)

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma
    # If we are on the last timestep, output the denoised image
    return pred
