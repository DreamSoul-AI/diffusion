import torch
from torch.nn import functional as F
from tqdm.notebook import tqdm, trange
from model import get_alphas_sigmas, get_index_from_list
from config import cfg


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


@torch.no_grad()
def ddim_sample_loop_Epsilon(model, x, t, steps, eta, classes, guidance_scale=1.):
    alphas, sigmas = get_alphas_sigmas(cfg['steps'])  # sigma: noise level

    # Pre-calculate different terms for closed form
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = sigmas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    sigma_t = get_index_from_list(sigmas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # The sampling loop
    for i in trange(steps):
        # Call model (current image - noise prediction)
        pred = sqrt_recip_alphas_t * (
                x - sigma_t * model(x, t)['target'] / sqrt_one_minus_alphas_cumprod_t
        )  # Epsilon model output is the predicted noise
        posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
        eps = model(x, t)['target']

        if i < steps - 1:
            ddim_sigma = eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt() * \
                         (1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
            adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()
            x = pred * alphas[i + 1] + eps * adjusted_sigma  # ddim eq(12)

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    return pred


@torch.no_grad()
def ddim_sample_loop_Xzero():
    return


@torch.no_grad()
def ddim_sample_loop_Xprev():
    return
