import torch
from torch.nn import functional as F
import math
from tqdm.notebook import trange
from model import get_alphas_sigmas, get_index_from_list
from config import cfg

# Global buffers
buffers = {}

# Function to initialize global buffers
def initialize_global_buffers(num_timesteps=100):
    global buffers
    betas = torch.linspace(0.0001, 0.02, num_timesteps, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

    buffers = {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "posterior_variance": betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        "posterior_log_variance_clipped": torch.log(
            (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)).clamp(min=1e-20)
        ),
        "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod),
        "sqrt_recipm_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod - 1),
        "posterior_mean_coef1": betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        "posterior_mean_coef2": (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas_cumprod) / (1.0 - alphas_cumprod),
    }

initialize_global_buffers(num_timesteps=100)

def extract(a, t, x_shape):
    # retreive the data from the buffer according to the timestep and reshape to the shape wanted
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
def map_timestep_for_batch(t_i, num_timesteps, batch_size):
    # Map t_i (continuous) to a discrete timestep index
    discrete_index = torch.round(t_i * (num_timesteps - 1)).long()
    
    # Expand to match the batch size
    discrete_index_batch = torch.full((batch_size,), discrete_index, dtype=torch.long, device=t_i.device)
    return discrete_index_batch

# Function to compute q_posterior
def q_posterior(x_start, x, t, buffers):
    posterior_mean = (
        extract(buffers["posterior_mean_coef1"], t, x.shape) * x_start
        + extract(buffers["posterior_mean_coef2"], t, x.shape) * x
    )
    posterior_variance = extract(buffers["posterior_variance"], t, x.shape)
    posterior_log_variance = extract(buffers["posterior_log_variance_clipped"], t, x.shape)
    return posterior_mean, posterior_variance, posterior_log_variance

# Function to predict x_0 from noise
def predict_start_from_noise(x, t, pred_noise, buffers):
    return (
        extract(buffers["sqrt_recip_alphas_cumprod"], t, x.shape) * x
        - extract(buffers["sqrt_recipm_alphas_cumprod"], t, x.shape) * pred_noise
    )

# Function to compute p_mean_variance
def p_mean_variance(x, t, cond, core_model, buffers, guidance_scale):
    model_input = {
        'data': x,
        'target': cond,
        't': t,
        'training': False  # Set to False for inference
    }

    # Predict the noise from the core model (assumed MLP or similar)
    model_output = core_model(model_input)
    pred_noise = model_output['data']

    # Apply guidance scale if needed (e.g., classifier-free guidance)
    if guidance_scale != 1.0:
        pred_noise *= guidance_scale

    # Reconstruct x_0 from predicted noise
    x_recon = predict_start_from_noise(x, t, pred_noise, buffers)
    x_recon.clamp_(-1, 1)  # Clamp values to [-1, 1]

    # Get posterior mean and variance
    model_mean, posterior_variance, posterior_log_variance = q_posterior(x_recon, x, t, buffers)
    return model_mean, posterior_log_variance

# Function to perform a single sampling step
def p_sample(x, t, cond, core_model, buffers, eta=1.0, guidance_scale=1.0):
    model_mean, model_log_variance = p_mean_variance(x, t, cond, core_model, buffers, guidance_scale)
    
    # Sample noise
    noise = torch.randn_like(x)
    
    # Add noise if eta > 0, otherwise it's deterministic (eta=0 for deterministic DDIM)
    if eta > 0:
        x = model_mean + (0.5 * model_log_variance).exp() * noise
    else:
        x = model_mean

    return x

# DDIM Sampling Function for Epsilon model
@torch.no_grad()
def ddim_sample_loop_Epsilon(model, x, steps, eta, classes, guidance_scale=1.0):
    global buffers

    device = x.device
    batch_size = x.shape[0]  # Get the batch size from x

    # Ensure buffers are on the correct device
    for key in buffers:
        buffers[key] = buffers[key].to(device)

    # Generate timesteps for sampling loop (discrete mapping)
    t = torch.linspace(1, 0, steps, device=device)

    #for i in reversed(range(len(buffers["betas"]))):
    for i in trange(steps):
        t_i = t[i]
        discrete_t_i = map_timestep_for_batch(t_i, len(buffers["betas"]), batch_size)
        x = p_sample(x, discrete_t_i, classes, model, buffers, eta, guidance_scale)
    
    return x


@torch.no_grad()
def ddim_sample_loop_V(model, x, num_steps, eta, classes, guidance_scale=1.):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    t = torch.linspace(1, 0, num_steps + 1)[:-1].to(cfg['device'])
    alphas, sigmas = get_alphas_sigmas(t)

    input = {}
    pred = None
    # The sampling loop
    for i in tqdm(range(num_steps)):
        with torch.amp.autocast(cfg['device']):
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
        if i < num_steps - 1:
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
def ddim_sample_loop_Xzero():
    return


@torch.no_grad()
def ddim_sample_loop_Xprev():
    return
