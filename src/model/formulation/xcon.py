import torch
import torch.nn as nn
from torch.nn import functional as F
# from .diffusion import get_alphas_sigmas
# from ..backbone import expand_to_planes, FourierFeatures
# import lpips
from config import cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class Xcon(nn.Module):
    def __init__(self, backbone, data_shape, hidden_size, target_size, class_dropout):
        super().__init__()
        self.data_shape = data_shape
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.class_dropout = class_dropout
        self.timestep_embed = FourierFeatures(1, 16)
        self.class_embed = nn.Embedding(self.target_size + 1, 4)
        self.backbone = backbone
        # self.ema_model = ema_model
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # Move to GPU or CPU
        # self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

    def forward(self, x_0, t, cond, training=True):
        
        if training:
            # noise_reals is the x sample with noise xt, target is clear image X0
            noised_reals, targets, classes_drop = self.forward_diffusion_sample(x_0, t, cond) 
            # input xt, t, predict x0
            predicted_x0 = self.forward_diffusion_pass(noised_reals, t, classes_drop)
            output_target = predicted_x0
            loss = F.mse_loss(predicted_x0, targets)

            # Consistency Training
            t_next =  t / (1 + 0.5 * t) # 
            noised_reals_next, targets, classes_drop = self.forward_diffusion_sample(x_0, t_next, cond) 
            predicted_x0_next = self.forward_diffusion_pass(noised_reals_next, t_next, classes_drop)


            consistency_loss = self.loss_fn_alex(predicted_x0, predicted_x0_next).mean() ##？ mean teacher, mse loss

            # Total loss
            total_loss = loss + cfg['lambda'] * consistency_loss
            return output_target, total_loss
        else:
            predicted_x0 = self.forward_diffusion_pass(x_0, t, cond)
            output_target = predicted_x0
            loss = 0
        return output_target, loss

    def forward_diffusion_pass(self, x_0, t, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x_0.shape)
        class_embed = expand_to_planes(self.class_embed(cond + 1), x_0.shape)
        pred = self.backbone(torch.cat([x_0, class_embed, timestep_embed], dim=1))
        # pred_ema = self.ema_model(torch.cat([x_0, class_embed, timestep_embed], dim=1))
        return pred

    def forward_diffusion_sample(self, x_0, t, classes):
        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        noise = torch.randn_like(x_0)
        noised_reals = x_0 * alphas + noise * sigmas
        targets = x_0  # Update targets to be x_0 instead of noise

        # Drop out the class of the examples
        to_drop = torch.rand(classes.shape, device=classes.device).le(self.class_dropout)
        classes_drop = torch.where(to_drop, -torch.ones_like(classes), classes)
        return noised_reals, targets, classes_drop


def xcon(backbone, cfg):
    data_shape = cfg['data_shape']
    hidden_size = cfg['diffusion']['hidden_size']
    target_size = cfg['target_size']
    class_dropout = cfg['diffusion']['class_dropout']
    model = Xcon(backbone, data_shape, hidden_size, target_size, class_dropout)
    return model
