from config import cfg


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['formulation_mode'] = cfg['control']['formulation_mode']

    cfg['batch_size'] = 100
    cfg['step_period'] = 1
    cfg['num_steps'] = None
    cfg['eval_period'] = 200
    cfg['num_epochs'] = 10
    cfg['collate_mode'] = 'dict'
    cfg['gradient_scaler'] = False

    cfg['model'] = {}
    cfg['model']['model_name'] = cfg['model_name']
    cfg['model']['formulation_mode'] = cfg['formulation_mode']
    cfg['model']['linear'] = {}
    cfg['model']['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
    cfg['model']['cnn'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet10'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['model']['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}
    cfg['model']['diffusion'] = {'hidden_size': 64, 'class_dropout': 0.2}

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'AdamW'
    cfg[tag]['optimizer']['lr'] = 1e-3
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 1e-4
    cfg[tag]['optimizer']['nesterov'] = True
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': 5 * cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = 'LinearAnnealingLR'
    cfg[tag]['optimizer']['warmup_ratio'] = 0

    # Overwrite if needed
    # cfg['generate']['num_steps'] = 100
    # cfg['generate']['guidance_scale'] = 2.
    # The amount of noise to add each timestep when sampling
    # controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM)
    # 0 = no noise (DDIM)
    # 1 = full noise (DDPM)
    # cfg['generate']['eta'] = 0.
    cfg['generate']['batch_size'] = 5
    cfg['generate']['img_fmt'] = 'png'
    return
