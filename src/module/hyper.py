from config import cfg


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['model_mode'] = cfg['control']['model_mode']
    cfg['formulation_mode'] = cfg['control']['formulation_mode']

    cfg['batch_size'] = 256
    cfg['step_period'] = 1
    cfg['num_steps'] = None
    cfg['eval_period'] = 200
    cfg['num_epochs'] = 10
    cfg['collate_mode'] = 'dict'
    # cfg['gradient_scaler'] = False

    cfg['model'] = {}
    cfg['model']['model_name'] = cfg['model_name']
    cfg['model']['model_mode'] = cfg['model_mode']
    cfg['model']['formulation_mode'] = cfg['formulation_mode']
    cfg['model']['unet'] = {'hidden_size': 64}
    cfg['model']['diffusion'] = {'class_dropout': 0.2}
    cfg['model']['flow'] = {'class_dropout': 0.2}

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
    # cfg[tag]['optimizer']['scheduler_name'] = 'LinearAnnealingLR'
    cfg[tag]['optimizer']['scheduler_name'] = 'None'
    cfg[tag]['optimizer']['warmup_ratio'] = 0

    cfg['generate']['batch_size'] = 5
    cfg['generate']['img_fmt'] = 'png'
    cfg['generate']['normalize'] = False
    return
