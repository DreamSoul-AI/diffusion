from config import cfg


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['model_mode'] = cfg['control']['model_mode']
    cfg['formulation_mode'] = cfg['control']['formulation_mode']
    cfg['class_cond'] = cfg['control']['class_cond']

    cfg['batch_size'] = 256
    cfg['step_period'] = 1
    cfg['num_steps'] = None
    cfg['eval_period'] = 100
    if cfg['data_name'] in ['TwoMoons']:
        cfg['num_epochs'] = 1000
    else:
        cfg['num_epochs'] = 10
    cfg['collate_mode'] = 'dict'
    cfg['save_checkpoint'] = True

    cfg['model'] = {}
    cfg['model']['model_name'] = cfg['model_name']
    cfg['model']['model_mode'] = cfg['model_mode']
    cfg['model']['formulation_mode'] = cfg['formulation_mode']
    cfg['model']['unet'] = {'hidden_size': 64}
    if cfg['data_name'] in ['TwoMoons']:
        cfg['model']['mlp'] = {'hidden_size': [64, 64, 64], 'activation': 'relu'}
    else:
        cfg['model']['mlp'] = {'hidden_size': [128, 256], 'activation': 'relu'}

    # TODO: improve time embedding
    # https://github.com/facebookresearch/flow_matching/tree/main/examples/image
    # https://github.com/facebookresearch/flow_matching/blob/47c439602a360ef7d2ec2b9a521b4dda5f335112/examples/image/models/nn.py#L107
    # cfg['model']['timestep_embedding_mode'] = 'identity'
    cfg['model']['timestep_embedding_mode'] = 'fourier'
    # cfg['model']['timestep_embedding_size'] = 1
    cfg['model']['timestep_embedding_size'] = 16
    # cfg['model']['timestep_embedding_size'] = 0
    if cfg['class_cond'] > 0:
        class_dropout = 0.2
        cfg['model']['cond_embedding_size'] = 1
    else:
        class_dropout = 0
        cfg['model']['cond_embedding_size'] = 0

    cfg['model']['diffusion'] = {'class_dropout': class_dropout, 'regularization': {'v': 1, 'x0': 0.1, 'eps': 0.1}}
    # cfg['model']['flow'] = {'class_dropout': class_dropout, 'sig_min': 1e-3}
    cfg['model']['flow'] = {'class_dropout': class_dropout}


    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'AdamW'
    cfg[tag]['optimizer']['lr'] = 1e-3
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 1e-4
    cfg[tag]['optimizer']['nesterov'] = True
    cfg[tag]['optimizer']['test_batch_ratio'] = 4
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'],
                                           'test': cfg[tag]['optimizer']['test_batch_ratio'] * cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = 'CosineAnnealingLR'
    # cfg[tag]['optimizer']['scheduler_name'] = 'None'
    cfg[tag]['optimizer']['min_lr'] = 1e-5
    cfg[tag]['optimizer']['warmup_ratio'] = 0

    cfg['generate']['model_mode'] = cfg['model_mode']
    if cfg['data_name'] in ['TwoMoons']:
        cfg['generate']['batch_size'] = 150
    else:
        cfg['generate']['batch_size'] = 5
    cfg['generate']['img_fmt'] = 'png'
    cfg['generate']['normalize'] = False
    return
