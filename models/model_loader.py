

def load_model(config, sess=None):
    if config.setup.method == 'DP-MERF':
        from models.dp_merf import DP_MERF
        model = DP_MERF(config.model, config.setup.local_rank)
    elif config.setup.method == 'DP-Kernel':
        from models.dp_kernel import DP_Kernel
        model = DP_Kernel(config.model, config.setup.local_rank)
    elif config.setup.method == 'DPSDA':
        from models.dpsda import DPSDA
        model = DPSDA(config.model, config.setup.local_rank)
    elif config.setup.method == 'dpsgd-diffusion':
        from models.dpsgd_diffusion import DP_Diffusion
        model = DP_Diffusion(config.model, config.setup.local_rank)
    elif config.setup.method == 'G-PATE':
        from models.g_pate import G_PATE
        model = G_PATE(config.model, config.setup.local_rank, sess)
    elif config.setup.method == 'DataLens':
        from models.datalens import DataLens
        model = DataLens(config.model, config.setup.local_rank, sess)
    else:
        raise NotImplementedError('{} is not yet implemented.'.format(config.setup.method))
    
    return model