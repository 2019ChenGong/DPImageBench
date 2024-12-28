

def load_model(config, sess=None):
    
    if config.setup.method == 'DP-MERF':
        from models.dp_merf import DP_MERF
        model = DP_MERF(config.model, config.setup.local_rank)
    elif config.setup.method == 'DP-NTK':
        from models.dp_ntk import DP_NTK
        model = DP_NTK(config.model, config.setup.local_rank)
    elif config.setup.method == 'DP-Kernel':
        from models.dp_kernel import DP_Kernel
        model = DP_Kernel(config.model, config.setup.local_rank)
    elif config.setup.method == 'GS-WGAN':
        from models.gs_wgan import GS_WGAN
        config.train.pretrain['data_name'] = config.sensitive_data.name
        config.train.pretrain['eval_mode'] = config.eval.mode
        config.train.pretrain['data_path'] = config.sensitive_data.train_path
        config.train.pretrain['n_gpu'] = config.setup.n_gpus_per_node
        model = GS_WGAN(config.model, config.setup.local_rank)
    elif config.setup.method == 'PE':
        from models.pe import PE
        model = PE(config.model, config.setup.local_rank)
    elif config.setup.method == 'dpsgd-diffusion':
        from models.dpsgd_diffusion import DP_Diffusion
        model = DP_Diffusion(config.model, config.setup.local_rank)
    elif config.setup.method == 'dpsgd-ldm':
        from models.dpsgd_ldm_sc import DP_LDM
        model = DP_LDM(config, config.setup.local_rank)
    elif config.setup.method == 'dpsgd-lora':
        from models.dpsgd_lora_sc import DP_LORA
        model = DP_LORA(config, config.setup.local_rank)
    elif config.setup.method == 'dp-promise':
        from models.dp_promise import DP_Promise
        model = DP_Promise(config.model, config.setup.local_rank)
    elif config.setup.method == 'G-PATE':
        from models.g_pate import G_PATE
        model = G_PATE(config.model, config.setup.local_rank, sess)
    elif config.setup.method == 'DataLens':
        from models.datalens import DataLens
        model = DataLens(config.model, config.setup.local_rank, sess)
    elif config.setup.method == 'dpsgd-gan':
        from models.dpsgd_gan import DPGAN
        model = DPGAN(config.model, config.setup.local_rank)
    else:
        raise NotImplementedError('{} is not yet implemented.'.format(config.setup.method))

    return model, config