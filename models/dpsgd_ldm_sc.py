import os, sys, argparse
from packaging import version
import logging
import numpy as np
from omegaconf import OmegaConf

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from models.ldm.data.util import VirtualBatchWrapper, DataModuleFromDataset
from models.ldm.util import instantiate_from_config
from models.ldm.gen import generate_batch
from models.ldm.privacy.myopacus import MyDPLightningDataModule
from models.DP_Diffusion.utils.util import make_dir
from models.synthesizer import DPSynther

from models.ldm.callbacks.cuda import CUDACallback                         # noqa: F401
from models.ldm.callbacks.image_logger import ImageLogger                  # noqa: F401
from models.ldm.callbacks.setup import SetupCallback                       # noqa: F401
from models.ldm.data.util import DataModuleFromConfig, WrappedDataset, WrappedDataset_ldm  # noqa: F401

class DP_LDM(DPSynther):
    def __init__(self, config, device):
        super().__init__()
        self.local_rank = config.setup.local_rank
        self.global_rank = config.setup.global_rank
        self.global_size = config.setup.global_size

        self.config = config
        self.device = 'cuda:%d' % self.local_rank

        self.is_pretrain = True
    
    def pretrain(self, public_dataloader, config):
        if public_dataloader is None:
            self.is_pretrain = False
            return
        if self.global_rank == 0:
            make_dir(config.log_dir)
        
        dataset = WrappedDataset_ldm(public_dataloader.dataset)
        
        if self.config.train.model.params.ckpt_path is not None:
            return
        elif self.config.pretrain.unet.model.params.first_stage_config.params.ckpt_path is None:
            ae_logdir = os.path.join(config.log_dir, "autoencoder")
            self.pretrain_autoencoder(dataset, config.autoencoder, ae_logdir)
            config.unet.model.params.first_stage_config.params.ckpt_path = os.path.join(ae_logdir, "checkpoints", "last.ckpt")
        
        config.unet.data.params.batch_size = config.batch_size
        config.unet.lightning.trainer.max_epochs = config.n_epochs
        self.pretrain_unet(dataset, config.unet, config.log_dir)
        self.config.train.model.params.ckpt_path = os.path.join(config.log_dir, "checkpoints", "last.ckpt")

        torch.cuda.empty_cache()
    
    def pretrain_autoencoder(self, dataset, config, logdir):
        data = DataModuleFromDataset(train=dataset, **config.data.params)
        self.running_flow(data, config, logdir)

    def pretrain_unet(self, dataset, config, logdir):
        logdir = os.path.join(logdir, "unet")
        data = DataModuleFromDataset(train=dataset, **config.data.params)
        self.running_flow(data, config, logdir)

    def train(self, sensitive_dataloader, config):
        if sensitive_dataloader is None:
            return
        
        if self.global_rank == 0:
            make_dir(config.log_dir)
        
        config.model.params.ckpt_path = self.config.train.model.params.ckpt_path
        config.lightning.trainer.max_epochs = config.n_epochs
        config.data.params.batch_size = config.batch_size
        config.model.params.dp_config.delta = config.dp.delta
        config.model.params.dp_config.epsilon = config.dp.epsilon
        config.model.params.dp_config.max_grad_norm = config.dp.max_grad_norm
        config.model.params.dp_config.max_batch_size = config.batch_size // config.n_splits

        # data = DataModuleFromDataset(train=WrappedDataset_ldm(sensitive_dataloader.dataset), validation=WrappedDataset_ldm(sensitive_dataloader.dataset), **config.data.params)
        data = DataModuleFromDataset(train=WrappedDataset_ldm(sensitive_dataloader.dataset), **config.data.params)
        
        self.running_flow(data, config, config.log_dir)

    def running_flow(self, data, config, logdir):
        sys.path.append(os.getcwd())

        parser = argparse.ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        opt, unknown = parser.parse_known_args()

        config_parse = {"name": "", "resume": "", "base": "", "train": True, "no-test": False, "project": None, "debug": False, "seed": 0, "postfix": "", "logdir": "logs", "scale_lr": True}
        for k in config_parse:
            setattr(opt, k, config_parse[k])

        ckptdir = os.path.join(logdir, "checkpoints")
        cfgdir = os.path.join(logdir, "configs")
        seed_everything(opt.seed)

        # init and save configs
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*[config], cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        if 'dp' not in config:
            trainer_config["accelerator"] = trainer_config.get("accelerator", "ddp")
        else:
            trainer_config["accelerator"] = "gpu"
        trainer_config["gpus"] = ",".join([str(i) for i in range(self.global_size)]) + ","
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if "gpus" not in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # default_logger_cfgs = {
        #     "testtube": {
        #         "target": "pytorch_lightning.loggers.TestTubeLogger",
        #         "params": {
        #             "name": "testtube",
        #             "save_dir": logdir,
        #         }
        #     },
        # }
        # default_logger_cfg = default_logger_cfgs["testtube"]
        # logger_cfg = lightning_config.get("logger", OmegaConf.create())
        # logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        # trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        modelckpt_cfg = lightning_config.get("modelcheckpoint", OmegaConf.create())
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "models.dpsgd_ldm.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": "",
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "models.dpsgd_ldm.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "models.dpsgd_ldm.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        callbacks_cfg = lightning_config.get("callbacks", OmegaConf.create())

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print('Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint': {
                    "target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                    'params': {
                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        'save_top_k': -1,
                        'every_n_train_steps': 10000,
                        'save_weights_only': True
                    }
                }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        # Build the trainer
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir

        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"  {k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
        dp_config = config.model.params.get("dp_config")
        if dp_config and dp_config.enabled and dp_config.poisson_sampling:
            print("Using Poisson sampling")
            data = MyDPLightningDataModule(data)
            if dp_config.get("max_batch_size", None):
                print("Using virtual batch size of", dp_config.max_batch_size)
                data = VirtualBatchWrapper(data, dp_config.max_batch_size)

        # Configure learning rate
        print("#### Learning Rate ####")
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        gpus = lightning_config.trainer.gpus.strip(",").split(',')
        ngpu = len(gpus) if not cpu else 1
        accumulate_grad_batches = lightning_config.trainer.get("accumulate_grad_batches", 1)
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(f"Setting learning rate to {model.learning_rate:.2e} ",
                f"= {accumulate_grad_batches} (accumulate_grad_batches) ",
                f"* {ngpu} (num_gpus) ",
                f"* {bs} (batchsize) ",
                f"* {base_lr:.2e} (base_lr)")
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "on_signal.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb
                pudb.set_trace()

        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                trainer.save_checkpoint(os.path.join(ckptdir, "on_exception.ckpt"))
                raise
        # if not opt.no_test and not trainer.interrupted:
        #     trainer.test(model, data)


    def generate(self, config):
        logging.info("start to generate {} samples".format(config.data_num))
        if self.global_rank == 0 and not os.path.exists(config.log_dir):
            make_dir(config.log_dir)
        
        model_path = os.path.join(self.config.train.log_dir, "checkpoints", "last.ckpt")
        syn_data, syn_labels = generate_batch(self.config.train, config.data_num, model_path, num_classes=self.config.sensitive_data.n_classes, batch_size=config.batch_size)

        logging.info("Generation Finished!")

        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        show_images = []
        for cls in range(self.config.sensitive_data.n_classes):
            show_images.append(syn_data[syn_labels==cls][:8])
        show_images = np.concatenate(show_images)
        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
        return syn_data, syn_labels


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))