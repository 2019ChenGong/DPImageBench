import os
import logging
import datetime

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models.model_loader import load_model
from data.dataset_loader import load_data


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        raise ValueError('Directory already exists.')

def run(func, config):
    if config.setup.run_type == "normal":
        config.setup.local_rank = 0
        config.setup.global_rank = 0
        config.setup.global_size = config.setup.n_nodes * config.setup.n_gpus_per_node
        config.model.local_rank = config.setup.local_rank
        config.model.global_rank = config.setup.global_rank
        config.model.global_size = config.setup.global_size
        func(config)
    elif config.setup.run_type == "torchmp":
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        processes = []
        for rank in range(config.setup.n_gpus_per_node):
            config.setup.local_rank = rank
            config.setup.global_rank = rank + \
                config.setup.node_rank * config.setup.n_gpus_per_node
            config.setup.global_size = config.setup.n_nodes * config.setup.n_gpus_per_node
            config.model.local_rank = config.setup.local_rank
            config.model.global_rank = config.setup.global_rank
            config.model.global_size = config.setup.global_size
            config.model.fid_stats = config.sensitive_data.fid_stats
            print('Node rank %d, local proc %d, global proc %d' % (
                config.setup.node_rank, config.setup.local_rank, config.setup.global_rank))
            p = mp.Process(target=setup, args=(config, func))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    elif config.setup.run_type == "torchrun":
        dist.init_process_group("nccl")
        config.setup.local_rank = int(os.environ["LOCAL_RANK"])
        config.model.local_rank = config.setup.local_rank
        config.setup.global_rank = int(os.environ["RANK"])
        config.model.global_rank = config.setup.global_rank
        config.setup.global_size = dist.get_world_size()
        config.model.global_size = config.setup.global_size
        config.model.fid_stats = config.sensitive_data.fid_stats
        func(config)
    else:
        NotImplementedError('run_type {} is not yet implemented.'.format(config.setup.run_type))

def setup(config, fn):
    os.environ['MASTER_ADDR'] = config.setup.master_address
    os.environ['MASTER_PORT'] = '%d' % config.setup.master_port
    os.environ['OMP_NUM_THREADS'] = '%d' % config.setup.omp_n_threads
    torch.cuda.set_device(config.setup.local_rank)
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=config.setup.global_rank,
                            world_size=config.setup.global_size)
    fn(config)
    # dist.barrier()
    dist.destroy_process_group()

def set_logger(gfile_stream):
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')


def initialize_environment(config):
    config.setup.root_folder = "."
    config.pretrain.log_dir = config.setup.workdir + "/pretrain"
    config.train.log_dir = config.setup.workdir + "/train"
    config.gen.log_dir = config.setup.workdir + "/gen"
    config.gen.n_classes = config.sensitive_data.n_classes
    if config.setup.global_rank == 0:
        workdir = os.path.join(config.setup.root_folder, config.setup.workdir)
        if os.path.exists(workdir):
            gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'a')
            set_logger(gfile_stream)
        else:
            make_dir(workdir)
            gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'a')
            set_logger(gfile_stream)
            logging.info(config)


def parse_config(opt, unknown):
    if float(opt.epsilon) > 5:
        config_epsilon = 10.0
    else:
        config_epsilon = 1.0
    config_path = os.path.join(opt.config_dir, opt.method, opt.data_name + "_eps" + str(config_epsilon) + opt.config_suffix + ".yaml")
    if not os.path.exists(config_path):
        configs = [OmegaConf.load(os.path.join(opt.config_dir, opt.method, "custom.yaml"))]
    else:
        configs = [OmegaConf.load(config_path)]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    config.train.dp.epsilon = float(opt.epsilon)
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if opt.resume_exp is not None:
        config.setup.workdir = "exp/{}/{}".format(str.lower(opt.method), opt.resume_exp)
    else:
        config.setup.workdir = "exp/{}/{}_eps{}{}{}-{}".format(str.lower(opt.method), opt.data_name, opt.epsilon, opt.config_suffix, opt.exp_description, nowTime)
    if not os.path.exists(config_path):
        config.sensitive_data.data_name = opt.data_name
        config.sensitive_data.train_path = os.path.join("dataset", opt.data_name, "train_32.zip")
        config.sensitive_data.test_path = os.path.join("dataset", opt.data_name, "test_32.zip")
        config.sensitive_data.fid_stats = os.path.join("dataset", opt.data_name, "fid_stats_32.npz")
    if opt.method in ["PE", "PE-SD"]:
        config.train.tmp_folder = config.sensitive_data.name
        config.train.private_num_classes = config.sensitive_data.n_classes
        return config
    config.model.private_num_classes = config.sensitive_data.n_classes
    config.model.public_num_classes = config.public_data.n_classes
    if config.public_data.name is None or opt.method in ['PrivImage', 'DP-FETA']:
        config.model.public_num_classes = config.model.private_num_classes
    return config