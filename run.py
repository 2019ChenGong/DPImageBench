import logging
import datetime
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sys
import argparse
from omegaconf import OmegaConf

from models.model_loader import load_model
from data.dataset_loader import load_data


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        raise ValueError('Directory already exists.')

def mp_main(config):
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
        p = mp.Process(target=setup, args=(config, main))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


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
    dist.barrier()
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
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    config.setup.workdir = config.setup.workdir + '-' + nowTime
    config.pretrain.log_dir = config.setup.workdir + "/pretrain"
    config.train.log_dir = config.setup.workdir + "/train"
    config.gen.log_dir = config.setup.workdir + "/gen"
    if config.setup.global_rank == 0:
        workdir = os.path.join(config.setup.root_folder, config.setup.workdir)
        make_dir(workdir)
        gfile_stream = open(os.path.join(workdir, 'stdout.txt'), 'w')
        set_logger(gfile_stream)
        logging.info(config)


def main(config):
    initialize_environment(config)

    model = load_model(config)

    sensitive_train_loader, sensitive_test_loader, public_train_loader = load_data(config)

    # model.pretrain(public_train_loader, config.pretrain)

    model.train(sensitive_train_loader, config.train)

    syn_data, syn_labels = model.generate(config.gen)

    # evaluation
    


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs="*", default=["configs/PrivImage/mnist.yaml"])
    opt, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in opt.config]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    if config.setup.torchmp:
        mp_main(config)
    else:
        config.setup.local_rank = 0
        config.setup.global_rank = 0
        main(config)
