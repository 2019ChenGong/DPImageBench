import os
import logging
import datetime

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models.model_loader import load_model
from data.dataset_loader import load_data
# from evaluation.evaluator import Evaluator


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        raise ValueError('Directory already exists.')

def run(func, config):
    if config.setup.run_type == "normal":
        config.setup.local_rank = 0
        config.setup.global_rank = 0
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

    # elif config.setup.run_type == "tfmp":
    #     import tensorflow as tf
    #     config.setup.local_rank = 0
    #     config.setup.global_rank = 0
    #     global FLAGS
    #     FLAGS = config
    #     tf.app.run(main=main_tf)
    else:
        NotImplementedError('run_type {} is not yet implemented.'.format(config.setup.run_type))


# def main_tf(_):
#     import tensorflow as tf
#     global FLAGS
#     run_config = tf.ConfigProto()
#     run_config.gpu_options.allow_growth = True
#     with tf.Session(config=run_config) as sess:
#         initialize_environment(FLAGS)

#         model = load_model(FLAGS, sess)

#         sensitive_train_loader, sensitive_test_loader, public_train_loader = load_data(FLAGS)

#         model.pretrain(public_train_loader, FLAGS.pretrain)

#         model.train(sensitive_train_loader, FLAGS.train)

#         syn_data, syn_labels = model.generate(FLAGS.gen)

#     evaluator = Evaluator(FLAGS)
#     evaluator.eval(syn_data, syn_labels, sensitive_test_loader)

def setup(config, fn):
    os.environ['MASTER_ADDR'] = config.setup.master_address
    import socket
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('127.0.0.1', port))
            return result == 0
    port = config.setup.master_port
    while is_port_in_use(port):
        port += 1
    config.setup.master_port = port
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
    config_path = os.path.join(opt.config_dir, opt.method, opt.data_name + "_eps" + str(opt.epsilon) + opt.config_suffix + ".yaml")
    configs = [OmegaConf.load(config_path)]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    return config