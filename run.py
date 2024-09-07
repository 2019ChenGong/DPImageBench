import os
import sys
import argparse

from models.model_loader import load_model
from data.dataset_loader import load_data
from utils.utils import initialize_environment, run, parse_config


def main(config):
    initialize_environment(config)

    model = load_model(config)

    sensitive_train_loader, sensitive_test_loader, public_train_loader = load_data(config)

    model.pretrain(public_train_loader, config.pretrain)

    model.train(sensitive_train_loader, config.train)

    syn_data, syn_labels = model.generate(config.gen)

    # evaluation
    


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs="*", default=["configs/DPGAN/eps1.0/fmnist.yaml"])
    opt, unknown = parser.parse_known_args()

    config = parse_config(opt, unknown)

    run(main, config)
