import argparse

import pytorch_lightning as pl

## Utils to handle newer PyTorch Lightning changes from version 0.6
## ==================================================================================================== ##
import yaml


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try:  # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except:  # Works for version > 0.6.0
            return fn(self)

    return func_wrapper


def get_config(f):
    with open(f, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    return args
