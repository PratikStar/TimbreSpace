import argparse
import json
import re
import pytorch_lightning as pl
import inspect
from prodict import Prodict
## Utils to handle newer PyTorch Lightning changes from version 0.6
## ==================================================================================================== ##
import yaml
from functools import reduce

def re_nest_configs(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict

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


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(*args):
        # print(f"getattr {args[1]}")
        if args[1] in ("__getstate__", "__setstate__"):
            # print(f"type {type(args[0])}")
            return args[0]

        val = dict.get(*args)
        return dotdict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __call__ = dict.__call__

def merge(a, b, path=None): # merge nested dict
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                a[key] = b[key] # override
        else:
            a[key] = b[key]
    return a

def get_config(f):

    with open(f, 'r') as file:
        try:
            config = Prodict.from_dict(yaml.safe_load(file))
        except yaml.YAMLError as exc:
            print(exc)
            return None
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae.yaml')

    args, unknown = parser.parse_known_args()

    d = {}
    for param in unknown:
        k, v = param.split('=')
        if v[0] == "[":  # list
            v = json.loads(v)
        elif len(re.findall('[0-9]*\.[0-9]*|[0-9]*', v)) > 0:  # number
            matches = re.findall('[0-9]*\.[0-9]*|[0-9]*', v)
            for match in matches:
                if match != '':
                    v = match
        elif v == "True":
            v = True
        elif v == "False":
            v = False
        else:
            print(f"Invalid value {v} for key {k}. Ignoring")
            continue

        ksub = k.split('.')
        ksub.reverse()
        t = {ksub[0]: v}
        for i in ksub[1:-1]:
            temp = {i: t}
            t = temp
        d = merge(d, t)
    return args, d
