import yaml

from . import env, loader


__version__ = '0.4.1'


def load(stream, additional_vars={}):
    data = yaml.load(stream, loader.Loader)
    return env.interpolate(data, additional_vars)


def load_all(stream, additional_vars={}):
    for data in yaml.load_all(stream, loader.Loader):
        yield env.interpolate(data, additional_vars)
