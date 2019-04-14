# pylint: disable=undefined-variable
import os
import re
from munch import munchify
import yaml

try:
    from collections.abc import Mapping, Sequence, Set
except ImportError:
    # Python 2.7
    from collections import Mapping, Sequence, Set  # noqa

import six


def objwalk(obj, path=(), memo=None):
    if memo is None:
        memo = set()
    iterator = None
    if isinstance(obj, Mapping):
        iterator = six.iteritems
    elif isinstance(
            obj, (Sequence, Set)
    ) and not isinstance(obj, six.string_types):
        iterator = enumerate
    if iterator:
        if id(obj) not in memo:
            memo.add(id(obj))
            for path_component, value in iterator(obj):
                for result in objwalk(value, path + (path_component,), memo):
                    yield result
            memo.remove(id(obj))
    else:
        yield path, obj


class EnvVar(object):
    __slots__ = ['name', 'default', 'string', 'yaml_data', 'additional_vars']

    RE = re.compile(
        r'\$\{(?P<name>[^:-]+)(?:(?P<separator>:?-)(?P<default>.+))?\}')

    def __init__(self, name, default, string, yaml_data, additional_vars):
        self.name = name
        self.default = default
        self.string = string
        self.yaml_data = yaml_data
        self.additional_vars = additional_vars
        print ('==========',additional_vars)
    @property
    def value(self):

        # Try from environ
        value = os.environ.get(self.name)
        if not value and self.name in self.yaml_data:
            value = str(eval(f'self.yaml_data.{self.name}'))
        
        if not value: 
            value = self.additional_vars.get(self.name, None)
            
        # Recursion . WARNING: Can be infinite
        if value:                
            res2 = EnvVar.from_string(value, self.yaml_data, self.additional_vars)
            if res2 is not None: 
                value = str(res2.value)
        
        if not value and self.default: 
            value = self.default
        
        if value: 
            return self.RE.sub(value, self.string)
        else:
            raise ValueError('Missing value and default for {}'.format(self.name))
        
        
    @classmethod
    def from_string(cls, s, yaml_data, additional_vars):
        if not isinstance(s, six.string_types):
            return None
        data = cls.RE.search(s)
        if not data:
            return None
        data = data.groupdict()
        return cls(data['name'], data['default'], s, yaml_data, additional_vars)


def interpolate(data, additional_vars):
    datam = munchify(data)
    for path, obj in objwalk(data):
        # print(' - ', obj, path)
        e = EnvVar.from_string(obj, datam, additional_vars)
        if e is not None:
            x = data
            for k in path[:-1]:
                x = x[k]
            x[path[-1]] = e.value
    return data
