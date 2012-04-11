import numpy
import os, os.path

product = lambda x: reduce(lambda x1, x2: x1 * x2, x, 1)


class AttrDict(dict):

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def needsUpdate(self, other):
        for key in other:
            assert key in self, "Unknown key: " + key
            if self[key] != other[key]:
                return True

        return False


def loadTemplateFor(filename):
    name, ext = os.path.splitext(filename)
    template = name + ".cu.mako"
    return open(template).read()


class Computation:

    def __init__(self, env, debug=False, enqueue_to=None):
        self._env = env
        self._debug = debug
        self._queue = enqueue_to

        self._basis = AttrDict(**self._get_default_basis())

    def enqueue_to(self, target):
        self._queue = target
        return self

    def prepare(self, **kwds):
        if self._basis.needsUpdate(kwds):
            self._basis.update(kwds)

        self._derived = self._construct_derived()

        return self

    def prepare_for(self, *args, **kwds):
        b = self._construct_basis(*args, **kwds)
        return self.prepare(**b)

    def _construct_basis(self, *args, **kwds):
        return AttrDict()

    def _construct_derived(self):
        return AttrDict()

    def __call__(self, *args, **kwds):
        if self._debug:
            bs = self._construct_basis(*args, **kwds)
            if self._basis.needsUpdate(bs):
                raise Exception("Given arguments require different basis")

        return self._call(*args, **kwds)


def min_blocks(length, block):
    return (length - 1) / block + 1

def log2(n):
    pos = 0
    for pow in [16, 8, 4, 2, 1]:
        if n >= 2 ** pow:
            n /= (2 ** pow)
            pos += pow
    return pos


