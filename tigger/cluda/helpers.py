import os.path
from mako.template import Template

product = lambda x: reduce(lambda x1, x2: x1 * x2, x, 1)


class AttrDict(dict):

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def __repr__(self):
        return "AttrDict(" + dict.__repr__(self) + ")"


def template_for(filename):
    name, ext = os.path.splitext(filename)
    return Template(filename=name + ".mako")

def template_from(template_str):
    return Template(template_str)

def min_blocks(length, block):
    return (length - 1) / block + 1

def log2(n):
    pos = 0
    for pow in [16, 8, 4, 2, 1]:
        if n >= 2 ** pow:
            n /= (2 ** pow)
            pos += pow
    return pos

def factors(n):
    result = []
    for i in range(1, int(n ** 0.5) + 1):
        div, mod = divmod(n, i)
        if mod == 0:
            result.append((i, div))
    return result + [(div, f) for f, div in reversed(result)]
