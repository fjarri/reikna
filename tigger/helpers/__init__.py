import collections
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

def template_source_for(filename):
    name, ext = os.path.splitext(filename)
    return open(name + ".mako").read()

def template_from(template_str):
    return Template(template_str)

def template_defs_for_code(code, argnames):
    return (
        "<%def name='code_functions(" + ", ".join(argnames) + ")'>\n" +
        code.pop('functions', "") +
        "\n</%def>" +
        "<%def name='code_kernel(" + ", ".join(argnames) + ")'>\n" +
        code['kernel'] +
        "\n</%def>")

def template_for(filename):
    return template_from(template_source_for(filename))

def min_blocks(length, block):
    return (length - 1) / block + 1

def log2(n):
    pos = 0
    for pow in [16, 8, 4, 2, 1]:
        if n >= 2 ** pow:
            n /= (2 ** pow)
            pos += pow
    return pos

def factors(n, limit=None):
    if limit is None:
        limit = n

    result = []
    for i in range(1, min(limit, int(n ** 0.5) + 1)):
        div, mod = divmod(n, i)
        if mod == 0:
            result.append((i, div))

    if limit > result[-1][0]:
        result = result + [(div, f) for f, div in reversed(result)]
        return [r for r in result if r[0] <= limit]
    else:
        return result

def wrap_in_tuple(x):
    if isinstance(x, collections.Iterable):
        return tuple(x)
    else:
        return (x,)
