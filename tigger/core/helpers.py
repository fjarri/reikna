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
    return Template(filename=name + ".cluda.mako")

def min_blocks(length, block):
    return (length - 1) / block + 1

def log2(n):
    pos = 0
    for pow in [16, 8, 4, 2, 1]:
        if n >= 2 ** pow:
            n /= (2 ** pow)
            pos += pow
    return pos
