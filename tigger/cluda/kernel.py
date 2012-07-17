import os.path
from logging import error

import numpy
from mako.template import Template
from mako import exceptions

from tigger.cluda import dtypes

_PRELUDE = Template(filename=os.path.join(os.path.split(__file__)[0], "prelude.cluda.mako"))
_FUNCTIONS = Template(filename=os.path.join(os.path.split(__file__)[0], "functions.cluda.mako"))


class FuncCollector:

    def __init__(self, prefix=""):
        self.prefix = prefix
        self.functions = {}

    def cast(self, out_dtype, in_dtype):
        out_ctype = dtypes.ctype(out_dtype)
        in_ctype = dtypes.ctype(in_dtype)
        name = "_{prefix}_cast_{out}_{in_}".format(prefix=self.prefix, out=out_ctype, in_=in_ctype)
        self.functions[name] = ('cast', (out_dtype, in_dtype))
        return name

    def mul(self, dtype1, dtype2, out=None):
        if out is None:
            out = numpy.result_type(dtype1, dtype2)
        ctypes = [dtypes.ctype(dt) for dt in (dtype1, dtype2)]
        out_ctype = dtypes.ctype(out)

        name = "_{prefix}_mul__{out}__{signature}".format(
            prefix=self.prefix, out=out_ctype, signature = '_'.join(ctypes))

        self.functions[name] = ('mul', (out, dtype1, dtype2))
        return name

    def render(self):
        return _FUNCTIONS.render(dtypes=dtypes, functions=self.functions)


def render_prelude(env):
    return _PRELUDE.render(api=env.api)

def render_without_funcs(template, func_c, **kwds):
    try:
        src = template.render(func=func_c, **kwds)
    except:
        error("Failed to render template:\n" + exceptions.text_error_template().render())
        raise Exception("Template rendering failed")
    return src

def render_kernel(template, **kwds):
    func_c = FuncCollector()
    src = render_without_funcs(template, func_c, **kwds)
    return func_c.render() + src
