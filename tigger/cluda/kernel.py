import os.path
from logging import error

import numpy
from mako.template import Template
from mako import exceptions

from tigger.cluda import dtypes

_PRELUDE = Template(filename=os.path.join(os.path.split(__file__)[0], "prelude.mako"))
_FUNCTIONS = Template(filename=os.path.join(os.path.split(__file__)[0], "functions.mako"))


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
        ctypes = [ctype.replace(' ', '_') for ctype in ctypes]
        out_ctype = dtypes.ctype(out).replace(' ', '_')

        name = "_{prefix}_mul__{out}__{signature}".format(
            prefix=self.prefix, out=out_ctype, signature = '_'.join(ctypes))

        self.functions[name] = ('mul', (out, dtype1, dtype2))
        return name

    def div(self, dtype1, dtype2, out=None):
        if out is None:
            out = numpy.result_type(dtype1, dtype2)
        ctypes = [dtypes.ctype(dt) for dt in (dtype1, dtype2)]
        ctypes = [ctype.replace(' ', '_') for ctype in ctypes]
        out_ctype = dtypes.ctype(out).replace(' ', '_')

        name = "_{prefix}_div__{out}__{signature}".format(
            prefix=self.prefix, out=out_ctype, signature = '_'.join(ctypes))

        self.functions[name] = ('div', (out, dtype1, dtype2))
        return name

    def render(self):
        return _FUNCTIONS.render(dtypes=dtypes, functions=self.functions)


def render_prelude(ctx):
    return _PRELUDE.render(api=ctx.api.API_ID)

def render_without_funcs(template, func_c, *args, **kwds):
    # add some "built-ins" to kernel
    render_kwds = dict(dtypes=dtypes, numpy=numpy, func=func_c)
    assert set(render_kwds).isdisjoint(set(kwds))
    render_kwds.update(kwds)

    try:
        src = template.render(*args, **render_kwds)
    except:
        error("Failed to render template:\n" + exceptions.text_error_template().render())
        raise Exception("Template rendering failed")
    return src

def render_template_source(template_src, *args, **kwds):
    return render_template(Template(template_src), *args, **kwds)

def render_template(template, *args, **kwds):
    func_c = FuncCollector()
    src = render_without_funcs(template, func_c, *args, **kwds)
    return func_c.render() + src
