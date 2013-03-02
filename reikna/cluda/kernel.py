import os.path
from logging import error

import numpy
from mako.template import Template
from mako import exceptions

from reikna.cluda import dtypes
from reikna.helpers import template_for

TEMPLATE = template_for(__file__)


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

    def conj(self, dtype):
        ctype = dtypes.ctype(dtype).replace(' ', '_')

        name = "_{prefix}_div__{ctype}".format(
            prefix=self.prefix, ctype=ctype)

        self.functions[name] = ('conj', (dtype,))
        return name

    def exp(self, dtype):
        ctype = dtypes.ctype(dtype).replace(' ', '_')

        name = "_{prefix}_div__{ctype}".format(
            prefix=self.prefix, ctype=ctype)

        self.functions[name] = ('exp', (dtype,))
        return name

    def polar(self, dtype):
        ctype = dtypes.ctype(dtype).replace(' ', '_')

        name = "_{prefix}_div__{ctype}".format(
            prefix=self.prefix, ctype=ctype)

        self.functions[name] = ('polar', (dtype,))
        return name

    def render(self):
        src = []
        for func_name, params in self.functions.items():
            tmpl_name, args = params
            src.append(TEMPLATE.get_def(tmpl_name).render(func_name, *args, dtypes=dtypes))
        return "\n".join(src)


def render_prelude(ctx):
    return TEMPLATE.get_def('prelude').render(api=ctx.api.API_ID, ctx_fast_math=ctx._fast_math)

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
