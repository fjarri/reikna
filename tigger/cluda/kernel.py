from tigger.cluda import dtypes
import os.path
from mako.template import Template
from mako import exceptions

_PRELUDE = Template(filename=os.path.join(os.path.split(__file__)[0], 'prelude.cluda.mako'))
_MUL = Template(filename=os.path.join(os.path.split(__file__)[0], 'mul.cluda.mako'))


class MulCollector:

    def __init__(self):
        self.functions = {}

    def __call__(self, dtype1, dtype2, out_dtype=None):
        if out_dtype is None:
            out_dtype = numpy.result_type(dtype1, dtype2)
        ctypes = [dtypes.ctype(dt) for dt in (dtype1, dtype2)]
        out_ctype = dtypes.ctype(out_dtype)

        name = '_mul_' + '_'.join(ctypes) + '__' + out_ctype

        self.functions[name] = (dtype1, dtype2, out_dtype)
        return name


def render_prelude(env):
    return _PRELUDE.render(api=env.params.api)

def render_kernel(env, template, **kwds):
    mul_c = MulCollector()

    try:
        src = template.render(mul=mul_c, **kwds)
    except:
        # TODO: output to stderr?
        print exceptions.text_error_template().render()
        raise Exception("Template rendering failed")

    muls = _MUL.render(dtypes=dtypes, mul_functions=mul_c.functions)
    return muls + src
