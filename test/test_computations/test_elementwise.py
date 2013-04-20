import numpy
import pytest

from helpers import *
from reikna.helpers import template_func
from reikna.cluda import Module
from reikna.elementwise import Elementwise
import reikna.cluda.dtypes as dtypes


def test_errors(ctx):

    argnames = (('output',), ('input',), ('param',))
    elw = Elementwise(ctx).set_argnames(*argnames)

    code = lambda output, input, param: Module(
        template_func(
            ['output', 'input', 'param'],
            """
            ${input.ctype} a1 = ${input.load}(idx);
            ${input.ctype} a2 = ${input.load}(idx + ${size});
            ${output.store}(idx, a1 + a2 + ${param});
            """),
        render_kwds=dict(size=output.size),
        snippet=True)

    N = 1000
    a = get_test_array(N * 2, numpy.float32)
    a_dev = ctx.to_device(a)
    b_dev = ctx.array(N, numpy.float32)
    param = 1

    elw.prepare_for(b_dev, a_dev, numpy.float32(param), code=code)
    elw(b_dev, a_dev, param)
    assert diff_is_negligible(ctx.from_device(b_dev), a[:N] + a[N:] + param)


def test_nontrivial_code(ctx):

    argnames = (('output',), ('input',), ('param',))
    elw = Elementwise(ctx).set_argnames(*argnames)

    function = lambda output, input, param: Module(
        template_func(
            ['prefix'],
            """
            WITHIN_KERNEL ${otype} ${prefix}(${itype} val, ${ptype} param)
            {
                return val + param;
            }
            """),
        render_kwds=dict(
            otype=dtypes.ctype(output.dtype),
            itype=dtypes.ctype(input.dtype),
            ptype=dtypes.ctype(param.dtype)))

    code = lambda output, input, param: Module(
        template_func(
            ['output', 'input', 'param'],
            """
            ${input.ctype} a1 = ${input.load}(idx);
            ${input.ctype} a2 = ${input.load}(idx + ${size});
            ${output.store}(idx, a1 + ${func}(a2, ${param}));
            """),
        render_kwds=dict(func=function(output, input, param), size=output.size),
        snippet=True)

    N = 1000
    a = get_test_array(N * 2, numpy.float32)
    a_dev = ctx.to_device(a)
    b_dev = ctx.array(N, numpy.float32)
    param = 1

    elw.prepare_for(b_dev, a_dev, numpy.float32(param), code=code)
    elw(b_dev, a_dev, param)
    assert diff_is_negligible(ctx.from_device(b_dev), a[:N] + a[N:] + param)
