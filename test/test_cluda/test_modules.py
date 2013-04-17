import pytest

from helpers import *
from reikna.helpers import *
from reikna.cluda.kernel import Module


TEMPLATE = template_from("""
<%def name="multiplier(prefix)">
<%
    ctype = dtypes.ctype(dtype)
%>
WITHIN_KERNEL ${ctype} ${prefix}multiplier(${ctype} x, ${ctype} y)
{
    return ${func.mul(dtype, dtype, out=dtype)}(x, y) + ${num};
}
</%def>


<%def name="subtractor(prefix)">
<%
    ctype = dtypes.ctype(dtype)
%>
WITHIN_KERNEL ${ctype} ${prefix}subtractor(${ctype} x, ${ctype} y)
{
    return ${m}multiplier(x, y) - ${num};
}
</%def>


<%def name="combinator(prefix)">
<%
    ctype = dtypes.ctype(dtype)
%>
WITHIN_KERNEL ${ctype} ${prefix}combinator(${ctype} x, ${ctype} y)
{
    return ${m}multiplier(x, y) + ${s}subtractor(x, y);
}
</%def>

<%def name="snippet(a, b)">
dest[idx] = ${c}combinator(${a}[idx], ${b}[idx]);
</%def>
""")


class Multiplier(Module):
    def __init__(self, dtype, num=1):
        Module.__init__(
            self,
            TEMPLATE.get_def('multiplier'),
            render_kwds=dict(dtype=dtype, num=num))

class Subtractor(Module):
    def __init__(self, dtype, mnum=1, num=1):
        m = Multiplier(dtype, num=mnum)
        Module.__init__(
            self,
            TEMPLATE.get_def('subtractor'),
            render_kwds=dict(dtype=dtype, num=num, m=m))

class Combinator(Module):
    def __init__(self, dtype, m1num=1, m2num=1, snum=1):
        m = Multiplier(dtype, num=m1num)
        s = Subtractor(dtype, mnum=m2num, num=snum)
        Module.__init__(
            self,
            TEMPLATE.get_def('combinator'),
            render_kwds=dict(dtype=dtype, m=m, s=s))

class CombinatorCall(Module):
    def __init__(self, dtype, m1num=1, m2num=1, snum=1):
        c = Combinator(dtype, m1num=m1num, m2num=m2num, snum=snum)
        Module.__init__(
            self,
            TEMPLATE.get_def('snippet'),
            render_kwds=dict(c=c),
            snippet=True)


def test_modules(some_ctx):

    dtype = numpy.float32
    m1num = 2
    m2num = 3
    snum = 10
    N = 128

    module = some_ctx.compile(
        """
        KERNEL void test(GLOBAL_MEM float *dest, GLOBAL_MEM float *a, GLOBAL_MEM float *b)
        {
            const int idx = get_global_id(0);
            dest[idx] = ${c}combinator(a[idx], b[idx]);
        }
        """,
        render_kwds=dict(c=Combinator(dtype, m1num=m1num, m2num=m2num, snum=snum)))

    a = get_test_array(N, dtype)
    b = get_test_array(N, dtype)
    a_dev = some_ctx.to_device(a)
    b_dev = some_ctx.to_device(b)
    dest_dev = some_ctx.empty_like(a_dev)

    module.test(dest_dev, a_dev, b_dev, local_size=N, global_size=N)
    ref = (a * b + m1num) + ((a * b + m2num) - snum)

    assert diff_is_negligible(dest_dev.get(), ref)


def test_snippet(some_ctx):

    dtype = numpy.float32
    m1num = 2
    m2num = 3
    snum = 10
    N = 128

    module = some_ctx.compile(
        """
        KERNEL void test(GLOBAL_MEM float *dest, GLOBAL_MEM float *a, GLOBAL_MEM float *b)
        {
            const int idx = get_global_id(0);
            ${s('a', 'b')}
        }
        """,
        render_kwds=dict(s=CombinatorCall(dtype, m1num=m1num, m2num=m2num, snum=snum)))

    a = get_test_array(N, dtype)
    b = get_test_array(N, dtype)
    a_dev = some_ctx.to_device(a)
    b_dev = some_ctx.to_device(b)
    dest_dev = some_ctx.empty_like(a_dev)

    module.test(dest_dev, a_dev, b_dev, local_size=N, global_size=N)
    ref = (a * b + m1num) + ((a * b + m2num) - snum)

    assert diff_is_negligible(dest_dev.get(), ref)
