import pytest

from helpers import *
from reikna.helpers import *


TEMPLATE = template_from("""
<%def name="multiplier()">
<%
    ctype = dtypes.ctype(dtype)
%>
WITHIN_KERNEL ${ctype} ${_prefix}multiplier(${ctype} x, ${ctype} y)
{
    return ${func.mul(dtype, dtype, out=dtype)}(x, y) + ${num};
}
</%def>


<%def name="subtractor()">
<%
    ctype = dtypes.ctype(dtype)
%>
WITHIN_KERNEL ${ctype} ${_prefix}subtractor(${ctype} x, ${ctype} y)
{
    return ${_prefix.m}multiplier(x, y) - ${num};
}
</%def>


<%def name="combinator()">
<%
    ctype = dtypes.ctype(dtype)
%>
WITHIN_KERNEL ${ctype} ${_prefix}combinator(${ctype} x, ${ctype} y)
{
    return ${_prefix.m}multiplier(x, y) + ${_prefix.s}subtractor(x, y);
}
</%def>

<%def name="snippet(modules, a, b)">
dest[idx] = ${modules.c}combinator(${a}[idx], ${b}[idx]);
</%def>
""")


def multiplier(dtype, num=1):
    return (
        TEMPLATE.get_def('multiplier'),
        dict(dtype=dtype, num=num),
        None)


def subtractor(dtype, mnum=1, num=1):
    return (
        TEMPLATE.get_def('subtractor'),
        dict(dtype=dtype, num=num),
        dict(m=multiplier(dtype, num=mnum)))


def combinator(dtype, m1num=1, m2num=1, snum=1):
    return (
        TEMPLATE.get_def('combinator'),
        dict(dtype=dtype),
        dict(
            m=multiplier(dtype, num=m1num),
            s=subtractor(dtype, mnum=m2num, num=snum)))


def snippet(dtype, m1num=1, m2num=1, snum=1):
    return (
        TEMPLATE.get_def('snippet'),
        None,
        dict(c=combinator(dtype, m1num=m1num, m2num=m2num, snum=snum)))


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
            dest[idx] = ${_prefix.c}combinator(a[idx], b[idx]);
        }
        """,
        modules=dict(c=combinator(dtype, m1num=m1num, m2num=m2num, snum=snum)))

    a = get_test_array(N, dtype)
    b = get_test_array(N, dtype)
    a_dev = some_ctx.to_device(a)
    b_dev = some_ctx.to_device(b)
    dest_dev = some_ctx.empty_like(a_dev)

    module.test(dest_dev, a_dev, b_dev, local_size=N, global_size=N)
    ref = (a * b + m1num) + ((a * b + m2num) - snum)

    assert diff_is_negligible(dest_dev.get(), ref)


def atest_snippet(some_ctx):

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
            ${snippets.s('a', 'b')}
        }
        """,
        snippets=dict(s=snippet(dtype, m1num=m1num, m2num=m2num, snum=snum)))

    a = get_test_array(N, dtype)
    b = get_test_array(N, dtype)
    a_dev = some_ctx.to_device(a)
    b_dev = some_ctx.to_device(b)
    dest_dev = some_ctx.empty_like(a_dev)

    module.test(dest_dev, a_dev, b_dev, local_size=N, global_size=N)
    ref = (x * y + m1num) + ((x * y + m2num) - snum)

    assert diff_is_negligible(dest_dev.get(), ref)
