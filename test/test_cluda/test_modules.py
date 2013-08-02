import pytest

from helpers import *
from reikna.helpers import *
from reikna.cluda import Module, Snippet
import reikna.cluda.functions as functions


TEMPLATE = template_from("""
<%def name="multiplier(prefix)">
<%
    ctype = dtypes.ctype(dtype)
%>
WITHIN_KERNEL ${ctype} ${prefix}multiplier(${ctype} x, ${ctype} y)
{
    return ${mul}(x, y) + ${num};
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


def multiplier(dtype, num=1):
    mul = functions.mul(dtype, dtype, out_dtype=dtype)
    return Module(
        TEMPLATE.get_def('multiplier'),
        render_kwds=dict(dtype=dtype, num=num, mul=mul))


def subtractor(dtype, mnum=1, num=1):
    m = multiplier(dtype, num=mnum)
    return Module(
        TEMPLATE.get_def('subtractor'),
        render_kwds=dict(dtype=dtype, num=num, m=m))


def combinator(dtype, m1num=1, m2num=1, snum=1):
    m = multiplier(dtype, num=m1num)
    s = subtractor(dtype, mnum=m2num, num=snum)
    return Module(
        TEMPLATE.get_def('combinator'),
        render_kwds=dict(dtype=dtype, m=m, s=s))


def combinator_call(dtype, m1num=1, m2num=1, snum=1):
    c = combinator(dtype, m1num=m1num, m2num=m2num, snum=snum)
    return Snippet(
        TEMPLATE.get_def('snippet'),
        render_kwds=dict(c=c))


def test_modules(some_thr):

    dtype = numpy.float32
    m1num = 2
    m2num = 3
    snum = 10
    N = 128

    program = some_thr.compile(
        """
        KERNEL void test(GLOBAL_MEM float *dest, GLOBAL_MEM float *a, GLOBAL_MEM float *b)
        {
            const SIZE_T idx = get_global_id(0);
            dest[idx] = ${c}combinator(a[idx], b[idx]);
        }
        """,
        render_kwds=dict(c=combinator(dtype, m1num=m1num, m2num=m2num, snum=snum)))

    a = get_test_array(N, dtype)
    b = get_test_array(N, dtype)
    a_dev = some_thr.to_device(a)
    b_dev = some_thr.to_device(b)
    dest_dev = some_thr.empty_like(a_dev)

    program.test(dest_dev, a_dev, b_dev, local_size=N, global_size=N)
    ref = (a * b + m1num) + ((a * b + m2num) - snum)

    assert diff_is_negligible(dest_dev.get(), ref)


def test_snippet(some_thr):

    dtype = numpy.float32
    m1num = 2
    m2num = 3
    snum = 10
    N = 128

    program = some_thr.compile(
        """
        KERNEL void test(GLOBAL_MEM float *dest, GLOBAL_MEM float *a, GLOBAL_MEM float *b)
        {
            const SIZE_T idx = get_global_id(0);
            ${s('a', 'b')}
        }
        """,
        render_kwds=dict(s=combinator_call(dtype, m1num=m1num, m2num=m2num, snum=snum)))

    a = get_test_array(N, dtype)
    b = get_test_array(N, dtype)
    a_dev = some_thr.to_device(a)
    b_dev = some_thr.to_device(b)
    dest_dev = some_thr.empty_like(a_dev)

    program.test(dest_dev, a_dev, b_dev, local_size=N, global_size=N)
    ref = (a * b + m1num) + ((a * b + m2num) - snum)

    assert diff_is_negligible(dest_dev.get(), ref)


CACHING_TEMPLATE = template_from("""
<%def name="data_structures(prefix)">
typedef struct ${prefix}_TEST
{
    int a;
    int b;
} ${prefix}TEST;
</%def>


<%def name="module1(prefix)">
WITHIN_KERNEL ${data}TEST ${prefix}(${data}TEST x)
{
    ${data}TEST res;
    res.a = x.b;
    res.b = x.a;
    return res;
}
</%def>


<%def name="module2(prefix)">
WITHIN_KERNEL ${data}TEST ${prefix}(${data}TEST x)
{
    ${data}TEST res;
    res.a = x.a + 2;
    res.b = x.b + 3;
    return res;
}
</%def>
""")

caching_data = Module(CACHING_TEMPLATE.get_def('data_structures'))
caching_module1 = Module(CACHING_TEMPLATE.get_def('module1'), render_kwds=dict(data=caching_data))
caching_module2 = Module(CACHING_TEMPLATE.get_def('module2'), render_kwds=dict(data=caching_data))


def test_caching(some_thr):
    """
    Tests that the root module with the data structure declaration is rendered only once,
    despite being used both by kernel and by two other modules.
    This allows the data structure to be passed to functions from both modules without type errors.
    """

    dtype = numpy.int32
    size = 128

    program = some_thr.compile(
        """
        KERNEL void test(GLOBAL_MEM int *dest)
        {
            const SIZE_T idx = get_global_id(0);
            ${data}TEST x;
            x.a = dest[idx];
            x.b = dest[idx + ${size}];

            x = ${module1}(x);
            x = ${module2}(x);

            dest[idx] = x.a;
            dest[idx + ${size}] = x.b;
        }
        """,
        render_kwds=dict(
            size=size,
            module1=caching_module1, module2=caching_module2, data=caching_data))

    a = get_test_array((2, size), dtype)
    a_dev = some_thr.to_device(a)

    program.test(a_dev, local_size=size, global_size=size)
    a_ref = numpy.vstack([a[1] + 2, a[0] + 3])

    assert diff_is_negligible(a_dev.get(), a_ref)
