import numpy
import pytest

import reikna.cluda as cluda
import reikna.cluda.dtypes as dtypes


def flatten_dtype(dtype, prefix=[]):

    if dtype.names is None:
        return [(prefix, dtype)]
    else:
        result = []
        for name in dtype.names:
            nested_dtype, offset = dtype.fields[name]
            result += flatten_dtype(nested_dtype, prefix=prefix + [name])
        return result


def extract_field(arr, path):
    if len(path) == 0:
        return arr
    else:
        return extract_field(arr[path[0]], path[1:])


def check_struct_fill(thr, dtype):
    """
    Fill every field of the given ``dtype`` with its number and check the results.
    This helps to detect issues with alignment in the struct.
    """
    struct = dtypes.get_struct_module(thr, dtype)

    program = thr.compile(
    """
    KERNEL void test(GLOBAL_MEM ${struct} *dest)
    {
      const SIZE_T i = get_global_id(0);
      ${struct} res;

      %for i, field_info in enumerate(flatten_dtype(dtype)):
      res.${".".join(field_info[0])} = ${i};
      %endfor

      dest[i] = res;
    }
    """, render_kwds=dict(
        struct=struct,
        dtype=dtype,
        flatten_dtype=flatten_dtype))

    test = program.test

    a_dev = thr.array(128, dtype)
    test(a_dev, global_size=128)
    a = a_dev.get()

    for i, field_info in enumerate(flatten_dtype(dtype)):
        path, _ = field_info
        assert (extract_field(a, path) == i).all()


def test_hardcoded_alignment(thr):
    """
    Test the correctness of alignment for an explicit set of field offsets.
    """

    dtype_nested = numpy.dtype(dict(
        names=['val1', 'pad'],
        formats=[numpy.int32, numpy.int8],
        offsets=[0, 4],
        itemsize=8))

    dtype = numpy.dtype(dict(
        names=['val1', 'val2', 'nested'],
        formats=[numpy.int32, numpy.int16, dtype_nested],
        offsets=[0, 4, 8],
        itemsize=32))

    check_struct_fill(thr, dtype)


def test_adjusted_alignment(thr):
    """
    Test the correctness of alignment for field offsets adjusted automatically.
    """

    dtype_nested = numpy.dtype([
        ('val1', numpy.int32), ('pad', numpy.int8)])

    dtype = numpy.dtype([
        ('val1', numpy.int32),
        ('val2', numpy.int16),
        ('nested', dtype_nested)])

    dtype = dtypes.adjust_alignment(thr, dtype)

    check_struct_fill(thr, dtype)


def test_nested_array(thr):
    """
    Check that structures with nested arrays are processed correctly.
    """
    dtype_nested = numpy.dtype(dict(
        names=['val1', 'pad'],
        formats=[numpy.int8, numpy.int8]))

    dtype = numpy.dtype(dict(
        names=['pad', 'struct_arr', 'regular_arr'],
        formats=[numpy.int32, numpy.dtype((dtype_nested, 2)), numpy.dtype((numpy.int16, 3))]))

    dtype = dtypes.adjust_alignment(thr, dtype)
    struct = dtypes.get_struct_module(thr, dtype)

    program = thr.compile(
    """
    KERNEL void test(GLOBAL_MEM ${struct} *dest)
    {
      const SIZE_T i = get_global_id(0);
      ${struct} res;

      res.struct_arr[0].val1 = i + 0;
      res.struct_arr[1].val1 = i + 1;

      res.regular_arr[0] = i + 2;
      res.regular_arr[1] = i + 3;
      res.regular_arr[2] = i + 4;

      dest[i] = res;
    }
    """, render_kwds=dict(
        struct=struct,
        dtype=dtype,
        flatten_dtype=flatten_dtype))

    test = program.test

    a_dev = thr.array(64, dtype)
    test(a_dev, global_size=64)
    a = a_dev.get()

    idxs = numpy.arange(64)
    assert (a['struct_arr'][:,0]['val1'] == idxs + 0).all()
    assert (a['struct_arr'][:,1]['val1'] == idxs + 1).all()
    assert (a['regular_arr'][:,0] == idxs + 2).all()
    assert (a['regular_arr'][:,1] == idxs + 3).all()
    assert (a['regular_arr'][:,2] == idxs + 4).all()

