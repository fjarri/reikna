import numpy
from grunnur import dtypes
from numpy.lib.stride_tricks import as_strided

from reikna.helpers import wrap_in_tuple

# Default tolerances for numpy.allclose().
# Should be enough to detect a error, but not enough to trigger a fail
# in case of a slightly imprecise implementation of some operation on a GPU.
SINGLE_RTOL = 1e-5
SINGLE_ATOL = 1e-8

DOUBLE_RTOL = 1e-11
DOUBLE_ATOL = 1e-11


def get_test_array_like(arr, **_kwds):
    metadata = arr.as_array_metadata()
    return get_test_array(
        shape=metadata.shape,
        dtype=metadata.dtype,
        strides=metadata.strides,
        offset=metadata.first_element_offset,
    )


def get_test_array(shape, dtype, strides=None, offset=0, high=None, *, no_zeros=False):
    shape = wrap_in_tuple(shape)
    dtype = numpy.dtype(dtype)

    if offset != 0:
        raise NotImplementedError("Non-zero offsets are not supported")

    if dtype.names is not None:
        result = numpy.empty(shape, dtype)
        for name in dtype.names:
            result[name] = get_test_array(shape, dtype[name], no_zeros=no_zeros, high=high)
    else:
        rng = numpy.random.default_rng()
        if dtypes.is_integer(dtype):
            low = 1 if no_zeros else 0
            if high is None:
                high = 100  # will work even with signed chars
            get_arr = lambda: rng.integers(low, high, shape, dtype=dtype)
        else:
            low = 0.01 if no_zeros else 0
            if high is None:
                high = 1.0
            get_arr = lambda: rng.uniform(low, high, shape).astype(dtype)

        result = get_arr() + 1j * get_arr() if dtypes.is_complex(dtype) else get_arr()

    if strides is not None:
        result = as_strided(result, result.shape, strides)

    return result


def diff_is_negligible(m, m_ref, atol=None, rtol=None, *, verbose=True):
    if m.dtype.names is not None:
        return all(diff_is_negligible(m[name], m_ref[name]) for name in m.dtype.names)

    assert m.dtype == m_ref.dtype

    if dtypes.is_integer(m.dtype):
        close = m == m_ref
    else:
        if atol is None:
            atol = DOUBLE_ATOL if dtypes.is_double(m.dtype) else SINGLE_ATOL
        if rtol is None:
            rtol = DOUBLE_RTOL if dtypes.is_double(m.dtype) else SINGLE_RTOL

        close = numpy.isclose(m, m_ref, atol=atol, rtol=rtol)

    if close.all():
        return True

    if verbose:
        far_idxs = numpy.vstack(numpy.where(close)).T
        print(  # noqa: T201
            f"diff_is_negligible() with atol={atol} and rtol={rtol} "
            f"found {far_idxs.shape[0]} differences, first ones are:"
        )
        for idx, _ in zip(far_idxs, range(10), strict=True):
            t_idx = tuple(idx)
            print(f"idx: {t_idx}, test: {m[t_idx]}, ref: {m_ref[t_idx]}")  # noqa: T201

    return False
