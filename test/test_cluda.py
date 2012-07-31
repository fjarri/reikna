import itertools

import pytest

import tigger.cluda as cluda
import tigger.cluda.dtypes as dtypes
from helpers import *


TEST_DTYPES = [
	numpy.int8, numpy.int16, numpy.int32, numpy.int64,
	numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
	numpy.float32, numpy.float64,
	numpy.complex64, numpy.complex128
]


def simple_context_test(ctx):
    shape = (1000,)
    dtype = numpy.float32

    a = get_test_array(shape, dtype)
    a_dev = ctx.to_device(a)
    a_back = ctx.from_device(a_dev)

    assert diff_is_negligible(a, a_back)

def test_create_new_context(cluda_api):
	ctx = cluda_api.Context.create()
	simple_context_test(ctx)
	ctx.release()

def test_connect_to_context(cluda_api):
	ctx = cluda_api.Context.create()

	ctx2 = cluda_api.Context(ctx.context)
	ctx3 = cluda_api.Context(ctx.context, async=False)

	simple_context_test(ctx)
	simple_context_test(ctx2)
	simple_context_test(ctx3)

	ctx3.release()
	ctx2.release()

	ctx.release()

def test_connect_to_context_and_stream(cluda_api):
	ctx = cluda_api.Context.create()
	stream = ctx.create_stream()

	ctx2 = cluda_api.Context(ctx.context, stream=stream)
	ctx3 = cluda_api.Context(ctx.context, stream=stream, async=False)

	simple_context_test(ctx)
	simple_context_test(ctx2)
	simple_context_test(ctx3)

	ctx3.release()
	ctx2.release()

	ctx.release()

def test_transfers(ctx):
	a = get_test_array(1024, numpy.float32)

	def to_device1(x):
		return ctx.to_device(x)
	def to_device2(x):
		y = ctx.empty_like(x)
		ctx.to_device(x, dest=y)
		return y
	def from_device1(x):
		return x.get()
	def from_device2(x):
		return ctx.from_device(x)
	def from_device3(x):
		y = numpy.empty(x.shape, x.dtype)
		ctx.from_device(x, dest=y)
		return y
	def from_device4(x):
		y = ctx.from_device(x, async=True)
		ctx.synchronize()
		return y
	def from_device5(x):
		y = numpy.empty(x.shape, x.dtype)
		ctx.from_device(x, dest=y, async=True)
		ctx.synchronize()
		return y

	to_device = (to_device1, to_device2)
	from_device = (from_device1, from_device2, from_device3, from_device4, from_device5)

	for to_d, from_d in itertools.product(to_device, from_device):
		a_device = to_d(a)
		a_copy = ctx.copy_array(a_device)
		a_back = from_d(a_copy)
		assert diff_is_negligible(a, a_back)

@pytest.mark.parametrize(
	"dtype", TEST_DTYPES,
	ids=[dtypes.normalize_type(dtype).name for dtype in TEST_DTYPES])
def test_dtype_support(ctx, dtype):
	# Test passes if either context correctly reports that it does not support given dtype,
	# or it successfully compiles kernel that operates with this dtype.

	N = 256

	if not ctx.supports_dtype(dtype):
		return

	module = ctx.compile(
	"""
	KERNEL void test(
		GLOBAL_MEM ${ctype} *dest, GLOBAL_MEM ${ctype} *a, GLOBAL_MEM ${ctype} *b)
	{
	  const int i = LID_0;
	  ${ctype} temp = ${func.mul(dtype, dtype)}(a[i], b[i]);
	  dest[i] = ${func.div(dtype, dtype)}(temp, b[i]);
	}
	""", ctype=dtypes.ctype(dtype), dtype=dtype)

	test = module.test

	# we need results to fit even in unsigned char
	a = get_test_array(N, dtype, high=8)
	b = get_test_array(N, dtype, no_zeros=True, high=8)

	a_dev = ctx.to_device(a)
	b_dev = ctx.to_device(b)
	dest_dev = ctx.empty_like(a_dev)
	test(dest_dev, a_dev, b_dev, local_size=(N,), global_size=(N,))
	assert diff_is_negligible(ctx.from_device(dest_dev), a)
