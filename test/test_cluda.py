import pytest

import tigger.cluda as cluda
from helpers import *


def simple_context_test(ctx):
    shape = (1000,)
    dtype = numpy.float32

    a = getTestArray(shape, dtype)
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

def test_compilation(ctx):

	N = 256
	coeff = 2

	module = ctx.compile(
	"""
	KERNEL void multiply_them(GLOBAL_MEM float *dest, GLOBAL_MEM float *a, GLOBAL_MEM float *b)
	{
	  const int i = LID_0;
	  dest[i] = ${func.mul(numpy.float32, numpy.float32)}(a[i], b[i]) * (float)${coeff};
	}
	""", coeff=coeff)

	multiply_them = module.multiply_them

	a = numpy.random.randn(N).astype(numpy.float32)
	b = numpy.random.randn(N).astype(numpy.float32)
	a_dev = ctx.to_device(a)
	b_dev = ctx.to_device(b)
	dest_dev = ctx.empty_like(a_dev)

	multiply_them(dest_dev, a_dev, b_dev, block=(N,1,1), grid=(1,1))
	assert diff(ctx.from_device(dest_dev), a * b * coeff) < SINGLE_EPS
