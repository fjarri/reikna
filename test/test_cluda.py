import pytest

import tigger.cluda as cluda
from helpers import *


def simple_context_test(ctx):
    shape = (1000,)
    dtype = numpy.float32

    a = getTestArray(shape, dtype)
    a_dev = ctx.to_device(a)
    a_back = ctx.from_device(a_dev)

    assert diff(a, a_back) < SINGLE_EPS

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
