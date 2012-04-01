import numpy

def getTestArray(shape, dtype):
	return numpy.random.normal(size=shape).astype(dtype)

def diff(m1, m2):
	return numpy.linalg.norm(m1 - m2) / numpy.linalg.norm(m1)
