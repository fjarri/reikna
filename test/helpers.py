import numpy

def getTestArray(shape, dtype):
	return numpy.random.normal(size=shape).astype(dtype)

def diff(m, m_ref):
	return numpy.linalg.norm(m - m_ref) / numpy.linalg.norm(m_ref)
