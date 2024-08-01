# Reikna, the pure Python GPGPU library

[![pypi package][pypi-image]][pypi-link] ![License][pypi-license-image] [![Docs][rtd-image]][rtd-link] [![Coverage][cov-image]][cov-link]

[pypi-image]: https://img.shields.io/pypi/v/reikna
[pypi-link]: https://pypi.org/project/reikna/
[pypi-license-image]: https://img.shields.io/pypi/l/reikna
[rtd-image]: https://readthedocs.org/projects/reikna/badge/?version=latest
[rtd-link]: https://reikna.readthedocs.io/en/latest/
[cov-image]: https://codecov.io/gh/fjarri/reikna/branch/master/graph/badge.svg
[cov-link]: https://codecov.io/gh/fjarri/reikna


`Reikna` is a library containing various GPU algorithms built on top of [PyCUDA](http://documen.tician.de/pycuda) and [PyOpenCL](http://documen.tician.de/pyopencl).
The main design goals are:

* separation of computation cores (matrix multiplication, random numbers generation etc) from simple transformations on their input and output values (scaling, typecast etc);
* separation of the preparation and execution stage, maximizing the performance of the execution stage at the expense of the preparation stage (in other words, aiming at large simulations)
* partial abstraction from CUDA/OpenCL

For more information proceed to the [project documentation page](http://reikna.publicfields.net). If you have a general question that does not qualify as an issue, you can ask it at the [discussion forum](https://github.com/fjarri/reikna/discussions).

