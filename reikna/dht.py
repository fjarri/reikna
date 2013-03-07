import numpy
import unittest, itertools
from numpy.polynomial import Hermite as H

from reikna.helpers import *
from reikna.core import *
import reikna.cluda.dtypes as dtypes
from reikna.transpose import Transpose
from reikna.matrixmul import MatrixMul


def factorial(n):
    """
    Replacement of ``scipy.misc.factorial()``.
    First, it avoids requiring scipy; second, it returns integer instead of float
    (which gives more precision for subsequent calculations).
    """
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res


def hermite(mode):
    """Returns an orthonormal Hermite polynomial"""
    def func(x):
        norm = 1. / (numpy.pi ** 0.25) / numpy.sqrt(float(factorial(mode) * 2 ** mode))
        return H([0] * mode + [1])(x) * norm

    return func


def h_roots(n):
    """
    Recursive root finding algorithm, taken from Numerical Recipes.
    More accurate than the standard h_roots() from scipy.
    """

    EPS = 1.0e-16
    PIM4 = numpy.pi ** (-0.25) # 0.7511255444649425
    MAXIT = 20 # Maximum iterations.

    x = numpy.empty(n)
    w = numpy.empty(n)
    m = (n + 1) // 2

    z = 0

    for i in xrange(m):
        if i == 0: # Initial guess for the largest root.
            z = numpy.sqrt(float(2 * n + 1)) - 1.85575 * float(2 * n + 1) ** (-0.16667)
        elif i == 1:
            z -= 1.14 * float(n) ** 0.426 / z
        elif i == 2:
            z = 1.86 * z + 0.86 * x[0]
        elif i == 3:
            z = 1.91 * z + 0.91 * x[1]
        else:
            z = 2.0 * z + x[i - 2]

        for its in xrange(MAXIT):
            p1 = PIM4
            p2 = 0.0
            p3 = 0.0
            for j in xrange(n):
                p3 = p2
                p2 = p1
                p1 = z * numpy.sqrt(2.0 / (j + 1)) * p2 - numpy.sqrt(float(j) / (j + 1)) * p3

            pp = numpy.sqrt(float(2 * n)) * p2
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) <= EPS:
                break

        if its >= MAXIT:
            raise Exception("Too many iterations")

        x[n - 1 - i] = z
        x[i] = -z
        w[i] = 2.0 / (pp ** 2)
        w[n - 1 - i] = w[i]

    return x, w


def get_spatial_points(modes, order, add_points=0):
    """
    Returns the number of points in coordinate space
    which allows the calculation of up to ``order`` of any function
    initially defined on mode space.
    ``add_points`` can be used to increase the size of the grid.
    """
    points = ((modes - 1) * (order + 1)) // 2 + 1

    # TODO: Population calculated in mode and in x-space is slightly different
    # The more points we take in addition to minimum necessary for precise
    # G.-H. quadrature, the less is the difference.
    # Looks like it is not a bug, just inability to integrate Hermite function
    # in x-space precisely (oscillates too fast maybe?).
    # But this still requres investigation.
    # In addition: with dp != 0 X-M-X transform of TF state
    # gives non-smooth curve. Certainly TF state has some higher harmonics
    # (infinite number of them, to be precise), but why it is smooth when dp = 0?
    points += add_points

    return points


def get_spatial_grid(modes, order, add_points=0):
    """
    Returns a pair of arrays ``(points, weights)`` for Gauss-Hermite quadrature.
    """
    points = get_spatial_points(modes, order, add_points=add_points)
    roots, weights = h_roots(points)

    return roots * numpy.sqrt(2.0 / (order + 1)), \
        weights * numpy.exp(roots ** 2) * numpy.sqrt(2.0 / (order + 1))


def harmonic(mode):
    """
    Returns an eigenfunction of order ``n`` for the harmonic oscillator.
    """
    H = hermite(mode)
    return lambda x: H(x) * numpy.exp(-(x ** 2) / 2)


def get_transformation_matrix(modes, order, add_points):
    """
    Returns the the matrix of values of mode functions taken at
    points of the spatial grid.
    """
    x, _ = get_spatial_grid(modes, order, add_points=add_points)

    res = numpy.zeros((modes, x.size))

    for mode in range(modes):
        res[mode, :] = harmonic(mode)(x)

    return res


class DHT(Computation):
    """
    Discrete transform to and from harmonic oscillator modes.
    See Dion & Cances, `PRE 67(4) 046706 (2003) <http://dx.doi.org/10.1103/PhysRevE.67.046706>`_
    for the description of the algorithm.

    .. py:method:: prepare_for(output, input, direction, axes=None)

        :param output: output array.
        :param input: input array.
        :param inverse: ``False`` for forward (coordinate space -> mode space) transform,
            ``True`` for inverse (mode space -> coordinate space) transform.
        :param axes: a tuple with axes over which to perform the transform.
            If not given, the transform is performed over all the axes.
        :param order: if ``F`` is a function in mode space, the number of spatial points
            is chosen so that the transformation ``DHT[(DHT^{-1}[F])^order]`` could be performed.
    """

    def _get_argnames(self):
        return ('output',), ('input',), tuple()

    def _get_basis_for(self, output, input, inverse=False, order=1, axes=None):
        bs = AttrDict()

        assert output.dtype == input.dtype

        if not inverse:
            coord_arr = input
            mode_arr = output
        else:
            coord_arr = output
            mode_arr = input

        assert len(output.shape) == len(input.shape)
        if axes is None:
            axes = tuple(range(len(output.shape)))
        else:
            axes = tuple(axes)

        bs.add_points = {}
        for axis in axes:
            modes = mode_arr.shape[axis]
            points = coord_arr.shape[axis]
            add_points = points - get_spatial_points(modes, order)
            assert add_points >= 0, "Not enough spatial points in axis " + str(axis) + \
                " to perform precise transformation"
            bs.add_points[axis] = add_points

        bs.axes = list(reversed(sorted(axes)))
        bs.input_shape = input.shape
        bs.output_shape = output.shape
        bs.dtype = output.dtype
        bs.inverse = inverse
        bs.order = order

        return bs

    def _get_argvalues(self, basis):
        return dict(
            output=ArrayValue(basis.output_shape, basis.dtype),
            input=ArrayValue(basis.input_shape, basis.dtype))

    def _construct_operations(self, basis, device_params):

        operations = self._get_operation_recorder()
        plan = []

        coord_shape = basis.output_shape if basis.inverse else basis.input_shape
        mode_shape = basis.input_shape if basis.inverse else basis.output_shape

        p_dtype = dtypes.real_for(basis.dtype) if dtypes.is_complex(basis.dtype) else basis.dtype

        current_mem = 'input'
        current_shape = list(basis.input_shape)
        current_axes = range(len(basis.output_shape))

        for i, axis in enumerate(basis.axes):

            # Transpose the current array so that the ``axis`` is in the end of axes list
            cur_pos = current_axes.index(axis)
            if cur_pos != len(current_axes) - 1:
                new_axes = list(current_axes)
                new_axes[cur_pos], new_axes[-1] = new_axes[-1], new_axes[cur_pos]
                new_shape = list(current_shape)
                new_shape[cur_pos], new_shape[-1] = new_shape[-1], new_shape[cur_pos]

                tr_output = operations.add_allocation(new_shape, basis.dtype)
                transpose = self.get_nested_computation(Transpose)
                operations.add_computation(transpose, tr_output, current_mem, axes=tr_axes)

                current_mem = tr_output
                current_shape = new_shape
                current_axes = new_axes

            # Prepare the transformation matrix
            p = get_transformation_matrix(mode_shape[axis], basis.order, basis.add_points[axis])
            p = p.astype(p_dtype)
            if not basis.inverse:
                _, w = get_spatial_grid(mode_shape[axis], basis.order, basis.add_points[axis])
                ww = numpy.tile(w.reshape(w.size, 1).astype(p_dtype), (1, mode_shape[axis]))
                p = p.transpose() * ww
            tr_matrix = operations.add_const_allocation(p)

            # Add matrix multiplication
            new_shape = list(current_shape)
            new_shape[-1] = p.shape[0]

            if i == len(basis.axes) - 1 and current_axes == range(len(basis.input_shape)):
                dot_output = 'output'
            else:
                # Cannot write to output if it is not the last transform,
                # or if we need to return to the initial axes order
                dot_output = operations.add_allocation(new_shape, basis.dtype)

            dot = self.get_nested_computation(MatrixMul)
            print "adding dot ", new_shape, current_shape, p.shape
            operations.add_computation(dot, dot_output, current_mem, tr_matrix)

            current_shape = new_shape
            current_mem = dot_output

        # Return to original shape if necessary
        if current_axes != range(len(basis.input_shape)):
            transpose = self.get_nested_computation(Transpose)
            operations.add_computation(transpose, 'output', current_mem, axes=current_axes)

        return operations
