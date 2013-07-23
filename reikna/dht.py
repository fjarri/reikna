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

    for i in range(m):
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

        for its in range(MAXIT):
            p1 = PIM4
            p2 = 0.0
            p3 = 0.0
            for j in range(n):
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


def get_spatial_grid_and_weights(modes, order, add_points=0):
    """
    Returns a pair of arrays ``(points, weights)`` for Gauss-Hermite quadrature.
    """
    points = get_spatial_points(modes, order, add_points=add_points)
    roots, weights = h_roots(points)

    return roots * numpy.sqrt(2.0 / (order + 1)), \
        weights * numpy.exp(roots ** 2) * numpy.sqrt(2.0 / (order + 1))


def get_spatial_grid(modes, order, add_points=0):
    """
    Returns the spatial grid required to calculate the ``order`` power of a function
    defined in the harmonic mode space of the size ``modes``.
    If ``add_points`` is 0, the grid has the minimum size required for exact
    transformation back to the mode space.
    """
    return get_spatial_grid_and_weights(modes, order, add_points=add_points)[0]


def get_spatial_weights(modes, order, add_points=0):
    return get_spatial_grid_and_weights(modes, order, add_points=add_points)[1]


def harmonic(mode):
    r"""
    Returns an eigenfunction of order :math:`n = \mathrm{mode}` for the harmonic oscillator:

    .. math::
        \phi_{n}
        = \frac{1}{\sqrt[4]{\pi} \sqrt{2^n n!}} H_n(x) \exp(-x^2/2),

    where :math:`H_n` is the :math:`n`-th order "physicists'" Hermite polynomial.
    The normalization is chosen so that :math:`\int \phi_n^2(x) dx = 1`.
    """
    H = hermite(mode)
    return lambda x: H(x) * numpy.exp(-(x ** 2) / 2)


def get_transformation_matrix(modes, order, add_points):
    """
    Returns the the matrix of values of mode functions taken at
    points of the spatial grid.
    """
    x = get_spatial_grid(modes, order, add_points=add_points)

    res = numpy.zeros((modes, x.size))

    for mode in range(modes):
        res[mode, :] = harmonic(mode)(x)

    return res


class DHT(Computation):
    r"""
    Discrete transform to and from harmonic oscillator modes.
    With ``inverse=True`` transforms a function defined by its expansion
    :math:`C_m,\,m=0 \ldots M-1` in the mode space with mode functions
    from :py:func:`~reikna.dht.harmonic`,
    to the coordinate space (:math:`F(x)` on the grid :math:`x`
    from :py:func:`~reikna.dht.get_spatial_grid`).
    With ``inverse=False`` guarantees to recover first :math:`M` modes of :math:`F^k(x)`,
    where :math:`k` is the ``order`` parameter.

    For multiple dimensions the operation is the same,
    and the mode functions are products of 1D mode functions, i.e.
    :math:`\phi_{l,m,n}^{3D}(x,y,z) = \phi_l(x) \phi_m(y) \phi_n(z)`.

    For the detailed description of the algorithm, see Dion & Cances,
    `PRE 67(4) 046706 (2003) <http://dx.doi.org/10.1103/PhysRevE.67.046706>`_

    :param output: output array.
        If ``inverse=False``, its shape is used to define the mode space size.
    :param input: input array.
        If ``inverse=True``, its shape is used to define the mode space size.
    :param inverse: ``False`` for forward (coordinate space -> mode space) transform,
        ``True`` for inverse (mode space -> coordinate space) transform.
    :param axes: a tuple with axes over which to perform the transform.
        If not given, the transform is performed over all the axes.
    :param order: if ``F`` is a function in mode space, the number of spatial points
        is chosen so that the transformation ``DHT[(DHT^{-1}[F])^order]`` could be performed.
    """

    def __init__(self, mode_arr, add_points=None, inverse=False, order=1, axes=None):

        if axes is None:
            axes = tuple(range(len(mode_arr.shape)))
        else:
            axes = tuple(axes)
        self._axes = list(sorted(axes))

        if add_points is None:
            add_points = [0] * len(mode_arr.shape)
        else:
            add_points = list(add_points)
        self._add_points = add_points

        coord_shape = list(mode_arr.shape)
        for axis in range(len(mode_arr.shape)):
            if axis in axes:
                coord_shape[axis] = get_spatial_points(
                    mode_arr.shape[axis], order, add_points=add_points[axis])
        coord_arr = Type(mode_arr.dtype, shape=coord_shape)

        if not inverse:
            output_arr = mode_arr
            input_arr = coord_arr
        else:
            output_arr = coord_arr
            input_arr = mode_arr

        self._inverse = inverse
        self._order = order

        Computation.__init__(self, [
            Parameter('output', Annotation(output_arr, 'o')),
            Parameter('input', Annotation(input_arr, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input):

        plan = plan_factory()

        dtype = input.dtype
        mode_shape = input.shape if self._inverse else output.shape

        p_dtype = dtypes.real_for(dtype) if dtypes.is_complex(dtype) else dtype

        current_mem = input
        seq_axes = list(range(len(input.shape)))
        current_axes = list(range(len(input.shape)))

        for i, axis in enumerate(self._axes):

            # Transpose the current array so that the ``axis`` is in the end of axes list

            cur_pos = current_axes.index(axis)
            if cur_pos != len(current_axes) - 1:

                # We can move the target axis to the end in different ways,
                # but this one will require only one transpose kernel.
                optimal_transpose = lambda seq: seq[:cur_pos] + seq[cur_pos+1:] + [seq[cur_pos]]

                tr_axes = optimal_transpose(seq_axes)
                new_axes = optimal_transpose(current_axes)

                transpose = Transpose(current_mem, axes=tr_axes)
                tr_output = plan.temp_array_like(transpose.parameter.output)
                plan.computation_call(transpose, tr_output, current_mem)

                current_mem = tr_output
                current_axes = new_axes

            # Prepare the transformation matrix

            p = get_transformation_matrix(mode_shape[axis], self._order, self._add_points[axis])
            p = p.astype(p_dtype)
            if not self._inverse:
                w = get_spatial_weights(mode_shape[axis], self._order, self._add_points[axis])
                ww = numpy.tile(w.reshape(w.size, 1).astype(p_dtype), (1, mode_shape[axis]))
                p = p.transpose() * ww
            tr_matrix = plan.persistent_array(p)

            # Add the matrix multiplication

            dot = MatrixMul(current_mem, tr_matrix)
            if i == len(self._axes) - 1 and current_axes == seq_axes:
                dot_output = output
            else:
                # Cannot write to output if it is not the last transform,
                # or if we need to return to the initial axes order
                dot_output = plan.temp_array_like(dot.parameter.output)
            plan.computation_call(dot, dot_output, current_mem, tr_matrix)
            current_mem = dot_output

        # If we ended up with the wrong order of axes,
        # return to the original order.

        if current_axes != seq_axes:
            tr_axes = [current_axes.index(i) for i in range(len(current_axes))]
            transpose = Transpose(current_mem, axes=tr_axes)
            plan.add_computation(transpose, output, current_mem)

        return plan
