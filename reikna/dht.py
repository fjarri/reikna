import numpy
from numpy.polynomial import Hermite

import reikna.cluda.dtypes as dtypes
from reikna.core import Computation, Parameter, Annotation, Type
from reikna.algorithms import Transpose
from reikna.linalg import MatrixMul


def factorial(num):
    """
    Replacement of ``scipy.misc.factorial()``.
    First, it avoids requiring scipy; second, it returns integer instead of float
    (which gives more precision for subsequent calculations).
    """
    res = 1
    for i in range(2, num + 1):
        res *= i
    return res


def hermite(mode):
    """Returns an orthonormal Hermite polynomial"""
    def func(x_coord):
        norm = 1. / (numpy.pi ** 0.25) / numpy.sqrt(float(factorial(mode) * 2 ** mode))
        return Hermite([0] * mode + [1])(x_coord) * norm

    return func


def h_roots(order):
    """
    Recursive root finding algorithm, taken from Numerical Recipes.
    More accurate than the standard h_roots() from scipy.
    """

    eps = 1.0e-14
    pim4 = numpy.pi ** (-0.25)
    max_iter = 20 # Maximum iterations.

    roots = numpy.empty(order)
    weights = numpy.empty(order)

    # Initial guess for the largest root
    curr_root = numpy.sqrt(2 * order + 1) - 1.85575 * (2 * order + 1) ** (-0.16667)

    for i in range((order + 1) // 2):

        # Initial guesses for the following roots
        if i == 1:
            curr_root -= 1.14 * order ** 0.426 / curr_root
        elif i == 2:
            curr_root = 1.86 * curr_root + 0.86 * roots[0]
        elif i == 3:
            curr_root = 1.91 * curr_root + 0.91 * roots[1]
        elif i > 3:
            curr_root = 2.0 * curr_root + roots[i - 2]

        # Refinement by Newton's method
        for _ in range(max_iter):
            pval = pim4
            pval_prev = 0.0

            # Recurrence relation to get the Hermite polynomial evaluated at ``curr_root``
            for j in range(order):
                pval, pval_prev = (curr_root * numpy.sqrt(2.0 / (j + 1)) * pval -
                    numpy.sqrt(float(j) / (j + 1)) * pval_prev), pval

            # ``pval`` is now the desired Hermite polynomial
            # We next compute ``pderiv``, its derivative, using ``pval_prev``, the polynomial
            # of one lower order.
            pderiv = numpy.sqrt(2 * order) * pval_prev
            prev_root = curr_root
            curr_root -= pval / pderiv
            if abs(curr_root - prev_root) <= eps:
                break
        else:
            raise Exception("Too many iterations")

        roots[order - 1 - i] = curr_root
        roots[i] = -curr_root
        weights[i] = 2.0 / (pderiv ** 2)
        weights[order - 1 - i] = weights[i]

    return roots, weights


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
    polynomial = hermite(mode)
    return lambda x_coord: polynomial(x_coord) * numpy.exp(-(x_coord ** 2) / 2)


def get_transformation_matrix(modes, order, add_points):
    """
    Returns the the matrix of values of mode functions taken at
    points of the spatial grid.
    """
    x_coords = get_spatial_grid(modes, order, add_points=add_points)

    res = numpy.zeros((modes, x_coords.size))

    for mode in range(modes):
        res[mode, :] = harmonic(mode)(x_coords)

    return res


class DHT(Computation):
    r"""
    Bases: :py:class:`~reikna.core.Computation`

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

    :param mode_arr: an array-like object defining the shape of mode space.
        If ``inverse=False``, its shape is used to define the mode space size.
    :param inverse: ``False`` for forward (coordinate space -> mode space) transform,
        ``True`` for inverse (mode space -> coordinate space) transform.
    :param axes: a tuple with axes over which to perform the transform.
        If not given, the transform is performed over all the axes.
    :param order: if ``F`` is a function in mode space, the number of spatial points
        is chosen so that the transformation ``DHT[(DHT^{-1}[F])^order]`` could be performed.
    :param add_points: a list of the same length as ``mode_arr`` shape,
        specifying the number of points in x-space to use in addition to minimally required
        (``0`` by default).

    .. py:method:: compiled_signature_forward(modes:o, coords:i)
    .. py:method:: compiled_signature_inverse(coords:o, modes:i)

        Depending on ``inverse`` value, either of these two will be created.

        :param modes: an array with the attributes of ``mode_arr``.
        :param coords: an array with the shape depending on ``mode_arr``, ``axes``, ``order``
            and ``add_points``, and the dtype of ``mode_arr``.
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

        self._inverse = inverse
        self._order = order

        if not inverse:
            parameters = [
                Parameter('modes', Annotation(mode_arr, 'o')),
                Parameter('coords', Annotation(coord_arr, 'i'))]
        else:
            parameters = [
                Parameter('coords', Annotation(coord_arr, 'o')),
                Parameter('modes', Annotation(mode_arr, 'i'))]

        Computation.__init__(self, parameters)

    def _get_transformation_matrix(self, dtype, modes, add_points):
        p_matrix = get_transformation_matrix(modes, self._order, add_points)
        p_matrix = p_matrix.astype(dtype)

        if not self._inverse:
            weights = get_spatial_weights(modes, self._order, add_points)
            tiled_weights = numpy.tile(
                weights.reshape(weights.size, 1).astype(dtype),
                (1, modes))
            p_matrix = p_matrix.transpose() * tiled_weights

        return p_matrix

    def _add_transpose(self, plan, current_mem, current_axes, axis):
        """
        Transpose the current array so that the ``axis`` is in the end of axes list.
        """

        seq_axes = list(range(len(current_axes)))

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

        return current_mem, current_axes

    def _build_plan(self, plan_factory, _device_params, output_arr, input_arr):

        plan = plan_factory()

        dtype = input_arr.dtype
        p_dtype = dtypes.real_for(dtype) if dtypes.is_complex(dtype) else dtype

        mode_shape = input_arr.shape if self._inverse else output_arr.shape

        current_mem = input_arr
        seq_axes = list(range(len(input_arr.shape)))
        current_axes = list(range(len(input_arr.shape)))

        for i, axis in enumerate(self._axes):
            current_mem, current_axes = self._add_transpose(plan, current_mem, current_axes, axis)

            tr_matrix = plan.persistent_array(
                self._get_transformation_matrix(p_dtype, mode_shape[axis], self._add_points[axis]))

            dot = MatrixMul(current_mem, tr_matrix)
            if i == len(self._axes) - 1 and current_axes == seq_axes:
                dot_output = output_arr
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
            transpose = Transpose(current_mem, output_arr_t=output_arr, axes=tr_axes)
            plan.add_computation(transpose, output_arr, current_mem)

        return plan
