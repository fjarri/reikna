import sys
from collections import defaultdict
import itertools

from reikna.cluda import OutOfResourcesError
from reikna.cluda.kernel import render_template
from reikna.helpers import product, log2, min_blocks, template_for, wrap_in_tuple, factors

TEMPLATE = template_for(__file__)


# For effectiveness' sake, since we have quite a lot of range() usage here,
# and Py2's range() that returns list will be too slow.
_range = xrange if sys.version_info[0] < 3 else range


class PrimeFactors:
    """
    Contains a natural number's decomposition into prime factors.
    """

    def __init__(self, factors):
        self.factors = factors

    @classmethod
    def decompose(cls, num):
        factors = defaultdict(lambda: 0)
        if num == 1:
            return cls(dict(factors))

        while num > 1:
            for i in _range(2, int(round(num ** 0.5)) + 1):
                if num % i == 0:
                    factors[i] += 1
                    num //= i
                    break
            else:
                factors[num] += 1
                num = 1

        return cls(dict(factors))

    def get_value(self):
        res = 1
        for pwr, exp in self.factors.items():
            res *= pwr ** exp
        return res

    def get_arrays(self):
        return self.factors.keys(), self.factors.values()

    def div_by(self, other):
        factors = dict(self.factors)
        for o_pwr, o_exp in other.factors.items():
            factors[o_pwr] -= o_exp
            if factors[o_pwr] == 0:
                del factors[o_pwr]
        return PrimeFactors(factors)


def _get_decompositions(num_factors, parts):
    """
    Helper recursive function for ``get_decompositions()``.
    Iterates over all possible decompositions of ``num_factors`` (of type ``PrimeFactors``)
    into ``parts`` factors.
    """
    if parts == 1:
        yield (num_factors.get_value(),)
        return

    powers, exponents = num_factors.get_arrays()
    for sub_exps in itertools.product(*[_range(exp, -1, -1) for exp in exponents]):
        part_factors = PrimeFactors(
            dict(((pwr,sub_exp) for pwr, sub_exp in zip(powers, sub_exps) if sub_exp > 0)))
        part = part_factors.get_value()
        remainder = num_factors.div_by(part_factors)
        for decomp in _get_decompositions(remainder, parts - 1):
            yield (part,) + decomp


def get_decompositions(num, parts):
    """
    Iterates overall possible decompositions of ``num`` into ``parts`` factors.
    """
    num_factors = PrimeFactors.decompose(num)
    return _get_decompositions(num_factors, parts)


def find_local_size(global_size, flat_local_size, threshold=0.05):
    """
    Returns a tuple of the same size as ``global_size``,
    with the product equal to ``flat_local_size``,
    and minimal difference between ``product(global_size)``
    and ``product(min_blocks(gs, ls) for gs, ls in zip(global_size, local_size))``
    (i.e. tries to minimize the amount of empty threads).
    """
    flat_global_size = product(global_size)
    if flat_local_size >= flat_global_size:
        return global_size

    threads_num = product(global_size)

    best_ratio = None
    best_local_size = None

    for local_size in get_decompositions(flat_local_size, len(global_size)):
        bounding_global_size = tuple(
            ls * min_blocks(gs, ls) for gs, ls in zip(global_size, local_size))
        empty_threads = product(bounding_global_size) - threads_num
        ratio = float(empty_threads) / threads_num

        # Stopping iteration early, because there may be a lot of elements to iterate over,
        # and we do not need the perfect solution.
        if ratio < threshold:
            return local_size

        if best_ratio is None or ratio < best_ratio:
            best_ratio = ratio
            best_local_size = local_size

    # This looks like the above loop can finish without setting `best_local_size`,
    # but providing flat_local_size <= product(global_size),
    # there is at least one decomposition (flat_local_size, 1, 1, ...).

    return best_local_size


def _group_dimensions(vdim, virtual_shape, adim, available_shape):
    """
    ``vdim`` and ``adim`` are used for the absolute addressing of dimensions during recursive calls.
    """
    if len(virtual_shape) == 0:
        return [], []

    vdim_group = 1 # number of currently grouped virtual dimensions
    adim_group = 1 # number of currently grouped available dimensions

    while 1:
        # If we have more elements in the virtual group than there is in the available group,
        # extend the available group by one dimension.
        if product(virtual_shape[:vdim_group]) > product(available_shape[:adim_group]):
            adim_group += 1
            continue

        # If the remaining available dimensions cannot accommodate the remaining virtual dimensions,
        # we try to fit one more virtual dimension in the virtual group.
        if product(virtual_shape[vdim_group:]) > product(available_shape[adim_group:]):
            vdim_group += 1
            continue

        # If we are here, it means that:
        # 1) the current available group can accommodate the current virtual group;
        # 2) the remaining available dimensions can accommodate the remaining virtual dimensions.
        # This means we can make a recursive call now.

        # Attach any following trivial virtual dimensions (of size 1) to this group
        # This will help to avoid unassigned trivial dimensions with no real dimensions left.
        while vdim_group < len(virtual_shape) and virtual_shape[vdim_group] == 1:
            vdim_group += 1

        v_res = tuple(range(vdim, vdim + vdim_group))
        a_res = tuple(range(adim, adim + adim_group))

        v_remainder, a_remainder = _group_dimensions(
            vdim + vdim_group, virtual_shape[vdim_group:],
            adim + adim_group, available_shape[adim_group:])
        return [v_res] + v_remainder, [a_res] + a_remainder


def group_dimensions(virtual_shape, available_shape):
    """
    Returns two lists, one of tuples with numbers of grouped virtual dimensions, the other
    one of tuples with numbers of corresponding group of available dimensions,
    such that for any group of virtual dimensions, the total number of elements they cover
    does not exceed the number of elements covered by the
    corresponding group of available dimensions.
    """
    assert product(virtual_shape) <= product(available_shape)
    return _group_dimensions(0, virtual_shape, 0, available_shape)


def ceiling_root(num, pwr):
    """
    Returns the integer ``num ** (1. / pwr)`` if num is a perfect square/cube/etc,
    or the integer ceiling of this value, if it's not.
    """
    res = num ** (1. / pwr)
    int_res = int(round(res))
    if int_res ** pwr == num:
        return int_res
    else:
        return int(res + 1)


def find_bounding_shape(virtual_size, available_shape):
    """
    Finds a tuple of the same length as ``available_shape``, with every element
    not greater than the corresponding element of ``available_shape``,
    and product not lower than ``virtual_size``.
    """
    assert virtual_size <= product(available_shape)

    free_size = virtual_size
    free_dims = set(range(len(available_shape)))
    bounding_shape = [None] * len(available_shape)

    while len(free_dims) > 0:
        guess = ceiling_root(free_size, len(free_dims))
        for fdim in free_dims:
            bounding_shape[fdim] = guess

        for fdim in free_dims:
            if bounding_shape[fdim] > available_shape[fdim]:
                bounding_shape[fdim] = available_shape[fdim]
                free_dims.remove(fdim)
                free_size = min_blocks(free_size, bounding_shape[fdim])
                break
        else:
            return tuple(bounding_shape)

    return tuple(available_shape)


class ShapeGroups:

    def __init__(self, virtual_shape, available_shape):
        self.real_dims = {}
        self.real_strides = {}
        self.virtual_strides = {}
        self.major_vdims = {}
        self.bounding_shape = tuple()
        self.skip_thresholds = []

        v_groups, a_groups = group_dimensions(virtual_shape, available_shape)

        for v_group, a_group in zip(v_groups, a_groups):
            virtual_subshape = virtual_shape[v_group[0]:v_group[-1]+1]
            virtual_subsize = product(virtual_subshape)

            bounding_subshape = find_bounding_shape(
                virtual_subsize,
                available_shape[a_group[0]:a_group[-1]+1])

            self.bounding_shape += bounding_subshape

            if virtual_subsize < product(bounding_subshape):
                strides = [(adim, product(bounding_subshape[:i])) for i, adim in enumerate(a_group)]
                self.skip_thresholds.append((virtual_subsize, strides))

            for vdim in v_group:
                self.real_dims[vdim] = a_group
                self.real_strides[vdim] = tuple(
                    product(self.bounding_shape[a_group[0]:adim]) for adim in a_group)
                self.virtual_strides[vdim] = product(virtual_shape[v_group[0]:vdim])

                # The major virtual dimension (the one that does not require
                # modulus operation when extracting its index from the flat index)
                # is the last non-trivial one (not of size 1).
                # Modulus will not be optimized away by the compiler,
                # but we know that all threads outside of the virtual group will be
                # filtered out by VIRTUAL_SKIP_THREADS.
                for major_vdim in _range(len(v_group) - 1, -1, -1):
                    if virtual_shape[v_group[major_vdim]] > 1:
                        break

                self.major_vdims[vdim] = v_group[major_vdim]


class VirtualSizes:

    def __init__(self, device_params, virtual_global_size,
            virtual_local_size=None, max_local_size=None):

        virtual_global_size = wrap_in_tuple(virtual_global_size)
        if virtual_local_size is not None:
            virtual_local_size = wrap_in_tuple(virtual_local_size)
            if len(virtual_local_size) != len(virtual_global_size):
                raise ValueError("Global size and local size must have the same dimensions")

        # Since the device uses column-major ordering of sizes, while we get
        # row-major ordered shapes, we temporarily invert our shapes
        # to facilitate internal handling.
        virtual_global_size = tuple(reversed(virtual_global_size))
        if virtual_local_size is not None:
            virtual_local_size = tuple(reversed(virtual_local_size))

        # Restrict local sizes using the provided explicit limit
        if max_local_size is not None:
            max_work_group_size = min(
                max_local_size,
                device_params.max_work_group_size,
                product(device_params.max_work_item_sizes))
            max_work_item_sizes = [
                min(max_local_size, mwis) for mwis in device_params.max_work_item_sizes]
        else:
            # Assuming:
            # 1) max_work_group_size <= product(max_work_item_sizes)
            # 2) max(max_work_item_sizes) <= max_work_group_size
            max_work_group_size = device_params.max_work_group_size
            max_work_item_sizes = device_params.max_work_item_sizes

        if virtual_local_size is None:
            # FIXME: we can obtain better results by taking occupancy into account here,
            # but for now we will assume that the more threads, the better.
            flat_global_size = product(virtual_global_size)
            multiple = device_params.warp_size

            if flat_global_size < max_work_group_size:
                flat_local_size = flat_global_size
            elif max_work_group_size < multiple:
                flat_local_size = 1
            else:
                flat_local_size = multiple * (max_work_group_size // multiple)

            # product(virtual_local_size) == flat_local_size <= max_work_group_size
            # Note: it's ok if local size elements are greater
            # than the corresponding global size elements as long as it minimizes the total
            # number of skipped threads.
            virtual_local_size = find_local_size(virtual_global_size, flat_local_size)
        else:
            if product(virtual_local_size) > max_work_group_size:
                raise OutOfResourcesError(
                    "Requested local size is greater than the maximum " + str(max_work_group_size))

        # Global and local sizes supported by CUDA or OpenCL restricted number of dimensions,
        # which may have limited size, so we need to pack our multidimensional sizes.

        virtual_grid_size = tuple(
            min_blocks(gs, ls) for gs, ls in zip(virtual_global_size, virtual_local_size))
        bounding_global_size = tuple(
            grs * ls for grs, ls in zip(virtual_grid_size, virtual_local_size))

        if product(virtual_grid_size) > product(device_params.max_num_groups):
            raise OutOfResourcesError(
                "Bounding global size " + repr(bounding_global_size) + " is too large")

        local_groups = ShapeGroups(virtual_local_size, max_work_item_sizes)
        grid_groups = ShapeGroups(virtual_grid_size, device_params.max_num_groups)

        # These can be different lenghts because of expansion into multiple dimensions
        # find_bounding_shape() does.
        real_local_size = tuple(local_groups.bounding_shape)
        real_grid_size = tuple(grid_groups.bounding_shape)

        diff = len(real_local_size) - len(real_grid_size)
        if diff > 0:
            self.real_local_size = real_local_size
            self.real_grid_size = real_grid_size + (1,) * abs(diff)
        else:
            self.real_local_size = real_local_size + (1,) * abs(diff)
            self.real_grid_size = real_grid_size

        self.real_global_size = tuple(
            gs * ls for gs, ls
            in zip(self.real_grid_size, self.real_local_size))

        # This function will be used to translate between internal column-major vdims
        # and user-supplied row-major vdims.
        vdim_inverse = lambda dim: len(virtual_local_size) - dim - 1

        self.vsize_functions = render_template(
            TEMPLATE,
            virtual_local_size=virtual_local_size,
            virtual_global_size=virtual_global_size,
            bounding_global_size=bounding_global_size,
            virtual_grid_size=virtual_grid_size,
            local_groups=local_groups,
            grid_groups=grid_groups,
            product=product,
            vdim_inverse=vdim_inverse)

        # Returning back to the row-major ordering
        self.virtual_local_size = tuple(reversed(virtual_local_size))
        self.virtual_global_size = tuple(reversed(virtual_global_size))
