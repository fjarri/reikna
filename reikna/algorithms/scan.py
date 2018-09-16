import numpy

from reikna.cluda import ocl_api

from reikna.cluda import Snippet
import reikna.helpers as helpers
from reikna.cluda import dtypes
from reikna.cluda import OutOfResourcesError
from reikna.core import Computation, Parameter, Annotation, Type
from reikna.algorithms import Transpose

TEMPLATE = helpers.template_for(__file__)


class Scan(Computation):
    """
    Bases: :py:class:`~reikna.core.Computation`

    Scans the array over given axis using given binary operation.
    Namely, from an array ``[a, b, c, d, ...]`` and an operation ``.``,
    produces ``[a, a.b, a.b.c, a.b.c.d, ...]`` if ``exclusive`` is ``False``
    and ``[0, a, a.b, a.b.c, ...]`` if ``exclusive`` is ``True``
    (here ``0`` is the operation's identity element).

    :param arr_t: an array-like defining the initial array.
    :param predicate: a :py:class:`~reikna.algorithms.Predicate` object.
    :param axes: a list of non-repeating axes to scan over.
        (Note that the result will depend on the order of the axes).
        If ``None``, the whole array will be scanned over.
        This means that the selected axes will be flattened (in the specified order)
        and treated like a single axis for the purposes of the scan.
    :param exclusive: whether to perform an exclusive scan (see above).
    :param max_work_group_size: the maximum workgroup size to be used for the scan kernel.
    :param seq_size: the number of elements to be scanned sequentially.
        If not given, Reikna will attempt to choose the one resulting in the best performance,
        but sometimes a manual choice may be better.

    .. py:method:: compiled_signature(output:o, input:i)

        :param input: an array with the attributes of ``arr_t``.
        :param output: an array with the attributes of ``arr_t``.
    """

    def __init__(
            self, arr_t, predicate, axes=None, exclusive=False, max_work_group_size=None,
            seq_size=None):

        self._max_work_group_size = max_work_group_size
        self._seq_size = seq_size
        self._exclusive = exclusive
        ndim = len(arr_t.shape)
        self._axes = helpers.normalize_axes(ndim, axes)
        if not helpers.are_axes_innermost(ndim, self._axes):
            self._transpose_to, self._transpose_from = (
                helpers.make_axes_innermost(ndim, self._axes))
            self._axes = tuple(range(ndim - len(self._axes), ndim))
        else:
            self._transpose_to = None
            self._transpose_from = None

        if len(set(self._axes)) != len(self._axes):
            raise ValueError("Cannot scan twice over the same axis")

        if hasattr(predicate.empty, 'dtype'):
            if arr_t.dtype != predicate.empty.dtype:
                raise ValueError("The predicate and the array must use the same data type")
            empty = predicate.empty
        else:
            empty = dtypes.cast(arr_t.dtype)(predicate.empty)

        self._predicate = predicate

        Computation.__init__(self, [
            Parameter('output', Annotation(arr_t, 'o')),
            Parameter('input', Annotation(arr_t, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):
        plan = plan_factory()

        if self._transpose_to is not None:

            transpose_to = Transpose(input_, axes=self._transpose_to)
            transposed = plan.temp_array_like(transpose_to.parameter.output)

            sub_scan = Scan(
                transposed, self._predicate, axes=self._axes, exclusive=self._exclusive,
                max_work_group_size=self._max_work_group_size)
            transposed_scanned = plan.temp_array_like(sub_scan.parameter.output)

            transpose_from = Transpose(
                transposed_scanned, axes=self._transpose_from, output_arr_t=output)

            plan.computation_call(transpose_to, transposed, input_)
            plan.computation_call(sub_scan, transposed_scanned, transposed)
            plan.computation_call(transpose_from, output, transposed_scanned)

        else:

            scan_ndim = len(self._axes) # assuming that at this point axes are inner and sorted
            batch_shape = output.shape[:-scan_ndim]
            batch_size = helpers.product(batch_shape)
            scan_shape = output.shape[-scan_ndim:]
            scan_size = helpers.product(scan_shape)

            if self._max_work_group_size is None:
                max_wg_size = device_params.max_work_group_size
            else:
                max_wg_size = self._max_work_group_size

            # The current algorithm requires workgroup size to be a power of 2.
            assert max_wg_size == 2**helpers.log2(max_wg_size)

            # Using algorithm cascading: sequential reduction, and then the parallel one.
            # According to Brent's theorem, the optimal sequential size is O(log(n)).
            # So, ideally we want the minimum `wg_size` for which
            # `wg_size * log2(wg_size) >= scan_size`.
            if self._seq_size is None:
                wg_size = 2
                while wg_size < max_wg_size:
                    seq_size = helpers.bounding_power_of_2(helpers.log2(wg_size) - 1)
                    if wg_size * seq_size >= scan_size:
                        break
                    wg_size *= 2
            else:
                seq_size = self._seq_size
                wg_size = helpers.bounding_power_of_2(helpers.min_blocks(scan_size, seq_size))
                if wg_size > max_wg_size:
                    raise ValueError(
                        "Sequential size " + str(seq_size)
                        + " cannot be set because of the maximum workgroup size " + max_wg_size)

            wg_totals_size = helpers.min_blocks(scan_size, wg_size * seq_size)
            wg_totals = plan.temp_array((batch_size, wg_totals_size,), output.dtype)

            if wg_totals_size > 1:
                temp_output = plan.temp_array_like(output)
            else:
                temp_output = output

            last_part_size = scan_size % (wg_size * seq_size)
            if last_part_size == 0:
                last_part_size = wg_size * seq_size

            plan.kernel_call(
                TEMPLATE.get_def('scan'),
                    [temp_output, input_, wg_totals],
                    kernel_name="kernel_scan_wg",
                    global_size=(batch_size, wg_size * wg_totals_size),
                    local_size=(1, wg_size),
                    render_kwds=dict(
                        slices=(len(batch_shape), len(scan_shape)),
                        log_num_banks=helpers.log2(device_params.local_mem_banks),
                        exclusive=self._exclusive,
                        wg_size=wg_size,
                        seq_size=seq_size,
                        scan_size=scan_size,
                        last_part_size=last_part_size,
                        wg_totals_size=wg_totals_size,
                        log_wg_size=helpers.log2(wg_size),
                        predicate=self._predicate
                        ))

            if wg_totals_size > 1:
                sub_scan = Scan(
                    wg_totals, self._predicate, axes=(1,), exclusive=True,
                    max_work_group_size=self._max_work_group_size)
                scanned_wg_totals = plan.temp_array_like(wg_totals)
                plan.computation_call(sub_scan, scanned_wg_totals, wg_totals)

                plan.kernel_call(
                    TEMPLATE.get_def('add_wg_totals'),
                        [output, temp_output, scanned_wg_totals],
                        kernel_name="kernel_scan_add_wg_totals",
                        global_size=(batch_size, scan_size,),
                        render_kwds=dict(
                            slices=(len(batch_shape), len(scan_shape),),
                            wg_size=wg_size,
                            seq_size=seq_size,
                            ))

        return plan
