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

    def __init__(self, arr, axes=None, exclusive=False, max_work_group_size_override=None):
        self._max_work_group_size_override = max_work_group_size_override
        self._exclusive = exclusive
        ndim = len(arr.shape)
        self._axes = helpers.normalize_axes(ndim, axes)
        if not helpers.are_axes_innermost(ndim, self._axes):
            self._transpose_to, self._transpose_from = (
                helpers.make_axes_innermost(ndim, self._axes))
            self._axes = tuple(range(ndim - len(self._axes), ndim))
        else:
            self._transpose_to = None
            self._transpose_from = None

        Computation.__init__(self, [
            Parameter('output', Annotation(arr, 'o')),
            Parameter('input', Annotation(arr, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):
        plan = plan_factory()

        if self._transpose_to is not None:

            transpose_to = Transpose(input_, axes=self._transpose_to)
            transposed = plan.temp_array_like(transpose_to.parameter.output)

            sub_scan = Scan(
                transposed, axes=self._axes, exclusive=self._exclusive,
                max_work_group_size_override=self._max_work_group_size_override)
            transposed_scanned = plan.temp_array_like(sub_scan.parameter.output)

            transpose_from = Transpose(transposed_scanned, axes=self._transpose_from)

            plan.computation_call(transpose_to, transposed, input_)
            plan.computation_call(sub_scan, transposed_scanned, transposed)
            plan.computation_call(transpose_from, output, transposed_scanned)

        else:

            if self._max_work_group_size_override is None:
                wg_size = device_params.max_work_group_size
            else:
                wg_size = self._max_work_group_size_override

            # Using algorithm cascading: sequential reduction, and then the parallel one.
            # According to Brent's theorem, the optimal sequential size is O(log(n)).
            # It seems like 4 is optimal (for the current kernel)
            seq_size = 4 # helpers.bounding_power_of_2(helpers.log2(wg_size))

            scan_ndim = len(self._axes) # assuming that at this point axes are inner and sorted
            batch_shape = output.shape[:-scan_ndim]
            batch_size = helpers.product(batch_shape)
            scan_shape = output.shape[-scan_ndim:]
            scan_size = helpers.product(scan_shape)

            wg_totals_size = helpers.min_blocks(scan_size, wg_size * seq_size)
            wg_totals = plan.temp_array((batch_size, wg_totals_size,), output.dtype)

            if wg_totals_size > 1:
                temp_output = plan.temp_array_like(output)
            else:
                temp_output = output

            last_part_size = scan_size % (wg_size * seq_size)
            if last_part_size == 0:
                last_part_size = wg_size * seq_size

            """
            print()
            print("global_size", (batch_size, wg_size * wg_totals_size))
            print("local_size", (1, wg_size))
            print("wg_size", wg_size)
            print("seq_size", seq_size)
            print("scan_size", scan_size)
            print("last_part_size", last_part_size)
            print("wg_totals_size", wg_totals_size)
            """
            plan.kernel_call(
                TEMPLATE.get_def('scan'),
                    [temp_output, input_, wg_totals],
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
                        log_wg_size=helpers.log2(wg_size)
                        ))

            if wg_totals_size > 1:
                sub_scan = Scan(
                    wg_totals, axes=(1,), exclusive=True,
                    max_work_group_size_override=self._max_work_group_size_override)
                scanned_wg_totals = plan.temp_array_like(wg_totals)
                plan.computation_call(sub_scan, scanned_wg_totals, wg_totals)

                plan.kernel_call(
                    TEMPLATE.get_def('add_wg_totals'),
                        [output, temp_output, scanned_wg_totals],
                        global_size=(batch_size, scan_size,),
                        render_kwds=dict(
                            slices=(len(batch_shape), len(scan_shape),),
                            wg_size=wg_size,
                            seq_size=seq_size,
                            ))

        return plan
