from reikna.cluda import Snippet
import reikna.helpers as helpers
from reikna.core import Computation


class PureParallel(Computation):
    """
    Bases: :py:class:`~reikna.core.Computation`

    A general class for pure parallel computations
    (i.e. with no interaction between threads).

    :param parameters: a list of :py:class:`~reikna.core.Parameter` objects.
    :param code: a source code for a template def.
    :param guiding_array: an tuple with the array shape, or the name of one of ``parameters``.
        By default, the first parameter is chosen.
    :param render_kwds: a dictionary with render keywords for the ``code``.

    .. py:function:: compiled_signature(*args)

        :param args: corresponds to the given ``parameters``.
    """

    def __init__(self, parameters, code, guiding_array=None, render_kwds=None):

        Computation.__init__(self, parameters)
        self._root_parameters = list(self.signature.parameters.keys())
        self._snippet = Snippet(helpers.template_def(
            ['idxs'] + self._root_parameters, code), render_kwds=render_kwds)

        if guiding_array is None:
            guiding_array = self._root_parameters[0]

        if isinstance(guiding_array, str):
            self._guiding_shape = self.signature.parameters[guiding_array].annotation.type.shape
        else:
            self._guiding_shape = guiding_array

    def _build_plan(self, plan_factory, _device_params, *args):

        plan = plan_factory()

        argnames = [arg.name for arg in args]
        arglist = ", ".join(argnames)
        idx_names = ["_idx" + str(i) for i in range(len(self._guiding_shape))]

        template = helpers.template_def(
            ['kernel_declaration'] + argnames,
            """
            ${kernel_declaration}
            {
                VIRTUAL_SKIP_THREADS;
                VSIZE_T _flat_idx = virtual_global_id(0);

                %for i, idx_name in enumerate(idx_names):
                <%
                    stride = product(shape[i+1:])
                %>
                VSIZE_T ${idx_name} = _flat_idx / ${stride};
                _flat_idx -= ${idx_name} * ${stride};
                %endfor

                ${snippet(idx_names, """ + arglist + """)}
            }
            """)

        plan.kernel_call(
            template, args,
            global_size=helpers.product(self._guiding_shape),
            render_kwds=dict(
                shape=self._guiding_shape,
                idx_names=idx_names,
                product=helpers.product,
                snippet=self._snippet))

        return plan
