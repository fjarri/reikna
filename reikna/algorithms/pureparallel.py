from reikna.cluda import Snippet
import reikna.helpers as helpers
from reikna.core import Computation, Indices, Parameter, Annotation
from reikna.core.transformation import TransformationParameter


class PureParallel(Computation):
    """
    Bases: :py:class:`~reikna.core.Computation`

    A general class for pure parallel computations
    (i.e. with no interaction between threads).

    :param parameters: a list of :py:class:`~reikna.core.Parameter` objects.
    :param code: a source code for the computation.
        Can be a :py:class:`~reikna.cluda.Snippet` object which will be passed
        :py:class:`~reikna.core.Indices` object for the ``guiding_array`` as the first
        positional argument, and :py:class:`~reikna.core.transformation.KernelParameter` objects
        corresponding to ``parameters`` as the rest of positional arguments.
        If it is a string, such :py:class:`~reikna.cluda.Snippet` will be created out of it,
        with the parameter names ``idxs`` for the first one and the names of ``parameters``
        for the remaining ones.

    :param guiding_array: an tuple with the array shape, or the name of one of ``parameters``.
        By default, the first parameter is chosen.
    :param render_kwds: a dictionary with render keywords for the ``code``.

    .. py:function:: compiled_signature(*args)

        :param args: corresponds to the given ``parameters``.
    """

    def __init__(self, parameters, code, guiding_array=None, render_kwds=None):

        Computation.__init__(self, parameters)

        self._root_parameters = list(self.signature.parameters.keys())

        if isinstance(code, Snippet):
            self._snippet = code
        else:
            self._snippet = Snippet(helpers.template_def(
                ['idxs'] + self._root_parameters, code), render_kwds=render_kwds)

        if guiding_array is None:
            guiding_array = self._root_parameters[0]

        if isinstance(guiding_array, str):
            self._guiding_shape = self.signature.parameters[guiding_array].annotation.type.shape
        else:
            self._guiding_shape = guiding_array

    @classmethod
    def from_trf(cls, trf, guiding_array=None):
        """
        Creates a ``PureParallel`` instance from a :py:class:`~reikna.core.Transformation` object.
        ``guiding_array`` can be a string with a name of an array parameter from ``trf``,
        or the corresponding :py:class:`~reikna.core.transformation.TransformationParameter` object.
        """

        if guiding_array is None:
            guiding_array = list(trf.signature.parameters.keys())[0]

        if isinstance(guiding_array, TransformationParameter):
            if not guiding_array.belongs_to(trf):
                raise ValueError(
                    "The transformation parameter must belong to the provided transformation")
            guiding_array = str(guiding_array)

        guiding_param = trf.signature.parameters[guiding_array]
        if not guiding_param.annotation.array:
            raise ValueError("The parameter serving as a guiding array cannot be a scalar")

        # Transformation snippet is the same as required for PureParallel
        # In particular, it has arguments like "idxs, arg1, arg2, ...",
        # and load_same()/store_same() in the snippet will work because the PureParallel
        # computation defines index variables in its kernel (see _build_plan()).
        # This is a bit shady, but other variants are even worse
        # (e.g. creating a trivial computation and attaching this transformation to it
        # will either produce incorrect parameter order, or will require the usage of an
        # 'io' parameter, which has its own complications).
        # FIXME: find a solution which does not create an implicit dependence on
        # the way transformations are handled.
        res = cls(
            list(trf.signature.parameters.values()),
            trf.snippet,
            guiding_array=guiding_param.name)

        return res

    def _build_plan(self, plan_factory, _device_params, *args):

        plan = plan_factory()

        # Using root_parameters to avoid duplicated names
        # (can happen if this computation is nested and
        # the same arrays are passed to it as arguments)
        arglist = ", ".join(self._root_parameters)
        idxs = Indices(self._guiding_shape)

        template = helpers.template_def(
            ['kernel_declaration'] + self._root_parameters,
            """
            ${kernel_declaration}
            {
                VIRTUAL_SKIP_THREADS;

                %for i, idx in enumerate(idxs):
                VSIZE_T ${idx} = virtual_global_id(${i});
                %endfor

                ${snippet(idxs, """ + arglist + """)}
            }
            """)

        plan.kernel_call(
            template, args,
            kernel_name="kernel_pure_parallel",
            global_size=(1,) if len(self._guiding_shape) == 0 else self._guiding_shape,
            render_kwds=dict(
                idxs=idxs,
                product=helpers.product,
                snippet=self._snippet))

        return plan
