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

        root_parameters = list(self.signature.parameters.keys())

        if isinstance(code, Snippet):
            self._snippet = code
        else:
            self._snippet = Snippet(helpers.template_def(
                ['idxs'] + root_parameters, code), render_kwds=render_kwds)

        if guiding_array is None:
            guiding_array = root_parameters[0]

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
            guiding_array = trf.signature.parameters.keys()[0]

        if isinstance(guiding_array, TransformationParameter):
            if not guiding_array.belongs_to(trf):
                raise ValueError(
                    "The transformation parameter must belong to the provided transformation")
            guiding_array = str(guiding_array)

        guiding_param = trf.signature.parameters[guiding_array]
        if not guiding_param.annotation.array:
            raise ValueError("The parameter serving as a guiding array cannot be a scalar")

        res = cls(
            [Parameter(guiding_param.name, Annotation(guiding_param.annotation.type, 'io'))],
            Snippet.create(
                lambda idxs, arr:
                """
                ${arr.store_idx}(${idxs.all()}, ${arr.load_idx}(${idxs.all()}));
                """),
            guiding_array=guiding_param.name)

        # Relying on the order-preserving properties of connect().
        # Namely, since the guiding parameter is an i/o one, either output or input part of it
        # will remain in the signature after the connection,
        # so the connection will preserve its place relative to the other parameters
        # of the transformation.
        connection_kwds = {
            name:name for name in trf.signature.parameters
            if name != guiding_param.name}
        res.connect(
            guiding_param.name, trf, guiding_param.name,
            **connection_kwds)

        return res

    def _build_plan(self, plan_factory, _device_params, *args):

        plan = plan_factory()

        argnames = [arg.name for arg in args]
        arglist = ", ".join(argnames)
        idxs = Indices(self._guiding_shape)

        template = helpers.template_def(
            ['kernel_declaration'] + argnames,
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
            global_size=self._guiding_shape,
            render_kwds=dict(
                shape=self._guiding_shape,
                idxs=idxs,
                product=helpers.product,
                snippet=self._snippet))

        return plan
