from typing import Any, Callable, Mapping, Type

import numpy
from grunnur import ArrayMetadata, AsArrayMetadata, DeviceParameters, Template
from numpy.typing import NDArray

import reikna.helpers as helpers
from reikna.core import Annotation, Computation, ComputationPlan, KernelArguments, Parameter

from .bijections import Bijection, philox
from .samplers import SAMPLERS, Sampler
from .tools import KeyGenerator

TEMPLATE = Template.from_associated_file(__file__)


class CBRNG(Computation):
    """
    Bases: :py:class:`~reikna.core.Computation`

    Counter-based pseudo-random number generator class.

    :param randoms_arr: an array intended for storing generated random numbers.
    :param generators_dim: the number of dimensions (counting from the end)
        which will use independent generators.
        For example, if ``randoms_arr`` has the shape ``(100, 200, 300)`` and
        ``generators_dim`` is ``2``, then in every sub-array ``(j, :, :)``,
        ``j = 0 .. 99``, every element will use an independent generator.
    :param sampler: a :py:class:`~reikna.cbrng.samplers.Sampler` object.
    :param seed: ``None`` for random seed, or an integer.

    .. py:classmethod:: sampler_name(randoms_arr, generators_dim, sampler_kwds=None, seed=None)

        A convenience constructor for the sampler ``sampler_name``
        from :py:mod:`~reikna.cbrng.samplers`.
        The contents of the dictionary ``sampler_kwds`` will be passed to the sampler constructor
        function (with ``bijection`` being created automatically,
        and ``dtype`` taken from ``randoms_arr``).

    .. py:method:: compiled_signature(counters:io, randoms:o)

        :param counters: the RNG "state".
            All attributes are equal to the ones of the result of :py:meth:`create_counters`.
        :param randoms: generated random numbers.
            All attributes are equal to the ones of ``randoms_arr`` from the constructor.
    """

    def __init__(
        self,
        randoms_arr_t: AsArrayMetadata,
        generators_dim: int,
        sampler: Sampler,
        seed: int | NDArray[numpy.uint32] | None = None,
    ):
        randoms_arr = randoms_arr_t.as_array_metadata()

        self._sampler = sampler
        self._keygen = KeyGenerator.create(sampler.bijection, seed=seed, reserve_id_space=True)

        assert sampler.dtype == randoms_arr.dtype

        counters_size = randoms_arr.shape[-generators_dim:]

        self._generators_dim = generators_dim
        self._counters_t = ArrayMetadata(dtype=sampler.bijection.counter_dtype, shape=counters_size)

        Computation.__init__(
            self,
            [
                Parameter("counters", Annotation(self._counters_t, "io")),
                Parameter("randoms", Annotation(randoms_arr, "o")),
            ],
        )

    def create_counters(self) -> numpy.ndarray[Any, numpy.dtype[Any]]:
        """
        Create a counter array for use in :py:class:`~reikna.cbrng.CBRNG`.
        """
        return numpy.zeros(self._counters_t.shape, self._counters_t.dtype)

    def _build_plan(
        self,
        plan_factory: Callable[[], ComputationPlan],
        _device_params: DeviceParameters,
        args: KernelArguments,
    ) -> ComputationPlan:
        plan = plan_factory()

        counters = args.counters
        randoms = args.randoms

        plan.kernel_call(
            TEMPLATE.get_def("cbrng"),
            [counters, randoms],
            kernel_name="kernel_cbrng",
            global_size=(helpers.product(counters.shape),),
            render_kwds=dict(
                sampler=self._sampler,
                keygen=self._keygen,
                batch=helpers.product(randoms.shape[: -self._generators_dim]),
                counters_slices=(self._generators_dim,),
                randoms_slices=(len(randoms.shape) - self._generators_dim, self._generators_dim),
            ),
        )

        return plan


# For some reason, closure did not work correctly.
# This class encapsulates the context and provides a classmethod for a given sampler.
class _ConvenienceCtr:
    def __init__(self, sampler_name: str):
        self._sampler_func = SAMPLERS[sampler_name]

    def __call__(
        self,
        cls: Type[CBRNG],
        randoms_arr_t: AsArrayMetadata,
        generators_dim: int,
        sampler_kwds: Mapping[str, Any] = {},
        seed: int | NDArray[numpy.uint32] | None = None,
    ) -> CBRNG:
        bijection = philox(64, 4)
        randoms_arr = randoms_arr_t.as_array_metadata()
        # TODO: hard to type kwargs; we really should be just using explicit methods.
        sampler = self._sampler_func(bijection, randoms_arr.dtype, **sampler_kwds)  # type: ignore[operator]
        return cls(randoms_arr, generators_dim, sampler, seed=seed)


# TODO: make it into explicit methods
# Add convenience constructors to CBRNG
for name in SAMPLERS:
    ctr = _ConvenienceCtr(name)
    setattr(CBRNG, name, classmethod(ctr))
