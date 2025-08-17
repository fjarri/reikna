import weakref
from collections import namedtuple
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy
from grunnur import (
    Array,
    ArrayMetadata,
    AsArrayMetadata,
    Buffer,
    Queue,
    StaticKernel,
    VirtualManager,
    cuda_api_id,
)

from ..helpers import Graph
from .signature import Annotation, Parameter, Signature, Type
from .transformation import TransformationParameter, TransformationTree

if TYPE_CHECKING:
    from grunnur import BoundDevice, DefTemplate, DeviceParameters

    from .transformation import Transformation


class ComputationParameter(Type):
    """
    Represents a typed computation parameter.
    Can be used as a substitute of an array for functions
    which are only interested in array metadata.
    """

    def __init__(self, computation: "ComputationCallable | Computation", name: str, type_: Type):
        """__init__()"""  # hide the signature from Sphinx
        super().__init__(type_._metadata, type_.ctype)  # noqa: SLF001
        self._computation = weakref.ref(computation)
        self._name = name

    def belongs_to(self, comp: "ComputationCallable | Computation") -> bool:
        return self._computation() is comp

    def connect(
        self,
        _trf: "Transformation",
        _tr_connector: TransformationParameter,
        **tr_from_comp: TransformationParameter,
    ) -> "Computation":
        """
        Shortcut for :py:meth:`~reikna.core.Computation.connect`
        with this parameter as a first argument.
        """
        return cast(Computation, self._computation()).connect(
            self._name, _trf, _tr_connector, **tr_from_comp
        )

    def __str__(self) -> str:
        return self._name


class Translator:
    """
    A callable that translates strings from the known list, and prefixes unknown ones.
    Used to introduce parameter names from a nested computation to the parent namespace.
    """

    def __init__(self, known_old: Sequence[str], known_new: Sequence[str], prefix: str):
        self._mapping = dict(zip(known_old, known_new, strict=False))
        self._prefix = prefix

    def __call__(self, name: str) -> str:
        if name in self._mapping:
            return self._mapping[name]
        return (self._prefix + "_" if self._prefix != "" else "") + name

    def get_nested(
        self, known_old: Sequence[str], known_new: Sequence[str], prefix: str
    ) -> "Translator":
        """Returns a new ``Translator`` with an extended prefix."""
        return Translator(known_old, known_new, prefix + self._prefix)

    @classmethod
    def identity(cls) -> "Translator":
        return cls([], [], "")


def check_external_parameter_name(name: str) -> None:
    """
    Checks that a user-supplied parameter name meets the special criteria.
    Basically, we do not want such names start with underscores
    to prevent them from conflicting with internal names.
    """
    # Raising errors here so we can provide better explanation for the user
    if name.startswith("_"):
        raise ValueError("External parameter name cannot start with the underscore.")


class ParameterContainer:
    """A convenience object with ``ComputationParameter`` attributes."""

    def __init__(
        self, parent: "ComputationCallable | Computation", parameters: Iterable[Parameter]
    ):
        self._param_objs = {
            param.name: ComputationParameter(parent, param.name, param.annotation.type)
            for param in parameters
        }

    def __getattr__(self, name: str) -> ComputationParameter:
        return self._param_objs[name]


class Computation:
    """
    A base class for computations, intended to be subclassed.

    :param root_parameters: a list of :py:class:`~reikna.core.Parameter` objects.

    .. py:attribute:: signature

        A :py:class:`~reikna.core.Signature` object representing current computation signature
        (taking into account connected transformations).

    .. py:attribute:: parameter

        A named tuple of :py:class:`~reikna.core.computation.ComputationParameter` objects
        corresponding to parameters from the current :py:attr:`signature`.
    """

    def __init__(self, root_parameters: Iterable[Parameter]):
        for param in root_parameters:
            check_external_parameter_name(param.name)

        # TODO: should this be stored in TransformationTree?
        self._original_parameters = list(root_parameters)

        self._tr_tree = TransformationTree(root_parameters)
        self._update_attributes()

    def _update_attributes(self) -> None:
        """
        Updates ``signature`` and ``parameter`` attributes.
        Called by the methods that change the signature.
        """
        leaf_params = self._tr_tree.get_leaf_parameters()
        self.signature = Signature(leaf_params)
        self.parameter = ParameterContainer(self, leaf_params)

    # The names are underscored to avoid name conflicts with ``tr_from_comp`` keys
    # (where the user can introduce new parameter names)
    def connect(
        self,
        _comp_connector: str | ComputationParameter,
        _trf: "Transformation",
        _tr_connector: TransformationParameter,
        **tr_from_comp: TransformationParameter,
    ) -> "Computation":
        """
        Connect a transformation to the computation.

        :param _comp_connector: connection target ---
            a :py:class:`~reikna.core.computation.ComputationParameter` object
            belonging to this computation object, or a string with its name.
        :param _trf: a :py:class:`~reikna.core.Transformation` object.
        :param _tr_connector: connector on the side of the transformation ---
            a :py:class:`~reikna.core.transformation.TransformationParameter` object
            belonging to ``tr``, or a string with its name.
        :param tr_from_comp: a dictionary with the names of new or old
            computation parameters as keys, and
            :py:class:`~reikna.core.transformation.TransformationParameter` objects
            (or their names) as values.
            The keys of ``tr_from_comp`` cannot include the name of the connection target.
        :returns: this computation object (modified).

        .. note::

            The resulting parameter order is determined by traversing
            the graph of connections depth-first (starting from the initial computation parameters),
            with the additional condition: the nodes do not change their order
            in the same branching level (i.e. in the list of computation or
            transformation parameters, both of which are ordered).

            For example, consider a computation with parameters ``(a, b, c, d)``.
            If you connect a transformation ``(a', c) -> a``, the resulting computation
            will have the signature ``(a', b, c, d)`` (as opposed to ``(a', c, b, d)``
            it would have for the pure depth-first traversal).
        """
        # Extract connector name
        if isinstance(_comp_connector, ComputationParameter) and not _comp_connector.belongs_to(
            self
        ):
            raise ValueError("The connection target must belong to this computation.")
        param_name = str(_comp_connector)

        # Extract transformation parameters names

        if param_name in tr_from_comp:
            raise ValueError(
                "Parameter '"
                + param_name
                + "' cannot be supplied "
                + "both as the main connector and one of the child connections"
            )

        tr_from_comp[param_name] = _tr_connector
        comp_from_tr = {}
        for comp_connection_name, tr_connection in tr_from_comp.items():
            check_external_parameter_name(comp_connection_name)
            if isinstance(tr_connection, TransformationParameter) and not tr_connection.belongs_to(
                _trf
            ):
                raise ValueError(
                    "The transformation parameter must belong to the provided transformation"
                )
            tr_connection_name = str(tr_connection)
            comp_from_tr[tr_connection_name] = comp_connection_name

        self._tr_tree.connect(param_name, _trf, comp_from_tr)
        self._update_attributes()
        return self

    def _translate_tree(self, translator: Translator) -> TransformationTree:
        return self._tr_tree.translate(translator)

    def _get_plan(
        self,
        tr_tree: TransformationTree,
        translator: Translator,
        bound_device: "BoundDevice",
        virtual_manager: VirtualManager | None,
        compiler_options: Iterable[str],
        *,
        fast_math: bool,
        keep: bool,
    ) -> "ComputationPlan":
        def plan_factory() -> ComputationPlan:
            return ComputationPlan(
                tr_tree,
                translator,
                bound_device,
                virtual_manager,
                compiler_options,
                fast_math=fast_math,
                keep=keep,
            )

        args = KernelArguments(
            {
                local_param.name: KernelArgument(param.name, param.annotation.type)
                for local_param, param in zip(
                    self._original_parameters, tr_tree.get_root_parameters(), strict=False
                )
            }
        )
        return self._build_plan(plan_factory, bound_device.params, args)

    def compile(
        self,
        bound_device: "BoundDevice",
        virtual_manager: VirtualManager | None = None,
        compiler_options: Iterable[str] = [],
        *,
        fast_math: bool = False,
        keep: bool = False,
    ) -> "ComputationCallable":
        """
        Compiles the computation with the given :py:class:`grunnur.BoundDevice` object
        and returns a :py:class:`~reikna.core.computation.ComputationCallable` object.
        If ``fast_math`` is enabled, the compilation of all kernels is performed using
        the compiler options for fast and imprecise mathematical functions.
        ``compiler_options`` can be used to pass a list of strings as arguments
        to the backend compiler.
        If ``keep`` is ``True``, the generated kernels and binaries will be preserved
        in temporary directories.
        """
        translator = Translator.identity()
        return self._get_plan(
            self._tr_tree,
            translator,
            bound_device,
            virtual_manager,
            compiler_options,
            fast_math=fast_math,
            keep=keep,
        ).finalize()

    def _build_plan(
        self,
        plan_factory: Callable[[], "ComputationPlan"],
        device_params: "DeviceParameters",
        args: "KernelArguments",
    ) -> "ComputationPlan":
        """
        Derived classes override this method.
        It is called by :py:meth:`compile` and
        supposed to return a :py:class:`~reikna.core.computation.ComputationPlan` object.

        :param plan_factory: a callable returning a new
            :py:class:`~reikna.core.computation.ComputationPlan` object.
        :param device_params: a :py:class:`grunnur.DeviceParameters` object corresponding
            to the thread the computation is being compiled for.
        :param args: :py:class:`~reikna.core.computation.KernelArgument` objects,
            corresponding to ``parameters`` specified during the creation
            of this computation object.
        """
        raise NotImplementedError


class KernelArguments:
    def __init__(self, args: "Mapping[str, KernelArgument]"):
        self._args = dict(args)

    def all(self) -> "Iterable[KernelArgument]":
        return self._args.values()

    def __getattr__(self, name: str) -> "KernelArgument":
        return self._args[name]


class IdGen:
    """Encapsulates a simple ID generator."""

    def __init__(self, prefix: str, counter: int = 0):
        self._counter = counter
        self._prefix = prefix

    def __call__(self) -> str:
        self._counter += 1
        return self._prefix + str(self._counter)


class KernelArgument(Type):
    """Represents an argument suitable to pass to planned kernel or computation call."""

    def __init__(self, name: str, type_: Type):
        """__init__()"""  # hide the signature from Sphinx
        super().__init__(type_._metadata, type_.ctype)  # noqa: SLF001
        self.name = name

    def __repr__(self) -> str:
        return "KernelArgument(" + self.name + ")"


class ComputationPlan:
    """Computation plan recorder."""

    def __init__(
        self,
        tr_tree: TransformationTree,
        translator: Translator,
        bound_device: "BoundDevice",
        virtual_manager: "VirtualManager | None",
        compiler_options: Iterable[str],
        *,
        fast_math: bool,
        keep: bool,
    ):
        """__init__()"""  # hide the signature from Sphinx
        self._bound_device = bound_device
        self._virtual_manager = virtual_manager
        self._is_cuda = bound_device.context.api.id == cuda_api_id()
        self._tr_tree = tr_tree
        self._translator = translator
        self._fast_math = fast_math
        self._compiler_options = compiler_options
        self._keep = keep

        self._nested_comp_idgen = IdGen("_nested")
        self._persistent_value_idgen = IdGen("_value")
        self._constant_value_idgen = IdGen("_constant")
        self._temp_array_idgen = IdGen("_temp")

        self._external_annotations = self._tr_tree.get_root_annotations()
        self._persistent_values: dict[str, Array | numpy.generic] = {}
        self._constant_arrays: dict[str, Array] = {}
        self._temp_arrays: set[str] = set()
        self._internal_annotations: dict[str, Annotation] = {}
        self._kernels: list[PlannedKernelCall] = []

    def _scalar(self, val: numpy.generic) -> KernelArgument:
        """
        Adds a persistent scalar to the plan, and returns the corresponding
        :py:class:`KernelArgument`.
        """
        name = self._translator(self._persistent_value_idgen())
        ann = Annotation(Type.scalar(val.dtype))
        self._internal_annotations[name] = ann
        # TODO: does `ann.type(val)` even do anything given that `val` is already a `numpy` object?
        self._persistent_values[name] = cast(numpy.generic, ann.type(val))
        return KernelArgument(name, ann.type)

    def persistent_array(self, arr: numpy.ndarray[Any, numpy.dtype[Any]]) -> KernelArgument:
        """
        Adds a persistent GPU array to the plan, and returns the corresponding
        :py:class:`KernelArgument`.
        """
        name = self._translator(self._persistent_value_idgen())
        ann = Annotation(arr, "i")
        self._internal_annotations[name] = ann
        self._persistent_values[name] = Array.from_host(self._bound_device, arr)
        return KernelArgument(name, ann.type)

    def temp_array(
        self,
        shape: Sequence[int] | int,
        dtype: numpy.dtype[Any],
        strides: Sequence[int] | None = None,
        offset: int = 0,
        nbytes: int | None = None,
    ) -> KernelArgument:
        """
        Adds a temporary GPU array to the plan, and returns the corresponding
        :py:class:`KernelArgument`.
        See :py:meth:`grunnur.Array` for the information about the parameters.

        Temporary arrays can share physical memory, but in such a way that
        their contents is guaranteed to persist between the first and the last use in a kernel
        during the execution of the plan.
        """
        name = self._translator(self._temp_array_idgen())
        ann = Annotation(
            Type.array(dtype, shape=shape, strides=strides, offset=offset, nbytes=nbytes), "io"
        )
        self._internal_annotations[name] = ann
        self._temp_arrays.add(name)
        return KernelArgument(name, ann.type)

    def constant_array(self, arr: numpy.ndarray[Any, numpy.dtype[Any]]) -> KernelArgument:
        """
        Adds a constant GPU array to the plan, and returns the corresponding
        :py:class:`KernelArgument`.
        """
        name = self._translator(self._constant_value_idgen())
        ann = Annotation(arr, constant=True)
        self._internal_annotations[name] = ann
        self._constant_arrays[name] = Array.from_host(self._bound_device, arr)
        return KernelArgument(name, ann.type)

    def temp_array_like(
        self, arr: AsArrayMetadata | numpy.ndarray[Any, numpy.dtype[Any]]
    ) -> KernelArgument:
        """
        Same as :py:meth:`temp_array`, taking the array properties
        from array or array-like object ``arr``.

        .. warning::

            Note that ``pycuda.GPUArray`` objects do not have the ``offset`` attribute.
        """
        metadata = ArrayMetadata.from_arraylike(arr)
        return self.temp_array(
            metadata.shape,
            metadata.dtype,
            strides=metadata.strides,
            offset=metadata.first_element_offset,
            nbytes=metadata.buffer_size,
        )

    def _get_annotation(self, name: str) -> Annotation:
        if name in self._external_annotations:
            return self._external_annotations[name]
        return self._internal_annotations[name]

    def _process_kernel_arguments(
        self, args: Iterable[KernelArgument | numpy.generic]
    ) -> tuple[list[Parameter], dict[str, Any]]:
        """
        Scan through kernel arguments passed by the user, check types,
        and wrap ad hoc values if necessary.

        Does not change the plan state.
        """
        processed_args = []
        adhoc_idgen = IdGen("_adhoc")
        adhoc_values = {}

        for arg in args:
            if not isinstance(arg, KernelArgument):
                if hasattr(arg, "shape") and hasattr(arg, "dtype"):
                    if len(arg.shape) > 0:
                        raise ValueError("Arrays are not allowed as ad hoc arguments")

                    # Not creating a new persistent scalar with _scalar(),
                    # because the kernel compilation may fail,
                    # in which case we would have to roll back the plan state.
                    # These arguments are local to this kernel anyway,
                    # so there's no need in registering them in the plan.
                    arg_name = self._translator(adhoc_idgen())
                    adhoc_values[arg_name] = arg
                    annotation = Annotation(Type.scalar(arg.dtype))
                else:
                    raise TypeError("Unknown argument type: " + str(type(arg)))
            else:
                annotation = self._get_annotation(arg.name)
                arg_name = arg.name

            processed_args.append(Parameter(arg_name, annotation))

        return processed_args, adhoc_values

    def _process_computation_arguments(
        self, signature: Signature, args: tuple[Any, ...], kwds: Mapping[str, Any]
    ) -> list[str]:
        """
        Scan through nested computation arguments passed by the user, check types,
        and create new persistent values for ad hoc arguments.

        Changes the plan state.
        """
        bound_args = signature.bind_with_defaults(args, kwds, cast=False)

        kargs = []
        for arg, param in zip(bound_args.args, signature.parameters.values(), strict=False):
            if not isinstance(arg, KernelArgument):
                if param.annotation.array:
                    raise ValueError("Ad hoc arguments are only allowed for scalar parameters")
                karg = self._scalar(param.annotation.type(arg))
            else:
                karg = arg

            annotation = self._get_annotation(karg.name)

            if not annotation.can_be_argument_for(param.annotation):
                raise TypeError(f"Got {annotation} for '{param.name}', expected {param.annotation}")

            kargs.append(karg)

        return [karg.name for karg in kargs]

    def kernel_call(
        self,
        template_def: "DefTemplate",
        args: Iterable[KernelArgument | numpy.generic],
        global_size: Sequence[int],
        local_size: Sequence[int] | None = None,
        render_kwds: Mapping[str, Any] = {},
        kernel_name: str = "_kernel_func",
    ) -> None:
        """
        Adds a kernel call to the plan.

        :param template_def: Mako template def for the kernel.
        :param args: a list consisting of
            :py:class:`~reikna.core.computation.KernelArgument` objects,
            or scalar values wrapped in ``numpy.ndarray``,
            that are going to be passed to the kernel during execution.
        :param global_size: global size to use for the call, in **row-major** order.
        :param local_size: local size to use for the call, in **row-major** order.
            If ``None``, the local size will be picked automatically.
        :param render_kwds: dictionary with additional values used to render the template.
        :param kernel_name: the name of the kernel function.
        """
        processed_args, adhoc_values = self._process_kernel_arguments(args)
        subtree = self._tr_tree.get_subtree(processed_args)

        kernel_declaration, kernel_leaf_names = subtree.get_kernel_declaration(
            kernel_name, skip_constants=self._is_cuda
        )
        kernel_argobjects = subtree.get_kernel_argobjects()

        render_kwds = dict(render_kwds)

        constant_arrays = {}
        if self._is_cuda:
            for karg in kernel_argobjects:
                if karg.name in self._constant_arrays:
                    constant_arrays[str(karg)] = self._constant_arrays[karg.name]

        kernel = StaticKernel(
            [self._bound_device],
            template_def,
            kernel_name,
            global_size,
            local_size=local_size,
            render_args=[kernel_declaration, *kernel_argobjects],
            render_globals=render_kwds,
            fast_math=self._fast_math,
            compiler_options=self._compiler_options,
            constant_arrays=constant_arrays,
            keep=self._keep,
        )

        if constant_arrays:
            queue = Queue(self._bound_device)
            for name, arr in constant_arrays.items():
                kernel.set_constant_array(queue, name, arr)
            queue.synchronize()

        self._kernels.append(PlannedKernelCall(kernel, kernel_leaf_names, adhoc_values))

    def computation_call(self, computation: Computation, *args: Any, **kwds: Any) -> None:
        """
        Adds a nested computation call.
        The ``computation`` value must be a :py:class:`~reikna.core.Computation` object.
        ``args`` and ``kwds`` are values to be passed to the computation.
        """
        signature = computation.signature
        argnames = self._process_computation_arguments(signature, args, kwds)

        # We want to preserve the Computation object for which we're calling _build_plan()
        # (because it may have attributes created by the constructor of the derived class),
        # but we need to translate its tree to integrate names of its nodes into
        # the parent namespace.
        translator = self._translator.get_nested(
            list(signature.parameters), argnames, self._nested_comp_idgen()
        )
        new_tree = computation._translate_tree(translator)  # noqa: SLF001
        new_tree.reconnect(self._tr_tree)

        self._append_plan(
            computation._get_plan(  # noqa: SLF001
                new_tree,
                translator,
                self._bound_device,
                self._virtual_manager,
                self._compiler_options,
                fast_math=self._fast_math,
                keep=self._keep,
            )
        )

    def _append_plan(self, plan: "ComputationPlan") -> None:
        self._kernels += plan._kernels  # noqa: SLF001
        if not self._is_cuda:
            # In case of CUDA each kernel manages its constant arrays itself,
            # no need to remember them.
            self._constant_arrays.update(plan._constant_arrays)  # noqa: SLF001
        self._persistent_values.update(plan._persistent_values)  # noqa: SLF001
        self._temp_arrays.update(plan._temp_arrays)  # noqa: SLF001
        self._internal_annotations.update(plan._internal_annotations)  # noqa: SLF001

    def finalize(self) -> "ComputationCallable":  # noqa: C901
        # We need to add inferred dependencies between temporary buffers.
        # Basically, we assume that if some buffer X was used first in kernel M
        # and last in kernel N, all buffers in kernels from M+1 till N-1 depend on it
        # (in other words, data in X has to persist from call M till call N)

        # Record first and last kernel when a certain temporary array was used.
        usage: dict[str, list[int]] = {}
        for i, kernel in enumerate(self._kernels):
            for argname in kernel.argnames:
                if argname not in self._temp_arrays:
                    continue
                if argname in usage:
                    usage[argname][1] = i
                else:
                    usage[argname] = [i, i]

        # Convert usage to dependencies.
        # Basically, if usage ranges of two arrays overlap, they are dependent.
        dependencies: Graph[str] = Graph()
        for name, pair in usage.items():
            start, end = pair
            for i in range(start, end + 1):
                for other_name in self._kernels[i].argnames:
                    if other_name not in self._temp_arrays or other_name == name:
                        continue
                    dependencies.add_edge(name, other_name)

        # Create a dictionary of internal values to be used by kernels.

        internal_args = dict(self._persistent_values)
        internal_args.update(self._constant_arrays)

        # Allocate buffers specifying the dependencies
        all_buffers = []
        for name in self._temp_arrays:
            dependent_buffers = [
                internal_args[dep] for dep in dependencies[name] if dep in internal_args
            ]
            type_ = self._internal_annotations[name].type

            allocator: Callable[[BoundDevice, int], Buffer]
            if self._virtual_manager:
                allocator = self._virtual_manager.allocator(dependent_buffers)
            else:
                allocator = Buffer.allocate

            new_buf = Array.empty(
                self._bound_device,
                type_.shape,
                type_.dtype,
                strides=type_.strides,
                first_element_offset=type_.offset,
                allocator=allocator,
            )
            internal_args[name] = new_buf
            all_buffers.append(new_buf)

        return ComputationCallable(
            self._bound_device,
            self._tr_tree.get_leaf_parameters(),
            self._kernels,
            internal_args,
            all_buffers,
        )


class PlannedKernelCall:
    def __init__(
        self,
        kernel: StaticKernel,
        argnames: Sequence[str],
        adhoc_values: Mapping[str, numpy.generic],
    ):
        self._kernel = kernel
        self.argnames = argnames
        self._adhoc_values = adhoc_values

    def finalize(self, known_args: Mapping[str, Array | numpy.generic]) -> "KernelCall":
        args: list[Array | numpy.generic | None] = [None] * len(self.argnames)
        external_arg_positions = []

        for i, name in enumerate(self.argnames):
            if name in known_args:
                args[i] = known_args[name]
            elif name in self._adhoc_values:
                args[i] = self._adhoc_values[name]
            else:
                external_arg_positions.append((name, i))

        return KernelCall(self._kernel, self.argnames, args, external_arg_positions)


class ComputationCallable:
    """
    A result of calling :py:meth:`~reikna.core.Computation.compile` on a computation.
    Represents a callable opaque GPGPU computation.

    .. py:attribute:: bound_device

        A :py:class:`grunnur.BoundDevice` object used to compile the computation.

    .. py:attribute:: signature

        A :py:class:`~reikna.core.Signature` object.

    .. py:attribute:: parameter

        A named tuple of :py:class:`~reikna.core.Type` objects corresponding
        to the callable's parameters.
    """

    def __init__(
        self,
        bound_device: "BoundDevice",
        parameters: Sequence[Parameter],
        kernel_calls: Iterable[PlannedKernelCall],
        internal_args: Mapping[str, Array | numpy.generic],
        temp_buffers: Iterable[Array],
    ):
        self.device = bound_device
        self.signature = Signature(parameters)
        self.parameter = ParameterContainer(self, parameters)
        self._kernel_calls = [kernel_call.finalize(internal_args) for kernel_call in kernel_calls]
        self._internal_args = internal_args
        self.__tempalloc__ = temp_buffers

    def __call__(
        self, queue: Queue, *args: Array | numpy.generic, **kwds: Array | numpy.generic
    ) -> Any:
        """
        Execute the computation.
        In case of the OpenCL backend, returns a list of ``pyopencl.Event`` objects
        from nested kernel calls.
        """
        bound_args = self.signature.bind_with_defaults(args, kwds, cast=True)
        return [kernel_call(queue, bound_args.arguments) for kernel_call in self._kernel_calls]


class KernelCall:
    def __init__(
        self,
        kernel: StaticKernel,
        argnames: Sequence[str],
        args: Iterable[Array | numpy.generic | None],
        external_arg_positions: Iterable[tuple[str, int]],
    ):
        self._argnames = argnames  # primarily for debugging purposes
        self._kernel = kernel
        self._args = list(args)
        self._external_arg_positions = external_arg_positions

    def __call__(self, queue: Queue, external_args: Mapping[str, Array | numpy.generic]) -> Any:
        for name, pos in self._external_arg_positions:
            self._args[pos] = external_args[name]

        result = self._kernel(queue, *cast(list[Array | numpy.generic], self._args))

        # releasing references to arrays
        for _name, pos in self._external_arg_positions:
            self._args[pos] = None

        return result
