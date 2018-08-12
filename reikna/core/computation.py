import weakref
from collections import namedtuple

from reikna.cluda import cuda_id
from reikna.helpers import Graph
from reikna.core.signature import Parameter, Annotation, Type, Signature
from reikna.core.transformation import TransformationTree, TransformationParameter


class ComputationParameter(Type):
    """
    Bases: :py:class:`~reikna.core.Type`

    Represents a typed computation parameter.
    Can be used as a substitute of an array for functions
    which are only interested in array metadata.
    """

    def __init__(self, computation, name, type_):
        """__init__()""" # hide the signature from Sphinx

        Type.__init__(
            self, type_.dtype, shape=type_.shape, strides=type_.strides, offset=type_.offset)
        self._computation = weakref.ref(computation)
        self._name = name

    def belongs_to(self, comp):
        return self._computation() is comp

    def connect(self, _trf, _tr_connector, **tr_from_comp):
        """
        Shortcut for :py:meth:`~reikna.core.Computation.connect`
        with this parameter as a first argument.
        """
        return self._computation().connect(self._name, _trf, _tr_connector, **tr_from_comp)

    def __str__(self):
        return self._name


class Translator:
    """
    A callable that translates strings from the known list, and prefixes unknown ones.
    Used to introduce parameter names from a nested computation to the parent namespace.
    """

    def __init__(self, known_old, known_new, prefix):
        self._mapping = dict((old, new) for old, new in zip(known_old, known_new))
        self._prefix = prefix

    def __call__(self, name):
        if name in self._mapping:
            return self._mapping[name]
        else:
            return (self._prefix + '_' if self._prefix != '' else '') + name

    def get_nested(self, known_old, known_new, prefix):
        """Returns a new ``Translator`` with an extended prefix."""
        return Translator(known_old, known_new, prefix + self._prefix)

    @classmethod
    def identity(cls):
        return cls([], [], "")


def check_external_parameter_name(name):
    """
    Checks that a user-supplied parameter name meets the special criteria.
    Basically, we do not want such names start with underscores
    to prevent them from conflicting with internal names.
    """
    # Raising errors here so we can provide better explanation for the user
    if name.startswith('_'):
        raise ValueError("External parameter name cannot start with the underscore.")


def make_parameter_container(parent, parameters):
    """
    Creates a convenience object with ``ComputationParameter`` attributes.
    """
    params_container = namedtuple(
        'ComputationParameters', [param.name for param in parameters])
    param_objs = [
        ComputationParameter(parent, param.name, param.annotation.type)
        for param in parameters]
    return params_container(*param_objs)


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

    def __init__(self, root_parameters):
        for param in root_parameters:
            check_external_parameter_name(param.name)
        self._tr_tree = TransformationTree(root_parameters)
        self._update_attributes()

    def _update_attributes(self):
        """
        Updates ``signature`` and ``parameter`` attributes.
        Called by the methods that change the signature.
        """
        leaf_params = self._tr_tree.get_leaf_parameters()
        self.signature = Signature(leaf_params)
        self.parameter = make_parameter_container(self, leaf_params)

    # The names are underscored to avoid name conflicts with ``tr_from_comp`` keys
    # (where the user can introduce new parameter names)
    def connect(self, _comp_connector, _trf, _tr_connector, **tr_from_comp):
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
        if isinstance(_comp_connector, ComputationParameter):
            if not _comp_connector.belongs_to(self):
                raise ValueError("The connection target must belong to this computation.")
        param_name = str(_comp_connector)

        # Extract transformation parameters names

        if param_name in tr_from_comp:
            raise ValueError(
                "Parameter '" + param_name + "' cannot be supplied " +
                "both as the main connector and one of the child connections")

        tr_from_comp[param_name] = _tr_connector
        comp_from_tr = {}
        for comp_connection_name, tr_connection in tr_from_comp.items():
            check_external_parameter_name(comp_connection_name)
            if isinstance(tr_connection, TransformationParameter):
                if not tr_connection.belongs_to(_trf):
                    raise ValueError(
                        "The transformation parameter must belong to the provided transformation")
            tr_connection_name = str(tr_connection)
            comp_from_tr[tr_connection_name] = comp_connection_name

        self._tr_tree.connect(param_name, _trf, comp_from_tr)
        self._update_attributes()
        return self

    def _translate_tree(self, translator):
        return self._tr_tree.translate(translator)

    def _get_plan(self, tr_tree, translator, thread, fast_math, compiler_options, keep):
        plan_factory = lambda: ComputationPlan(
            tr_tree, translator, thread, fast_math, compiler_options, keep)
        args = [
            KernelArgument(param.name, param.annotation.type)
            for param in tr_tree.get_root_parameters()]
        return self._build_plan(plan_factory, thread.device_params, *args)

    def compile(self, thread, fast_math=False, compiler_options=None, keep=False):
        """
        Compiles the computation with the given :py:class:`~reikna.cluda.api.Thread` object
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
            self._tr_tree, translator, thread, fast_math, compiler_options, keep).finalize()

    def _build_plan(self, plan_factory, device_params, *args):
        """
        Derived classes override this method.
        It is called by :py:meth:`compile` and
        supposed to return a :py:class:`~reikna.core.computation.ComputationPlan` object.

        :param plan_factory: a callable returning a new
            :py:class:`~reikna.core.computation.ComputationPlan` object.
        :param device_params: a :py:class:`~reikna.cluda.api.DeviceParameters` object corresponding
            to the thread the computation is being compiled for.
        :param args: :py:class:`~reikna.core.computation.KernelArgument` objects,
            corresponding to ``parameters`` specified during the creation
            of this computation object.
        """
        raise NotImplementedError


class IdGen:
    """
    Encapsulates a simple ID generator.
    """

    def __init__(self, prefix, counter=0):
        self._counter = counter
        self._prefix = prefix

    def __call__(self):
        self._counter += 1
        return self._prefix + str(self._counter)


class KernelArgument(Type):
    """
    Bases: :py:class:`~reikna.core.Type`

    Represents an argument suitable to pass to planned kernel or computation call.
    """

    def __init__(self, name, type_):
        """__init__()""" # hide the signature from Sphinx
        Type.__init__(
            self, type_.dtype, shape=type_.shape, strides=type_.strides, offset=type_.offset)
        self.name = name

    def __repr__(self):
        return "KernelArgument(" + self.name + ")"


class ComputationPlan:
    """
    Computation plan recorder.
    """

    def __init__(self, tr_tree, translator, thread, fast_math, compiler_options, keep):
        """__init__()""" # hide the signature from Sphinx

        self._thread = thread
        self._is_cuda = (thread.api.get_id() == cuda_id())
        self._tr_tree = tr_tree
        self._translator = translator
        self._fast_math = fast_math
        self._compiler_options = compiler_options
        self._keep = keep

        self._nested_comp_idgen = IdGen('_nested')
        self._persistent_value_idgen = IdGen('_value')
        self._constant_value_idgen = IdGen('_constant')
        self._temp_array_idgen = IdGen('_temp')

        self._external_annotations = self._tr_tree.get_root_annotations()
        self._persistent_values = {}
        self._constant_arrays = {}
        self._temp_arrays = set()
        self._internal_annotations = {}
        self._kernels = []

    def _scalar(self, val):
        """
        Adds a persistent scalar to the plan, and returns the corresponding
        :py:class:`KernelArgument`.
        """
        name = self._translator(self._persistent_value_idgen())
        ann = Annotation(val)
        self._internal_annotations[name] = ann
        self._persistent_values[name] = ann.type(val)
        return KernelArgument(name, ann.type)

    def persistent_array(self, arr):
        """
        Adds a persistent GPU array to the plan, and returns the corresponding
        :py:class:`KernelArgument`.
        """
        name = self._translator(self._persistent_value_idgen())
        ann = Annotation(arr, 'i')
        self._internal_annotations[name] = ann
        self._persistent_values[name] = self._thread.to_device(arr)
        return KernelArgument(name, ann.type)

    def temp_array(self, shape, dtype, strides=None, offset=0, nbytes=None):
        """
        Adds a temporary GPU array to the plan, and returns the corresponding
        :py:class:`KernelArgument`.
        See :py:meth:`~reikna.cluda.api.Thread.array` for the information about the parameters.

        Temporary arrays can share physical memory, but in such a way that
        their contents is guaranteed to persist between the first and the last use in a kernel
        during the execution of the plan.
        """
        name = self._translator(self._temp_array_idgen())
        ann = Annotation(
            Type(dtype, shape=shape, strides=strides, offset=offset, nbytes=nbytes), 'io')
        self._internal_annotations[name] = ann
        self._temp_arrays.add(name)
        return KernelArgument(name, ann.type)

    def constant_array(self, arr):
        """
        Adds a constant GPU array to the plan, and returns the corresponding
        :py:class:`KernelArgument`.
        """
        name = self._translator(self._constant_value_idgen())
        ann = Annotation(arr, constant=True)
        self._internal_annotations[name] = ann
        self._constant_arrays[name] = self._thread.to_device(arr)
        return KernelArgument(name, ann.type)

    def temp_array_like(self, arr):
        """
        Same as :py:meth:`temp_array`, taking the array properties
        from array or array-like object ``arr``.

        .. warning::

            Note that ``pycuda.GPUArray`` objects do not have the ``offset`` attribute.
        """
        if hasattr(arr, 'strides'):
            strides = arr.strides
        else:
            strides = None
        if hasattr(arr, 'offset'):
            offset = arr.offset
        else:
            offset = 0
        if hasattr(arr, 'nbytes'):
            nbytes = arr.nbytes
        else:
            nbytes = None
        return self.temp_array(
            arr.shape, arr.dtype, strides=strides, offset=offset, nbytes=nbytes)

    def _get_annotation(self, name):
        if name in self._external_annotations:
            return self._external_annotations[name]
        else:
            return self._internal_annotations[name]

    def _process_kernel_arguments(self, args):
        """
        Scan through kernel arguments passed by the user, check types,
        and wrap ad hoc values if necessary.

        Does not change the plan state.
        """
        processed_args = []
        adhoc_idgen = IdGen('_adhoc')
        adhoc_values = {}

        for arg in args:
            if not isinstance(arg, KernelArgument):
                if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
                    if len(arg.shape) > 0:
                        raise ValueError("Arrays are not allowed as ad hoc arguments")

                    # Not creating a new persistent scalar with _scalar(),
                    # because the kernel compilation may fail,
                    # in which case we would have to roll back the plan state.
                    # These arguments are local to this kernel anyway,
                    # so there's no need in registering them in the plan.
                    name = self._translator(adhoc_idgen())
                    adhoc_values[name] = arg
                    annotation = Annotation(Type(arg.dtype))
                    arg = KernelArgument(name, annotation.type)
                else:
                    raise TypeError("Unknown argument type: " + str(type(arg)))
            else:
                annotation = self._get_annotation(arg.name)

            processed_args.append(Parameter(arg.name, annotation))

        return processed_args, adhoc_values

    def _process_computation_arguments(self, signature, args, kwds):
        """
        Scan through nested computation arguments passed by the user, check types,
        and create new persistent values for ad hoc arguments.

        Changes the plan state.
        """
        bound_args = signature.bind_with_defaults(args, kwds, cast=False)

        args = []
        for arg, param in zip(bound_args.args, signature.parameters.values()):

            if not isinstance(arg, KernelArgument):
                if param.annotation.array:
                    raise ValueError("Ad hoc arguments are only allowed for scalar parameters")
                arg = self._scalar(param.annotation.type(arg))

            annotation = self._get_annotation(arg.name)

            if not annotation.can_be_argument_for(param.annotation):
                raise TypeError(
                    "Got {annotation} for '{name}', expected {param_annotation}".format(
                        annotation=annotation, name=param.name,
                        param_annotation=param.annotation))

            args.append(arg)

        return [arg.name for arg in args]

    def kernel_call(self, template_def, args, global_size,
            local_size=None, render_kwds=None, kernel_name='_kernel_func'):
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
            kernel_name, skip_constants=self._is_cuda)
        kernel_argobjects = subtree.get_kernel_argobjects()

        if render_kwds is None:
            render_kwds = {}
        else:
            render_kwds = dict(render_kwds)

        if self._is_cuda:
            constant_arrays = {}
            for karg in kernel_argobjects:
                if karg.name in self._constant_arrays:
                    constant_arrays[str(karg)] = self._constant_arrays[karg.name]
        else:
            constant_arrays = None

        kernel = self._thread.compile_static(
            template_def, kernel_name, global_size, local_size=local_size,
            render_args=[kernel_declaration] + kernel_argobjects,
            render_kwds=render_kwds,
            fast_math=self._fast_math,
            compiler_options=self._compiler_options,
            constant_arrays=constant_arrays,
            keep=self._keep)

        if self._is_cuda:
            for name, arr in constant_arrays.items():
                kernel.set_constant(name, arr)

        self._kernels.append(PlannedKernelCall(kernel, kernel_leaf_names, adhoc_values))

    def computation_call(self, computation, *args, **kwds):
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
            signature.parameters, argnames, self._nested_comp_idgen())
        new_tree = computation._translate_tree(translator)
        new_tree.reconnect(self._tr_tree)

        self._append_plan(computation._get_plan(
            new_tree, translator, self._thread, self._fast_math,
            self._compiler_options, self._keep))

    def _append_plan(self, plan):
        self._kernels += plan._kernels
        if not self._is_cuda:
            # In case of CUDA each kernel manages its constant arrays itself,
            # no need to remember them.
            self._constant_arrays.update(plan._constant_arrays)
        self._persistent_values.update(plan._persistent_values)
        self._temp_arrays.update(plan._temp_arrays)
        self._internal_annotations.update(plan._internal_annotations)

    def finalize(self):

        # We need to add inferred dependencies between temporary buffers.
        # Basically, we assume that if some buffer X was used first in kernel M
        # and last in kernel N, all buffers in kernels from M+1 till N-1 depend on it
        # (in other words, data in X has to persist from call M till call N)

        # Record first and last kernel when a certain temporary array was used.
        usage = {}
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
        dependencies = Graph()
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
            dependent_buffers = []
            for dep in dependencies[name]:
                if dep in internal_args:
                    dependent_buffers.append(internal_args[dep])

            type_ = self._internal_annotations[name].type
            new_buf = self._thread.temp_array(
                type_.shape, type_.dtype, strides=type_.strides, offset=type_.offset,
                dependencies=dependent_buffers)
            internal_args[name] = new_buf
            all_buffers.append(new_buf)

        return ComputationCallable(
            self._thread,
            self._tr_tree.get_leaf_parameters(),
            self._kernels,
            internal_args,
            all_buffers)


class PlannedKernelCall:

    def __init__(self, kernel, argnames, adhoc_values):
        self._kernel = kernel
        self.argnames = argnames
        self._adhoc_values = adhoc_values

    def finalize(self, known_args):
        args = [None] * len(self.argnames)
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

    .. py:attribute:: thread

        A :py:class:`~reikna.cluda.api.Thread` object used to compile the computation.

    .. py:attribute:: signature

        A :py:class:`~reikna.core.Signature` object.

    .. py:attribute:: parameter

        A named tuple of :py:class:`~reikna.core.Type` objects corresponding
        to the callable's parameters.
    """

    def __init__(self, thread, parameters, kernel_calls, internal_args, temp_buffers):
        self.thread = thread
        self.signature = Signature(parameters)
        self.parameter = make_parameter_container(self, parameters)
        self._kernel_calls = [kernel_call.finalize(internal_args) for kernel_call in kernel_calls]
        self._internal_args = internal_args
        self.__tempalloc__ = temp_buffers

    def __call__(self, *args, **kwds):
        """
        Execute the computation.
        In case of the OpenCL backend, returns a list of ``pyopencl.Event`` objects
        from nested kernel calls.
        """
        bound_args = self.signature.bind_with_defaults(args, kwds, cast=True)
        results = []
        for kernel_call in self._kernel_calls:
            results.append(kernel_call(bound_args.arguments))
        return results


class KernelCall:

    def __init__(self, kernel, argnames, args, external_arg_positions):
        self._argnames = argnames # primarily for debugging purposes
        self._kernel = kernel
        self._args = args
        self._external_arg_positions = external_arg_positions

    def __call__(self, external_args):
        for name, pos in self._external_arg_positions:
            self._args[pos] = external_args[name]

        result = self._kernel(*self._args)

        # releasing references to arrays
        for name, pos in self._external_arg_positions:
            self._args[pos] = None

        return result
