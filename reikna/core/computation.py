import weakref
import itertools
from collections import namedtuple

from reikna.helpers import Graph
from reikna.core.signature import Parameter, Annotation, Type
from reikna.core.transformation import TransformationTree, TransformationParameter


class ComputationParameter(Type):
    """
    Bases: :py:class:`~reikna.core.Type`

    Represents a typed computation parameter.
    Can be used as a substitute of an array for functions
    which are only interested in array metadata.

    .. py:attribute:: name

        Parameter name.
    """

    def __init__(self, computation, name, type_):
        """__init__()""" # hide the signature from Sphinx

        Type.__init__(self, type_.dtype, shape=type_.shape, strides=type_.strides)
        self._computation = weakref.ref(computation)
        self.name = name

    def belongs_to(self, comp):
        return self._computation() is comp

    def connect(self, _tr, _tr_connector, **connections):
        """
        Shortcut for :py:meth:`~reikna.core.Computation.connect`
        with this parameter as a first argument.
        """
        return self._computation().connect(self.name, _tr, _tr_connector, **connections)


class Translator:

    def __init__(self, known_old, known_new, prefix=""):
        self._mapping = {old:new for old, new in zip(known_old, known_new)}
        self._prefix = prefix

    def __call__(self, name):
        if name in self._mapping:
            return self._mapping[name]
        else:
            return self._prefix + name

    def get_nested(self, known_old, known_new, prefix):
        return Translator(known_old, known_new, prefix=(prefix + self._prefix))


class Computation:
    """
    A base class for computations, intended to be subclassed.

    :param root_parameters: a list of :py:class:`~reikna.core.Parameter` objects.

    .. py:method:: _build_plan(plan_factory, device_params)

        Derived classes override this method.
        It is called by :py:meth:`compile` and
        supposed to return a :py:class:`~reikna.core.computation.ComputationPlan` object.

        :param plan_factory: a callable returning a new
            :py:class:`~reikna.core.computation.ComputationPlan` object.
        :param device_params: a :py:class:`~reikna.cluda.api.DeviceParameters` object corresponding
            to the thread the computation is being compiled for.

    .. py:attribute:: signature

        A :py:class:`~reikna.core.Signature` object representing current computation signature
        (taking into account connected transformations).

    .. py:attribute:: parameter

        A named tuple of :py:class:`~reikna.core.computation.ComputationParameter` objects
        corresponding to parameters from the :py:attr:`signature`.
    """

    def __init__(self, root_parameters):
        self._tr_tree = TransformationTree(root_parameters)
        self._update_parameters()

    def _update_parameters(self):
        params = self.signature.parameters.values()
        params_container = namedtuple('ComputationParameters', [param.name for param in params])
        param_objs = [
            ComputationParameter(self, param.name, param.annotation.type)
            for param in params]
        self.parameter = params_container(*param_objs)

    @property
    def signature(self):
        return self._tr_tree.get_leaf_signature()

    # The names are underscored to avoid name conflicts with ``connections`` keys
    # (where the user can introduce new parameter names)
    def connect(self, _param, _tr, _tr_param, **param_connections):
        """
        Connect a transformation to the computation.

        :param _param: connection target ---
            a :py:class:`~reikna.core.computation.ComputationParameter` object
            beloning to this computation object, or a string with its name.
        :param _tr: a :py:class:`~reikna.core.Transformation` object.
        :param _tr_param: connector on the side of the transformation ---
            a :py:class:`~reikna.core.transformation.TransformationParameter` object
            beloning to ``tr``, or a string with its name.
        :param param_connections: a dictionary with the names of new or old
            computation parameters as keys, and
            :py:class:`~reikna.core.transformation.TransformationParameter` objects
            (or their names) as values.
        :returns: this computation object (modified).
        """

        if isinstance(_param, ComputationParameter):
            if not _param.belongs_to(self):
                raise ValueError("")
            param_name = _param.name
        elif isinstance(_param, str):
            param_name = _param
        else:
            raise ValueError("")

        param_connections[param_name] = _tr_param
        processed_connections = {}
        for comp_connection_name, tr_connection in param_connections.items():
            if isinstance(tr_connection, TransformationParameter):
                if not tr_connection.belongs_to(_tr):
                    raise ValueError("")
                tr_connection_name = tr_connection.name
            elif isinstance(tr_connection, str):
                tr_connection_name = tr_connection
            else:
                raise ValueError("")
            processed_connections[comp_connection_name] = tr_connection_name

        self._tr_tree.connect(param_name, _tr, processed_connections)
        self._update_parameters()
        return self

    def _get_plan(self, tr_tree, translator, thread):
        def plan_factory():
            return ComputationPlan(tr_tree, translator, thread)

        args = [
            KernelArgument(param.name, param.annotation.type)
            for param in tr_tree.root_signature.parameters.values()]

        return self._build_plan(plan_factory, thread.device_params, *args)

    def compile(self, thread):
        """
        Compiles the computation with given ``thread`` and returns
        a :py:class:`~reikna.core.computation.ComputationCallable` object.
        """
        translator = Translator([], [])
        return self._get_plan(self._tr_tree, translator, thread).finalize()


class IdGen:

    def __init__(self, prefix):
        self._counter = 0
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

        Type.__init__(self, type_.dtype, shape=type_.shape, strides=type_.strides)
        self.name = name


class ComputationPlan:
    """
    Computation plan recorder.
    """

    def __init__(self, tr_tree, translator, thread):
        """__init__()""" # hide the signature from Sphinx

        self._thread = thread
        self._tr_tree = tr_tree
        self._translator = translator

        self._nested_comp_idgen = IdGen('_nested')
        self._persistent_value_idgen = IdGen('_value')
        self._temp_array_idgen = IdGen('_temp')

        self._persistent_values = {}
        self._temp_arrays = set()
        self._dependencies = Graph()
        self._internal_params = {}
        self._kernels = []

    def _scalar(self, val):
        name = self._translator(self._persistent_value_idgen())
        annotation = Annotation(val)
        self._internal_params[name] = Parameter(name, annotation)
        self._persistent_values[name] = annotation.type(val)
        return KernelArgument(name, annotation.type)

    def persistent_array(self, arr):
        """
        Adds a persistent GPU array to the plan, and returns its identifier.
        """
        name = self._translator(self._persistent_value_idgen())
        parameter = Parameter(name, Annotation(arr, 'i'))
        self._internal_params[name] = parameter
        self._persistent_values[name] = self._thread.to_device(arr)
        return KernelArgument(name, parameter.annotation.type)

    def temp_array(self, shape, dtype, strides=None):
        """
        Adds a temporary GPU array to the plan, and returns its identifier.
        Temporary arrays can share physical memory and are only guaranteed
        not to be overwritten by writes to other temporary arrays which are explicitly
        marked as dependent in :py:meth:`kernel_call`.
        """
        name = self._translator(self._temp_array_idgen())
        parameter = Parameter(
            name, Annotation(Type(dtype, shape=shape, strides=strides), 'io'))
        self._internal_params[name] = parameter
        self._temp_arrays.add(name)
        return KernelArgument(name, parameter.annotation.type)

    def temp_array_like(self, arr):
        """
        Same as :py:meth:`temp_array`, taking the array properties
        from array or array-like object ``arr``.
        """
        return self.temp_array(arr.shape, arr.dtype, strides=arr.strides)

    def _process_user_arg(self, arg, known_annotation=None):
        if isinstance(arg, KernelArgument):
            return arg.name
        elif isinstance(arg, str):
            # FIXME: not prefixing this one, because it is a result of plan.temp_array() call
            # All computation parameters should be passes as ComputationParameter objects.
            # This should be changes when we start returning objects from temp_array()
            # instead of strings.
            return self._translator(arg)
        else:
            if known_annotation is not None:
                assert not known_annotation.array
                arg = known_annotation.type(arg)
            karg = self._scalar(arg)
            return karg.name

    def kernel_call(self, template_def, args, global_size,
            local_size=None, render_kwds=None,
            correlations=None, decorrelations=None):
        """
        Adds a kernel call to the plan.

        :param template_def: Mako template def for the kernel.
        :param args: a list of computation's
            :py:class:`~reikna.core.computation.ComputationParameter` objects,
            their names, identifiers returned by :py:meth:`temp_array`
            or :py:meth:`persistent_array`, or scalar values
            that are going to be passed to the kernel during execution.
        :param global_size: global size to use for the call.
        :param local_size: local size to use for the call.
            If ``None``, the local size will be picked automatically.
        :param render_kwds: dictionary with additional values used to render the template.
        :param correlations: list of pairs of array arguments with correlated access
            (see :ref:`access-correlations` for details);
            the rest will be considered to be decorrelated.
            Can't be used in conjunction with ``decorrelations``.
        :param decorrelations: list of pairs of array arguments with decorrelated access
            (see :ref:`access-correlations` for details);
            the rest will be considered to be correlated.
            Can't be used in conjunction with ``correlations``.

        .. note::

            If both ``correlations`` and ``decorrelations`` is None,
            all access is assumed to be decorrelated.
        """

        argnames = [self._process_user_arg(arg) for arg in args]
        params = []
        for argname in argnames:
            if argname in self._tr_tree.root_signature.parameters:
                params.append(self._tr_tree.root_signature.parameters[argname])
            else:
                params.append(self._internal_params[argname])

        if correlations is not None and decorrelations is not None:
            raise ValueError("Only one of 'correlations' or 'decorrelations' can be specified.")

        corrs = Graph()

        if correlations is not None:
            for mem1, mem2 in correlations:
                mem1 = self._process_user_arg(mem1)
                mem2 = self._process_user_arg(mem2)
                if mem1 not in self._persistent_values and mem2 not in self._persistent_values:
                    corrs.add_edge(mem1, mem2)

        elif decorrelations is not None:
            mems = [
                name for name in argnames
                if name not in self._persistent_values]
            corrs.add_cluster(mems)

            for mem1, mem2 in decorrelations:
                mem1 = self._process_user_arg(mem1)
                mem2 = self._process_user_arg(mem2)
                if mem1 not in self._persistent_values and mem2 not in self._persistent_values:
                    corrs.remove_edge(mem1, mem2)

        ts_kernel = TypedStaticKernel(
            params, template_def, global_size,
            local_size=local_size, render_kwds=render_kwds,
            correlations=corrs)
        ts_kernel.reconnect(self._tr_tree)

        kernel = ts_kernel.compile(self._thread)
        leaf_argnames = ts_kernel.get_leaf_argnames()
        dependencies = ts_kernel.get_dependencies()

        self._dependencies.add_edges(dependencies)
        self._kernels.append(PlannedKernelCall(kernel, leaf_argnames))

    def computation_call(self, computation, *args, **kwds):
        """
        Adds a nested computation call.
        The ``computation`` value must be a :py:class:`~reikna.core.Computation` object.
        ``args`` and ``kwds`` are values to be passed to the computation.
        """

        sig = computation.signature
        ba = sig.bind_with_defaults(args, kwds, cast=False)

        argnames = [
            self._process_user_arg(arg, known_annotation=param.annotation)
            for arg, param in zip(ba.args, sig.parameters.values())]

        translator = self._translator.get_nested(
            sig.parameters, argnames, self._nested_comp_idgen())
        tr_tree = computation._tr_tree.translate(translator)
        tr_tree.reconnect(self._tr_tree)

        self._append_plan(computation._get_plan(tr_tree, translator, self._thread))

    def _append_plan(self, plan):
        self._kernels += plan._kernels
        self._persistent_values.update(plan._persistent_values)
        self._temp_arrays.update(plan._temp_arrays)
        self._internal_params.update(plan._internal_params)
        self._dependencies.add_graph(plan._dependencies)

    def finalize(self):

        # The user specified dependencies between external arguments,
        # and dependencies for the arguments of each kernels.
        # Now we need to add inferred dependencies.
        # Basically, we assume that if some buffer X was used first in kernel M
        # and last in kernel N, all buffers in kernels from M+1 till N-1 depend on it
        # (in other words, data in X has to persist from call M till call N)
        usage = {}

        # We are interested in: 1) temporary allocations and 2) external array arguments
        watchlist = set(self._temp_arrays)
        watchlist.update(
            param.name for param in self._tr_tree.get_leaf_signature().parameters.values()
            if param.annotation.array)

        for i, kernel in enumerate(self._kernels):
            for argname in kernel.argnames:
                if argname not in watchlist:
                    continue
                if argname in usage:
                    usage[argname][1] = i
                else:
                    usage[argname] = [i, i]

        for name, pair in usage.items():
            start, end = pair
            if end - start < 2:
                continue
            for i in range(start + 1, end):
                for other_name in self._kernels[i].argnames:
                    if other_name not in watchlist or other_name == name:
                        continue
                    self._dependencies.add_edge(name, other_name)

        internal_args = dict(self._persistent_values)

        # Allocate buffers specifying the dependencies
        for name in self._temp_arrays:
            dependencies = []
            for dep in self._dependencies[name]:
                if dep in internal_args:
                    dependencies.append(internal_args[dep])
            type_ = self._internal_params[name].annotation.type
            internal_args[name] = self._thread.temp_array(
                type_.shape, type_.dtype, strides=type_.strides, dependencies=dependencies)

        return ComputationCallable(
            self._tr_tree.get_leaf_signature(),
            self._kernels,
            internal_args)


class TypedStaticKernel:

    def __init__(self, params, template_def, global_size, local_size=None,
            render_kwds=None, correlations=None):

        self.template_def = template_def
        self.tr_tree = TransformationTree(params)
        self.global_size = global_size
        self.local_size = local_size
        self.render_kwds = render_kwds
        self.correlations = Graph(correlations.pairs() if correlations is not None else None)

    def _propagate_correlations(self, ntr):

        tr_graph = ntr.get_node_correlations()

        conn_name = ntr.connector_node_name

        kernel_corrs = self.correlations[conn_name]
        tr_corrs = tr_graph[conn_name]

        kernel_params = self.tr_tree.get_leaf_signature().parameters
        tr_params = {
            node_name:ntr.tr.signature.parameters[tr_name].rename(node_name)
            for tr_name, node_name in ntr.node_from_tr.items()}

        pairs = []

        for kernel_corr in kernel_corrs:
            for tr_corr in tr_corrs:
                if tr_corr != kernel_corr:
                # The equality can happen if we are connecting on of the transformation's inputs
                # to an existing input parameter.
                    pairs.append((kernel_corr, tr_corr))

        self.correlations.add_graph(tr_graph)
        self.correlations.add_edges(pairs)

    def reconnect(self, other_tree):
        for ntr in other_tree.connections():
            if ntr.connector_node_name in self.tr_tree.nodes:
                self._propagate_correlations(ntr)
                self.tr_tree._connect(ntr)
                if ntr.connector_node_name not in self.tr_tree.get_leaf_signature().parameters:
                    self.correlations.remove_node(ntr.connector_node_name)

    def compile(self, thread):
        kernel_name = '_kernel_func'

        kernel_definition = self.tr_tree.get_kernel_definition(kernel_name)
        kernel_argobjects = self.tr_tree.get_argobjects()

        if self.render_kwds is None:
            render_kwds = {}
        else:
            render_kwds = dict(self.render_kwds)

        assert 'kernel_definition' not in render_kwds
        render_kwds['kernel_definition'] = kernel_definition

        return thread.compile_static(
            self.template_def, kernel_name, self.global_size, local_size=self.local_size,
            render_args=kernel_argobjects, render_kwds=render_kwds)

    def get_leaf_argnames(self):
        return [param.name for param in self.tr_tree.get_leaf_signature().parameters.values()]

    def get_dependencies(self):
        # Up to this point we were keeping correlations between parameters
        # regardless of whether they are input or output,
        # because we did not know what transformation may be attached in future;
        # time to get rid of correlations between two inputs.
        # On the other hand, we need to add dependencies between all of the outputs,
        # because they shouldn't overwrite each other.
        params = self.tr_tree.get_leaf_signature().parameters
        array_names = [param.name for param in params.values() if param.annotation.array]

        dep_pairs = []
        for name1, name2 in itertools.combinations(array_names, 2):
            p1 = params[name1]
            p2 = params[name2]
            if (
                (p1.annotation.output and p2.annotation.output) or
                (
                    (p1.annotation.output or p2.annotation.output) and
                    (
                        (name1, name2) not in self.correlations.pairs() or
                        # FIXME: need to compare strides too, as different strides break correlation
                        p1.annotation.type.dtype != p2.annotation.type.dtype))):
                dep_pairs.append((name1, name2))

        return dep_pairs


class PlannedKernelCall:

    def __init__(self, kernel, argnames):
        self._kernel = kernel
        self.argnames = argnames

    def finalize(self, known_args):
        args = [None] * len(self.argnames)
        external_arg_positions = []

        for i, name in enumerate(self.argnames):
            if name in known_args:
                args[i] = known_args[name]
            else:
                external_arg_positions.append((name, i))

        return KernelCall(self._kernel, self.argnames, args, external_arg_positions)


class ComputationCallable:
    """
    A result of calling :py:meth:`~reikna.core.Computation.compile` on a computation.
    Represents a callable opaque GPGPU computation.

    .. py:attribute:: signature

        A :py:class:`~reikna.core.Signature` object.
    """

    def __init__(self, signature, kernel_calls, internal_args):
        self.signature = signature
        self._kernel_calls = [kernel_call.finalize(internal_args) for kernel_call in kernel_calls]
        self._internal_args = internal_args

    def __call__(self, *args, **kwds):
        """
        Execute the computation.
        """
        ba = self.signature.bind_with_defaults(args, kwds, cast=True)
        for kernel_call in self._kernel_calls:
            kernel_call(ba.arguments)


class KernelCall:

    def __init__(self, kernel, argnames, args, external_arg_positions):
        self.argnames = argnames # primarily for debugging purposes
        self._kernel = kernel
        self._args = args
        self._external_arg_positions = external_arg_positions

    def __call__(self, external_args):
        for name, pos in self._external_arg_positions:
            self._args[pos] = external_args[name]

        self._kernel(*self._args)

        # releasing references to arrays
        for name, pos in self._external_arg_positions:
            self._args[pos] = None
