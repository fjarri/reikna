import tigger.cluda.dtypes as dtypes
from tigger.core.transformation import *
from tigger.cluda.kernel import render_prelude, render_template


class Argument:

    def __init__(self, name, dtype):
        self.dtype = dtype
        self.ctype = dtypes.ctype(dtype)
        self.load = load_macro_call(name)
        self.store = store_macro_call(name)
        self._name = name

    def __str__(self):
        return leaf_name(self._name)


class OperationRecorder:

    def __init__(self, ctx, tr_tree, basis, base_values):
        self._ctx = ctx
        self._tr_tree = tr_tree
        self.basis = basis
        self.values = AttrDict(base_values)
        self.operations = []
        self._allocations = {}
        self._const_allocations = {}

        self._temp_counter = 0
        self._const_counter = 0

    def add_allocation(self, shape, dtype):
        """
        Adds an allocation to the list of actions.
        Returns the string which can be used later in the list of argument names for kernels.
        """
        name = "_temp" + str(self._temp_counter)
        self._temp_counter += 1

        value = ArrayValue(shape, dtype)
        self.values[name] = value
        self._allocations[name] = value
        self._tr_tree.add_temp_node(name, value)
        return name

    def add_const_allocation(self, data):
        name = "_const" + str(self._const_counter)
        self._const_counter += 1

        value = ArrayValue(data.shape, data.dtype)
        self.values[name] = value
        self._const_allocations[name] = data
        self._tr_tree.add_temp_node(name, value)
        return name

    def add_kernel(self, template, defname, argnames,
            global_size, local_size=None, render_kwds=None, inplace=None):
        """
        Adds kernel execution to the list of actions.
        See :ref:`tutorial-advanced-computation` for details on how to write kernels.

        :param template: Mako template for the kernel.
        :param defname: name of the definition inside the template.
        :param argnames: names of the arguments the kernel takes.
            These must either belong to the list of external argument names,
            or be allocated by :py:meth:`add_allocation` earlier.
        :param global_size: global size to use for the call.
        :param local_size: local size to use for the call.
            If ``None``, the local size will be picked automatically.
        :param render_kwds: dictionary with additional values used to render the template.
        :param inplace: list of pairs (output, input) which can point to the same point in memory
            (used as a hint for the temporary memory manager).
        """

        subtemplate = template.get_def(defname)

        assert set(argnames).issubset(set(self.values))
        args = [Argument(name, self.values[name].dtype) for name in argnames]

        if render_kwds is None:
            render_kwds = {}

        additional_kwds = dict(
            basis=self.basis,
            kernel_definition=kernel_definition(defname))

        # check that user keywords do not overlap with our keywords
        intersection = set(render_kwds).intersection(additional_kwds)
        if len(intersection) > 0:
            raise ValueError("Render keywords clash with internal variables: " +
                ", ".join(intersection))

        render_kwds = dict(render_kwds) # shallow copy
        render_kwds.update(additional_kwds)
        src = render_template(subtemplate, *args, **render_kwds)

        op = KernelCall(defname, argnames, src, global_size, local_size=local_size)
        op.prepare(self._ctx, self._tr_tree)
        self.operations.append(op)

    def add_computation(self, cls, *argnames, **kwds):
        """
        Adds a nested computation call. The ``computation`` value must be a computation
        with necessary basis set and transformations connected.
        ``argnames`` list specifies which positional arguments will be passed to this kernel.
        """
        operation = ComputationCall(cls, *argnames, **kwds)
        connections = self._tr_tree.connections_for(operation.argnames)
        for tr, array_arg, new_array_args, new_scalar_args in connections:
            operation.connect(tr, array_arg, new_array_args, new_scalar_args)
        operation.prepare({name:value for name, value in self._tr_tree.leaf_signature()})
        self.operations.append(operation)

    def optimize_execution(self):

        # In theory, we can optimize the usage of temporary buffers with help of views
        # Now we just allocate them separately
        self.allocations = {}
        for name, value in self._allocations.items():
            self.allocations[name] = self._ctx.allocate(
                value.shape, value.dtype)

        for name, data in self._const_allocations.items():
            self.allocations[name] = self._ctx.to_device(data)


class Allocate:

    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class ComputationCall:

    def __init__(self, computation, *argnames, **kwds):
        self.computation = computation
        self.argnames = argnames
        self.kwds = kwds
        self._update_maps()

    def _update_maps(self):
        argnames = [x for x, _ in self.computation.leaf_signature()]
        self.map_to_internal = {external_name:internal_name
            for external_name, internal_name in zip(self.argnames, argnames)}
        self.map_to_external = {internal_name:external_name
            for external_name, internal_name in zip(self.argnames, argnames)}

    def prepare(self, values):
        args = [values[name] for name in self.argnames]
        self.computation.prepare_for(*args, **self.kwds)

        replace = lambda x: self.map_to_external.get(x, x)
        argnames = [x for x, _ in self.computation.leaf_signature()]
        self.leaf_argnames = [replace(name) for name in argnames]

    def __call__(self, *args):
        self.computation(*args)

    def connect(self, tr, array_arg, new_array_args, new_scalar_args=None):
        internal_array_arg = self.map_to_internal[array_arg]
        self.computation.connect(tr, internal_array_arg, new_array_args, new_scalar_args)

        new_signature = [x for x, _ in self.computation.leaf_signature()]
        new_argnames = []
        for internal_name in new_signature:
            if internal_name in self.map_to_external:
                new_argnames.append(self.map_to_external[internal_name])
            elif internal_name in new_array_args:
                new_argnames.append(internal_name)
            elif new_scalar_args is not None and internal_name in new_scalar_args:
                new_argnames.append(internal_name)

        self.argnames = new_argnames

        self._update_maps()


class KernelCall:

    def __init__(self, name, base_argnames, base_src, global_size,
            local_size=None):
        self.name = name
        self.base_argnames = list(base_argnames)
        self.local_size = local_size
        self.global_size = global_size
        self.src = base_src

    def prepare(self, ctx, tr_tree):
        transformation_code = tr_tree.transformations_for(self.base_argnames)
        self.full_src = transformation_code + self.src
        self.kernel = ctx.compile_static(self.full_src, self.name,
            self.global_size, local_size=self.local_size)
        self.leaf_argnames = [name for name, _ in tr_tree.leaf_signature(self.base_argnames)]

    def __call__(self, *args):
        self.kernel(*args)
