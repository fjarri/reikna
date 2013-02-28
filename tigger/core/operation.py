from collections import defaultdict

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

    def __init__(self, prefix, ctx, tr_tree, basis, base_values):
        self._ctx = ctx
        self._prefix = prefix
        self._tr_tree = tr_tree
        self.basis = basis
        self.values = AttrDict(base_values)
        self.kernels = []
        self._allocations = {}
        self._const_allocations = {}
        self._dependencies = defaultdict(set)

        self._temp_counter = 0
        self._const_counter = 0

    def add_allocation(self, shape, dtype, dependencies=None):
        """
        Adds an allocation to the list of actions.
        Returns the string which can be used later in the list of argument names for kernels.
        """
        name = "_temp" + str(self._temp_counter)
        self._temp_counter += 1

        value = ArrayValue(shape, dtype)
        self.values[self._prefix + name] = value
        self._allocations[self._prefix + name] = value
        self._tr_tree.add_temp_node(self._prefix + name, value)
        return name

    def add_dependency(self, mem1, mem2):
        self._dependencies[self._prefix + mem1].add(self._prefix + mem2)
        self._dependencies[self._prefix + mem2].add(self._prefix + mem1)

    def add_const_allocation(self, data):
        name = "_const" + str(self._const_counter)
        self._const_counter += 1

        value = ArrayValue(data.shape, data.dtype)
        self.values[self._prefix + name] = value
        self._const_allocations[self._prefix + name] = self._ctx.to_device(data)
        self._tr_tree.add_temp_node(self._prefix + name, value)
        return name

    def add_kernel(self, template, defname, argnames,
            global_size, local_size=None, render_kwds=None):
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
        """

        subtemplate = template.get_def(defname)
        argnames = [self._prefix + name for name in argnames]

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

        transformation_code = self._tr_tree.transformations_for(argnames)
        full_src = transformation_code + src
        kernel = self._ctx.compile_static(full_src, defname,
            global_size, local_size=local_size)
        leaf_argnames = [name for name, _ in self._tr_tree.leaf_signature(argnames)]

        self.kernels.append(KernelCall(kernel, leaf_argnames))

    def add_computation(self, computation, *argnames, **kwds):
        """
        Adds a nested computation call. The ``computation`` value must be a computation
        with necessary basis set and transformations connected.
        ``argnames`` list specifies which positional arguments will be passed to this kernel.
        """
        argnames = [self._prefix + name for name in argnames]
        connections = self._tr_tree.connections_for(argnames)
        int_argnames = computation.leaf_signature()

        ext_to_int = {e:i[0] for e, i in zip(argnames, int_argnames)}
        int_to_ext = {i[0]:e for e, i in zip(argnames, int_argnames)}

        map_to_int = lambda x: ext_to_int[x] if x in ext_to_int else x
        map_to_ext = lambda x: int_to_ext[x] if x in int_to_ext else x

        for tr, array_arg, new_array_args, new_scalar_args in connections:
            array_arg = map_to_int(array_arg)
            new_array_args = map(map_to_int, new_array_args)
            new_scalar_args = map(map_to_int, new_scalar_args)
            computation.connect(tr, array_arg, new_array_args, new_scalar_args)

        values = self._tr_tree.leaf_values_dict()
        ext_names = [map_to_ext(name) for name, _ in computation.leaf_signature()]

        args = [values[name] for name in ext_names]
        computation.prepare_for(*args, **kwds)

        nested_ops = computation._operations

        for kernel in nested_ops.kernels:
            kernel.argnames = ext_names
            self.kernels.append(kernel)

        for name, value in nested_ops._allocations.items():
            self._allocations[name] = value
            self._tr_tree.add_temp_node(name, value)

        for name, data in nested_ops._const_allocations.items():
            value = ArrayValue(data.shape, data.dtype)
            self._const_allocations[name] = data
            self._tr_tree.add_temp_node(name, value)

        for name, deps in nested_ops._dependencies.items():
            name = map_to_ext(name)
            deps = set(map(map_to_ext, deps))
            self._dependencies[name].update(deps)

    def finalize(self):

        self.allocations = {}
        for name, value in self._allocations.items():
            self.allocations[name] = self._ctx.array(
                value.shape, value.dtype)


class KernelCall:

    def __init__(self, kernel, argnames):
        self.kernel = kernel
        self.argnames = argnames

    def __call__(self, *args):
        self.kernel(*args)
