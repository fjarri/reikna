from collections import defaultdict

from reikna.helpers import AttrDict
import reikna.cluda.dtypes as dtypes
from reikna.core.transformation import ArrayValue


class OperationRecorder:

    def __init__(self, prefix, thr, tr_tree, basis, base_values):
        self._thr = thr
        self._prefix = prefix
        self._tr_tree = tr_tree
        self.basis = basis
        self.values = AttrDict(base_values)
        self.kernels = []
        self._allocations = {}
        self._const_allocations = {}
        self.scalars = {}
        self._dependencies = defaultdict(set)

        self._temp_counter = 0
        self._const_counter = 0
        self._scalar_counter = 0

    def add_allocation(self, shape, dtype):
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

    def add_const_allocation(self, data):
        name = "_const" + str(self._const_counter)
        self._const_counter += 1

        value = ArrayValue(data.shape, data.dtype)
        self.values[self._prefix + name] = value
        self._const_allocations[self._prefix + name] = self._thr.to_device(data)
        self._tr_tree.add_temp_node(self._prefix + name, value)
        return name

    def add_scalar(self, x):
        name = '_scalar' + str(self._scalar_counter)
        self._scalar_counter += 1

        dtype = detect_type(x)
        value = ScalarValue(dtype)
        self.values[self._prefix + name] = value
        self.scalars[self._prefix + name] = dtypes.cast(dtype)(x)
        self._tr_tree.add_temp_node(self._prefix + name, value)
        return name

    def add_kernel(self, template, argnames,
            global_size, local_size=None, render_kwds=None, dependencies=None):
        """
        Adds kernel execution to the list of actions.
        See :ref:`tutorial-advanced-computation` for details on how to write kernels.

        :param template: Mako template for the kernel.
        :param argnames: names of the arguments the kernel takes.
            These must either belong to the list of external argument names, or be allocated by
            :py:meth:`~reikna.core.operation.OperationRecorder.add_allocation` earlier.
        :param global_size: global size to use for the call.
        :param local_size: local size to use for the call.
            If ``None``, the local size will be picked automatically.
        :param render_kwds: dictionary with additional values used to render the template.
        :param dependencies: list of pairs of buffer identifiers which depend on each other
            (i.e., should not be assigned to the same physical memory allocation).
        """

        argnames = [self._prefix + name for name in argnames]
        assert set(argnames).issubset(set(self.values))

        kernel_name = '_kernel_func'
        kernel_definition, argobjects = self._tr_tree.transformations_for(kernel_name, argnames)

        if render_kwds is None:
            render_kwds = {}

        additional_kwds = dict(
            basis=self.basis,
            kernel_definition=kernel_definition)

        # check that user keywords do not overlap with our keywords
        intersection = set(render_kwds).intersection(additional_kwds)
        if len(intersection) > 0:
            raise ValueError("Render keywords clash with internal variables: " +
                ", ".join(intersection))
        render_kwds = dict(render_kwds) # shallow copy
        render_kwds.update(additional_kwds)

        kernel = self._thr.compile_static(
            template, kernel_name, global_size, local_size=local_size,
            render_args=argobjects, render_kwds=render_kwds)
        leaf_argnames = [name for name, _ in self._tr_tree.leaf_signature(argnames)]

        self.kernels.append(KernelCall(kernel, leaf_argnames))
        if dependencies is not None:
            for mem1, mem2 in dependencies:
                mem1 = self._prefix + mem1
                mem2 = self._prefix + mem2
                if mem1 in self._const_allocations or mem2 in self._const_allocations:
                    continue

                self._dependencies[mem1].add(mem2)
                self._dependencies[mem2].add(mem1)

    def add_computation(self, computation, *argnames, **kwds):
        """
        Adds a nested computation call. The ``computation`` value must be a computation
        with necessary basis set and transformations connected.
        ``argnames`` list specifies which positional arguments will be passed to this kernel.
        """
        for i, arg in enumerate(argnames):
            if not isinstance(arg, str):
                argnames[i] = self.add_scalar(arg)

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
            computation.connect(tr, array_arg, new_array_args,
                new_scalar_args=new_scalar_args, add_prefix=False)

        values = self._tr_tree.leaf_values_dict(argnames)
        ext_names = [map_to_ext(name) for name, _ in computation.leaf_signature()]

        args = [values[name] for name in ext_names]
        computation.prepare_for(*args, **kwds)

        nested_ops = computation._operations

        for kernel in nested_ops.kernels:
            kernel.argnames = [map_to_ext(name) for name in kernel.argnames]
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
            for other_name in deps:
                self._dependencies[other_name].add(name)

    def finalize(self):

        # The user specified dependencies between external arguments,
        # and dependencies for the arguments of each kernels.
        # Now we need to add inferred dependencies.
        # Basically, we assume that if some buffer X was used first in kernel M
        # and last in kernel N, all buffers in kernels from M+1 till N-1 depend on it
        # (in other words, data in X has to persist from call M till call N)
        usage = {}
        watchlist = set(name for name, value in self.values.items()
            if name not in self._const_allocations and value.is_array)

        for i, kernel in enumerate(self.kernels):
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
                for other_name in self.kernels[i].argnames:
                    if other_name not in watchlist:
                        continue
                    self._dependencies[name].add(other_name)
                    self._dependencies[other_name].add(name)

        # Allocate buffers specifying the dependencies
        self.allocations = {}
        for name, value in self._allocations.items():
            dependencies = []
            for dep in self._dependencies[name]:
                if dep in self.allocations:
                    dependencies.append(self.allocations[dep])
            self.allocations[name] = self._thr.temp_array(
                value.shape, value.dtype, dependencies=dependencies)


class KernelCall:

    def __init__(self, kernel, argnames):
        self.kernel = kernel
        self.argnames = argnames

    def __call__(self, *args):
        self.kernel(*args)
