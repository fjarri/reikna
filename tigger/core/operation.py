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

    def __init__(self, ctx, basis, base_values):
        self._ctx = ctx
        self.basis = basis
        self.values = AttrDict(base_values)
        self.operations = []
        self._allocations = {}

    def add_allocation(self, name, shape, dtype):
        assert name not in self.values
        value = ArrayValue(shape, dtype)
        self.values[name] = value
        self._allocations[name] = value

    def add_kernel(self, template, defname, argnames,
            global_size, local_size=None, render_kwds=None):

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

        render_kwds.update(additional_kwds)
        src = render_template(subtemplate, *args, **render_kwds)

        self.operations.append(KernelCall(defname, argnames, src,
            global_size, local_size=local_size))

    def add_computation(self, computation, *argnames):
        self.operations.append(ComputationCall(computation, *argnames))

    def get_allocation_values(self):
        return self._allocations

    def prepare(self, tr_tree):
        for operation in self.operations:
            if isinstance(operation, KernelCall):
                operation.prepare(self._ctx, tr_tree)
            elif isinstance(operation, ComputationCall):
                for tr, array_arg, new_array_args, new_scalar_args in tr_tree.connections_for(
                        operation.argnames):
                    operation.connect(tr, array_arg, new_array_args, new_scalar_args)
                operation.prepare()

    def connect(self, tr, array_arg, new_array_args, new_scalar_args):
        for op in self.operations:
            if isinstance(op, ComputationCall):
                op.connect(
                    tr, array_arg, new_array_args, new_scalar_args=new_scalar_args)

    def optimize_execution(self):

        # In theory, we can optimize the usage of temporary buffers with help of views
        # Now we just allocate them separately
        self.allocations = {}
        for name, value in self._allocations.items():
            self.allocations[name] = self._ctx.allocate(
                value.shape, value.dtype)


class Allocate:

    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class ComputationCall:

    def __init__(self, computation, *argnames):
        self.computation = computation
        self.argnames = argnames

        argnames = [x for x, _ in self.computation.leaf_signature()]
        self.map_to_internal = {external_name:internal_name
            for external_name, internal_name in zip(self.argnames, argnames)}
        self.map_to_external = {internal_name:external_name
            for external_name, internal_name in zip(self.argnames, argnames)}

    def prepare(self):
        self.computation.prepare()
        replace = lambda x: self.map_to_external.get(x, x)
        argnames = [x for x, _ in self.computation.leaf_signature()]
        self.leaf_argnames = [replace(name) for name in argnames]

    def __call__(self, *args):
        self.computation(*args)

    def connect(self, tr, array_arg, new_array_args, new_scalar_args=None):
        replace = lambda x: self.map_to_internal.get(x, x)
        array_arg = replace(array_arg)
        new_array_args = [replace(name) for name in new_array_args]
        new_scalar_args = [replace(name) for name in new_scalar_args]

        self.computation.connect(tr, array_arg, new_array_args, new_scalar_args)


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

