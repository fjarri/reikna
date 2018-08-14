import collections
import weakref

from reikna.helpers.sortedcollection import SortedCollection


def extract_dependencies(dependencies):
    """
    Recursively extracts allocation identifiers from an iterable or Array class.
    """
    results = set()

    if isinstance(dependencies, collections.Iterable):
        for dep in dependencies:
            results.update(extract_dependencies(dep))
    elif hasattr(dependencies, '__tempalloc__'):
        # hook for exposing temporary allocations in arbitrary classes
        results.update(extract_dependencies(dependencies.__tempalloc__))
    elif hasattr(dependencies, '__tempalloc_id__'):
        return set([dependencies.__tempalloc_id__])

    return results


class TemporaryManager:
    """
    Base class for a manager of temporary allocations.

    :param thr: an instance of :py:class:`~reikna.cluda.api.Thread`.
    :param pack_on_alloc: whether to repack allocations when a new allocation is requested.
    :param pack_on_free: whether to repack allocations when an allocation is freed.
    """

    def __init__(self, thr, pack_on_alloc=False, pack_on_free=False):
        self._thr = thr
        self._id_counter = 0
        self._arrays = {}
        self._pack_on_alloc = pack_on_alloc
        self._pack_on_free = pack_on_free

    def array(self, shape, dtype, strides=None, offset=0, nbytes=None, dependencies=None):
        """
        Returns a temporary array.

        :param shape: shape of the array.
        :param dtype: data type of the array.
        :param strides: tuple of bytes to step in each dimension when traversing an array.
        :param offset: the array offset (in bytes)
        :param nbytes: the buffer size for the array
            (if ``None``, the minimum required size will be used).
        :param dependencies: can be a :py:class:`~reikna.cluda.api.Array` instance
            (the ones containing persistent allocations will be ignored),
            an iterable with valid values,
            or an object with the attribute `__tempalloc__` which is a valid value
            (the last two will be processed recursively).
        """

        # Used to hook memory allocation in the array constructor
        # and save the requested raw memory size.
        class DummyAllocator:
            def __init__(self):
                self.size = None
            def __call__(self, size):
                self.size = size
                return 0

        new_id = self._id_counter
        self._id_counter += 1

        allocator = DummyAllocator()
        array = self._thr.array(
            shape, dtype, strides=strides, offset=offset, nbytes=nbytes, allocator=allocator)
        array.__tempalloc_id__ = new_id

        dependencies = extract_dependencies(dependencies)
        self._allocate(new_id, allocator.size, dependencies, self._pack_on_alloc)
        self._arrays[new_id] = weakref.ref(array, lambda _: self.free(new_id))

        if self._pack_on_alloc:
            self.update_all()
        else:
            self.update_buffer(new_id)

        return array

    def update_buffer(self, id_):
        array = self._arrays[id_]()
        buf = self._get_buffer(id_)
        array._tempalloc_update_buffer(buf)

    def update_all(self):
        for id_ in self._arrays:
            self.update_buffer(id_)

    def free(self, id_):
        array = self._arrays[id_]()
        if array is not None:
            raise Exception("Attempting to free the buffer of an existing temporary array")

        del self._arrays[id_]
        self._free(id_, self._pack_on_free)
        if self._pack_on_free:
            self.update_all()

    def pack(self):
        """
        Packs the real allocations possibly reducing total memory usage.
        This process can be slow.
        """
        self._pack()
        self.update_all()


class TrivialManager(TemporaryManager):
    """
    Trivial manager --- allocates a separate buffer for each allocation request.
    """

    def __init__(self, *args, **kwds):
        TemporaryManager.__init__(self, *args, **kwds)
        self._allocations = {}

    def _allocate(self, new_id, size, _dependencies, _pack):
        buf = self._thr.allocate(size)
        self._allocations[new_id] = buf

    def _get_buffer(self, id_):
        return self._allocations[id_]

    def _free(self, id_, _pack):
        del self._allocations[id_]

    def _pack(self):
        pass


class ZeroOffsetManager(TemporaryManager):
    """
    Tries to assign several allocation requests to a single real allocation,
    if dependencies allow that.
    All virtual allocations start from the beginning of real allocations.
    """

    VirtualAllocation = collections.namedtuple('VirtualAllocation', ['size', 'dependencies'])
    RealAllocation = collections.namedtuple('RealAllocation', ['buffer', 'virtual_ids'])
    RealSize = collections.namedtuple('RealSize', ['size', 'real_id'])
    VirtualMapping = collections.namedtuple('VirtualMapping', ['real_id', 'sub_region'])

    def __init__(self, *args, **kwds):
        TemporaryManager.__init__(self, *args, **kwds)

        self._virtual_allocations = {} # id -> VirtualAllocation
        self._real_sizes = SortedCollection(key=lambda x: x.size) # RealSize objects, sorted by size
        self._virtual_to_real = {} # id -> VirtualMapping
        self._real_allocations = {} # real_id -> RealAllocation
        self._real_id_counter = 0

    def _allocate(self, new_id, size, dependencies, pack):

        # Dependencies should be bidirectional.
        # So if some new allocation says it depends on earlier ones,
        # we need to update their dependency lists.
        dep_set = set(dependencies)
        for dep in dependencies:
            if dep in self._virtual_allocations:
                self._virtual_allocations[dep].dependencies.add(new_id)
            else:
                dep_set.remove(dep)

        # Save virtual allocation parameters
        self._virtual_allocations[new_id] = self.VirtualAllocation(size, dep_set)

        if pack:
            # If pack is requested, we can just do full re-pack right away.
            self._pack()
        else:
            # If not, find a real allocation using the greedy algorithm.
            self._fast_add(new_id, size, dep_set)

    def _fast_add(self, new_id, size, dep_set):
        """
        Greedy algorithm to find a real allocation for a given virtual allocation.
        """

        # Find the smallest real allocation which can hold the requested virtual allocation.
        try:
            idx_start = self._real_sizes.argfind_ge(size)
        except ValueError:
            idx_start = len(self._real_sizes)

        # Check all real allocations with suitable sizes, starting from the smallest one.
        # Use the first real allocation which does not contain ``new_id``'s dependencies.
        for idx in range(idx_start, len(self._real_sizes)):
            real_id = self._real_sizes[idx].real_id
            buf = self._real_allocations[real_id].buffer
            virtual_ids = self._real_allocations[real_id].virtual_ids
            if virtual_ids.isdisjoint(dep_set):
                virtual_ids.add(new_id)
                break
        else:
            # If no suitable real allocation is found, create a new one.
            buf = self._thr.allocate(size)
            real_id = self._real_id_counter
            self._real_id_counter += 1

            self._real_allocations[real_id] = self.RealAllocation(buf, set([new_id]))
            self._real_sizes.insert(self.RealSize(size, real_id))

        # Here it would be more appropriate to use buffer.get_sub_region(0, size),
        # but OpenCL does not allow several overlapping subregions to be used in a single kernel
        # for both read and write, which ruins the whole idea.
        # So we are passing full buffers and hope that overlapping Array class takes care of sizes.
        self._virtual_to_real[new_id] = self.VirtualMapping(
            real_id, self._real_allocations[real_id].buffer)

    def _get_buffer(self, id_):
        return self._virtual_to_real[id_].sub_region

    def _free(self, id_, pack=False):
        # Remove the allocation from the dependency lists of its dependencies
        dep_set = self._virtual_allocations[id_].dependencies
        for dep in dep_set:
            self._virtual_allocations[dep].dependencies.remove(id_)

        vtr = self._virtual_to_real[id_]

        # Clear virtual allocation data
        del self._virtual_allocations[id_]
        del self._virtual_to_real[id_]

        if pack:
            self._pack()
        else:
            # Fast and non-optimal free.
            # Remove the virtual allocation from the real allocation,
            # and delete the real allocation if its no longer used by other virtual allocations.
            ra = self._real_allocations[vtr.real_id]
            ra.virtual_ids.remove(id_)
            if len(ra.virtual_ids) == 0:
                del self._real_allocations[vtr.real_id]
                self._real_sizes.remove(self.RealSize(ra.buffer.size, vtr.real_id))

    def _pack(self):
        """
        Full memory re-pack.
        In theory, should find the optimal (with the minimal real allocation size) distribution
        of virtual allocations.
        """

        # Need to synchronize, because we are going to change allocation addresses,
        # and we do not want to free the memory some kernel is reading from.
        self._thr.synchronize()

        # Clear all real allocation data.
        self._real_sizes.clear()
        self._real_allocations = {}
        self._real_id_counter = 0

        va = self._virtual_allocations

        # Sort all virtual allocations by size
        virtual_sizes = sorted(
            [(va[id_].size, id_) for id_ in va],
            key=lambda x: x[0])

        # Application of greedy algorithm for virtual allocations starting from the largest one
        # should give the optimal distribution.
        for size, id_ in reversed(virtual_sizes):
            self._fast_add(id_, size, self._virtual_allocations[id_].dependencies)

    def _statistics(self):

        stats = dict(
            virtual_size_total=0,
            virtual_num=0,
            real_size_total=0,
            real_num=0,
            virtual_sizes=[],
            real_sizes=[])

        for va in self._virtual_allocations.values():
            stats['virtual_size_total'] += va.size
            stats['virtual_num'] += 1
            stats['virtual_sizes'].append(va.size)

        for ra in self._real_allocations.values():
            stats['real_size_total'] += ra.buffer.size
            stats['real_num'] += 1
            stats['real_sizes'].append(ra.buffer.size)

        stats['virtual_sizes'] = sorted(stats['virtual_sizes'])
        stats['real_sizes'] = sorted(stats['real_sizes'])

        return stats
