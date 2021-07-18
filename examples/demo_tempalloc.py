"""
This example illustrates how to manually use a temporary array manager (if you must).
"""

import numpy
from reikna.cluda import dtypes, any_api
from reikna.cluda.tempalloc import ZeroOffsetManager


api = any_api()
thr = api.Thread.create()


def demo_array_dependencies():

    # ZeroOffsetManager attempts to pack temporary allocations
    # in a collection of real allocations with minimal total size.
    # All the virtual allocations start at the beginning of the real allocations.

    # Create a manager that will try to minimize the total size of real allocations
    # every time a temporary allocation occurs, or a temporary array is freed.
    # Note that this may involve re-pointing a temporary array to a different part of memory,
    # so all of the data in it is lost.
    temp_manager = ZeroOffsetManager(thr, pack_on_alloc=True, pack_on_free=True)

    # Alternatively one can pass `False` to these keywords and call `.pack()` manually.
    # This can be useful if a lot of allocations are happening in a specific place at once.

    # Create two arrays that do not depend on each other.
    # This means the manager will allocate a single (200, int32) real array,
    # and point both `a1` and `a2` to its beginning.
    a1 = temp_manager.array(100, numpy.int32)
    a2 = temp_manager.array(200, numpy.int32)

    # You can see that the total size of virtual arrays is 1200,
    # but the total size of real arrays is only 800 (the size of the larger array).
    print("Allocated a1 = (100, int32) and a2 = (200, int32)")
    print(temp_manager._statistics())

    # Now we allocate a dependent array.
    # This means that the real memory `a3` points to cannot intersect with that of `a1`.
    # If we could point temporary arrays at any address within real allocations,
    # we could fit it into the second half of the existing real allocation.
    # But `ZeroOffsetManager` cannot do that, so it has to create another allocation.
    a3 = temp_manager.array(100, numpy.int32, dependencies=[a1])

    print("Allocated a3 = (100, int32) depending on a1")
    print(temp_manager._statistics())

    # Now that we deallocated `a1`, `a3` can now fit in the same real allocation as `a2`,
    # so one of the real allocations will be removed.
    del a1

    print("Freed a1")
    print(temp_manager._statistics())


class MyComputation:

    def __init__(self, temp_manager):
        self.temp_array = temp_manager.array(100, numpy.int32)

        # The magic property containing temporary arrays used
        self.__tempalloc__ = [self.temp_array]

    def __call__(self, array1, array2):
        # a sequence of kernel calls using `self.temp_array` to store some intermediate results
        pass


def demo_object_dependencies():

    temp_manager = ZeroOffsetManager(thr, pack_on_alloc=True, pack_on_free=True)

    # A `MyComputation` instance creates a temporary array for internal usage
    comp = MyComputation(temp_manager)

    print("MyComputation created")
    print(temp_manager._statistics())

    # Create another temporary array whose usage does not intersect with `MyComputation` usage.
    # This means that if `comp` is called, the contents of `a1` may be rewritten.
    a1 = temp_manager.array(100, numpy.int32)

    # It is put in the same real allocation as the temporary array of `comp`.
    print("Allocated a1 = (100, int32)")
    print(temp_manager._statistics())

    # Now let's say we want to put the result of `comp` call somewhere.
    # This means we want to make sure it does not occupy the same memory
    # as any of the temporary arrays in `comp`, so we are passing `comp` as a dependency.
    # It will pick up whatever `comp` declared in its `__tempalloc__` attribute.
    result = temp_manager.array(100, numpy.int32, dependencies=[comp])

    # You can see that a new real allocation was created to host the result.
    print("Allocated result = (100, int32)")
    print(temp_manager._statistics())


if __name__ == '__main__':
    demo_array_dependencies()
    demo_object_dependencies()
