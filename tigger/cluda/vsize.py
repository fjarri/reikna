import numpy
from tigger.helpers import *

TEMPLATE = template_for(__file__)


def find_local_size(device_params, max_workgroup_size, dims):
    """
    Simple algorithm to find local_size with given limitations
    """

    # shortcut for CPU devices
    if device_params.warp_size == 1:
        return [max_workgroup_size] + [1] * (dims - 1)

    # trying to find local size with dimensions which are multiples of warp_size
    unit = device_params.warp_size
    max_dims = device_params.max_work_item_sizes

    sizes = [1]
    for i in xrange(1, min_blocks(max_workgroup_size, unit)):
        if i * unit <= max_workgroup_size:
            sizes.append(i * unit)

    total_size = lambda indices: product([sizes[i] for i in indices])
    result_indices = [0] * dims
    pos = 0

    while True:
        result_indices[pos] += 1

        if result_indices[pos] < len(sizes) and total_size(result_indices) <= max_workgroup_size:
            if sizes[result_indices[pos]] > max_dims[pos]:
                result_indices[pos] -= 1

                if pos == len(result_indices):
                    break

                pos += 1
        else:
            result_indices[pos] -= 1
            break

    return tuple([sizes[i] for i in result_indices])


def render_stub_vsize_funcs():
    return TEMPLATE.get_def('stub_funcs').render()


class VirtualSizes:

    def __init__(self, device_params, max_workgroup_size, global_size, local_size):
        self.params = device_params

        self.global_size = wrap_in_tuple(global_size)

        if local_size is None:
            local_size = find_local_size(device_params, max_workgroup_size, len(global_size))

        self.local_size = wrap_in_tuple(local_size)

        if len(self.global_size) != len(self.local_size):
            raise ValueError("Global/local work sizes have differing dimensions")
        if len(self.global_size) > 3:
            raise ValueError("Virtual sizes are supported for 1D to 3D grids only")

        self.naive_bounding_grid = [min_blocks(gs, ls)
            for gs, ls in zip(self.global_size, self.local_size)]

        if product(self.local_size) > self.params.max_work_group_size:
            raise ValueError("Number of work items is too high")
        if product(self.naive_bounding_grid) > product(self.params.max_num_groups):
            raise ValueError("Number of work groups is too high")

        self.grid_parts = self.get_rearranged_grid(self.naive_bounding_grid)
        gdims = len(self.params.max_num_groups)
        self.grid = [product([row[i] for row in self.grid_parts])
            for i in range(gdims)]
        self.k_local_size = list(self.local_size) + [1] * (gdims - len(self.local_size))
        self.k_global_size = [l * g for l, g in zip(self.k_local_size, self.grid)]

    def get_rearranged_grid(self, grid):
        # This algorithm can be made much better, but at the moment we have a simple implementation
        # The guidelines are:
        # 1) the order of array elements should be preserved (so it is like a reshape() operation)
        # 2) the overhead of empty threads is considered negligible
        #    (usually it will be true because it will be hidden by global memory latency)
        # 3) assuming len(grid) <= 3
        max_grid = self.params.max_num_groups
        if len(grid) == 1:
            return self.get_rearranged_grid_1d(grid, max_grid)
        elif len(grid) == 2:
            return self.get_rearranged_grid_2d(grid, max_grid)
        elif len(grid) == 3:
            return self.get_rearranged_grid_3d(grid, max_grid)
        else:
            raise NotImplementedError()

    def get_rearranged_grid_2d(self, grid, max_grid):
        # A dumb algorithm which relies on 1d version
        grid1 = self.get_rearranged_grid_1d([grid[0]], max_grid)

        # trying to fit in remaining dimensions, to decrease the number of operations
        # in get_group_id()
        new_max_grid = [mg // g1d for mg, g1d in zip(max_grid, grid1[0])]
        if product(new_max_grid[1:]) >= grid[1]:
            grid2 = self.get_rearranged_grid_1d([grid[1]], new_max_grid[1:])
            grid2 = [[1] + grid2[0]]
        else:
            grid2 = self.get_rearranged_grid_1d([grid[1]], new_max_grid)

        return grid1 + grid2

    def get_rearranged_grid_3d(self, grid, max_grid):
        # same dumb algorithm, but relying on 2d version
        grid1 = self.get_rearranged_grid_2d(grid[:2], max_grid)

        # trying to fit in remaining dimensions, to decrease the number of operations
        # in get_group_id()
        new_max_grid = [mg // g1 // g2 for mg, g1, g2 in zip(max_grid, grid1[0], grid1[1])]
        if len(new_max_grid) > 2 and product(new_max_grid[2:]) >= grid[2]:
            grid2 = self.get_rearranged_grid_1d([grid[2]], new_max_grid[2:])
            grid2 = [[1, 1] + grid2[0]]
        elif len(new_max_grid) > 1 and product(new_max_grid[1:]) >= grid[2]:
            grid2 = self.get_rearranged_grid_1d([grid[2]], new_max_grid[1:])
            grid2 = [[1] + grid2[0]]
        else:
            grid2 = self.get_rearranged_grid_1d([grid[2]], new_max_grid)

        return grid1 + grid2

    def get_rearranged_grid_1d(self, grid, max_grid):
        g = grid[0]
        if g <= max_grid[0]:
            return [[g] + [1] * (len(max_grid) - 1)]

        # for cases when max_grid was passed from higher dimension methods,
        # and there is no space left
        if max_grid[0] == 0:
            return [[1] + self.get_rearranged_grid_1d([g], max_grid[1:])[0]]

        # first check if we can split the number
        fs = factors(g)
        for f, div in reversed(fs):
            if f <= max_grid[0]:
                break

        if f != 1 and div <= product(max_grid[1:]):
            res = self.get_rearranged_grid_1d([div], max_grid[1:])
            return [[f] + res[0]]

        # fallback: will have some empty threads
        # picking factor equal to the power of 2 to make id calculations faster
        # Starting from low powers in order to minimize the number of resulting empty threads
        for p in range(1, log2(max_grid[-1]) + 1):
            f = 2 ** p
            remainder = min_blocks(g, f)
            if remainder <= product(max_grid[:-1]):
                res = self.get_rearranged_grid_1d([remainder], max_grid[:-1])
                return [res[0] + [f]]

        # fallback 2: couldn't find suitable 2**n factor, so using the maximum size
        f = max_grid[0]
        remainder = min_blocks(g, f)
        res = self.get_rearranged_grid_1d([remainder], max_grid[1:])
        return [[f] + res[0]]

    def render_vsize_funcs(self):
        return TEMPLATE.get_def('normal_funcs').render(vs=self, product=product)

    def get_call_sizes(self):
        return tuple(self.k_global_size), tuple(self.k_local_size)
