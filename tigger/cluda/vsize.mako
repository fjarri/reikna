WITHIN_KERNEL int virtual_local_id(int dim)
{
    return get_local_id(dim);
}

WITHIN_KERNEL int virtual_local_size(int dim)
{
    return get_local_size(dim);
}

WITHIN_KERNEL int virtual_group_size(int dim)
{
%for dim in xrange(len(vs.naive_bounding_grid)):
    if (dim == ${dim}) return ${vs.naive_bounding_grid[dim] if dim < len(vs.naive_bounding_grid) else 1};
%endfor
    return 1;
}

WITHIN_KERNEL int virtual_group_id(int dim)
{
%for dim in xrange(len(vs.naive_bounding_grid)):
    if (dim == ${dim}) {
        int res = 0;
    %for rdim in xrange(len(vs.params.max_grid_sizes)):
    <%
        widths = [p[rdim] for p in vs.grid_parts]
        width_greater = product(widths[:dim+1])
        max_group_id = product(widths)
        width_lower = product(widths[:dim])
        multiplier = product(vs.grid_parts[dim][:rdim])
    %>
    ## if width_lower >= max_group_id, or width_greater == 1,
    ## the whole expression is zero
    %if max_group_id > 1 and width_lower < max_group_id and width_greater > 1:
        res += ((get_group_id(${rdim})
            ## if width_greater >= max_group_id, the '%' will have no effect
            ${('%' + str(width_greater)) if width_greater < max_group_id else ''}
            ) / ${width_lower}) * ${multiplier};
    %endif
    %endfor
        return res;
    }
%endfor
    return 0;
}

WITHIN_KERNEL int virtual_global_size(int dim)
{
%for dim in xrange(len(vs.naive_bounding_grid)):
    if(dim == ${dim}) return ${vs.global_size[dim] if dim < len(vs.global_size) else 1};
%endfor
    return 1;
}

WITHIN_KERNEL int virtual_global_id(int dim)
{
    return virtual_local_id(dim) + virtual_group_id(dim) * virtual_local_size(dim);
}

WITHIN_KERNEL int virtual_global_flat_size()
{
    return virtual_global_size(0) * virtual_global_size(1) * virtual_global_size(2);
}

WITHIN_KERNEL int virtual_global_flat_id()
{
    <%
    def get_expr(dims):
        if dims == 1:
            return "virtual_global_id(0)"
        else:
            return "{prev_expr} + virtual_global_id({i}) * {w}".format(
                prev_expr=get_expr(dims - 1), i=dims-1,
                w=product(vs.global_size[:dims-1]))
    %>
    return ${get_expr(len(vs.global_size))};
}

WITHIN_KERNEL bool virtual_skip_threads()
{
    if(
    %for i in xrange(len(vs.global_size)):
        %if vs.global_size[i] % vs.local_size[i] != 0:
        virtual_global_id(${i}) > ${vs.global_size[i]} - 1 ||
        %endif
    %endfor
        false
    ) return true;

    return false;
}

WITHIN_KERNEL bool virtual_skip_workgroups()
{
    if(
    %for i in xrange(len(vs.naive_bounding_grid)):
        %if vs.naive_bounding_grid[i] < product(vs.grid_parts[i]):
        virtual_group_id(${i}) > ${vs.naive_bounding_grid[i]} - 1 ||
        %endif
    %endfor
        false
    ) return true;

    return false;
}

#define VIRTUAL_SKIP_THREADS if(virtual_skip_workgroups() || virtual_skip_threads()) return;
