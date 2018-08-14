WITHIN_KERNEL VSIZE_T virtual_local_id(unsigned int dim)
{
    %for vdim in range(len(virtual_local_size)):
    if (dim == ${vdim_inverse(vdim)})
    {
        %if virtual_local_size[vdim] == 1:
        ## A shortcut, mostly to make the generated code more readable
        ## (the compiler would probably simplify the full version without any problems).

        return 0;

        %else:

        SIZE_T flat_id =
        %for i, rdim in enumerate(local_groups.real_dims[vdim]):
            get_local_id(${rdim}) * ${local_groups.real_strides[vdim][i]} +
        %endfor
            0;

        ## The modulus operation will not be optimized away by the compiler,
        ## but we can omit it for the major dimension,
        ## knowing that VIRTUAL_SKIP_THREADS will skip redundant threads.
        %if vdim == local_groups.major_vdims[vdim]:
        return (flat_id / ${local_groups.virtual_strides[vdim]});
        %else:
        return (flat_id / ${local_groups.virtual_strides[vdim]}) % ${virtual_local_size[vdim]};
        %endif

        %endif
    }
    %endfor

    return 0;
}

WITHIN_KERNEL VSIZE_T virtual_local_size(unsigned int dim)
{
    %for vdim in range(len(virtual_local_size)):
    if (dim == ${vdim_inverse(vdim)})
    {
        return ${virtual_local_size[vdim]};
    }
    %endfor

    return 1;
}

WITHIN_KERNEL VSIZE_T virtual_group_id(unsigned int dim)
{
    %for vdim in range(len(virtual_grid_size)):
    if (dim == ${vdim_inverse(vdim)})
    {
        %if virtual_grid_size[vdim] == 1:
        ## A shortcut, mostly to make the generated code more readable
        ## (the compiler would probably simplify the full version without any problems).

        return 0;

        %else:

        SIZE_T flat_id =
        %for i, rdim in enumerate(grid_groups.real_dims[vdim]):
            get_group_id(${rdim}) * ${grid_groups.real_strides[vdim][i]} +
        %endfor
            0;

        ## The modulus operation will not be optimized away by the compiler,
        ## but we can omit it for the major dimension,
        ## knowing that VIRTUAL_SKIP_THREADS will skip redundant threads.
        %if vdim == grid_groups.major_vdims[vdim]:
        return (flat_id / ${grid_groups.virtual_strides[vdim]});
        %else:
        return (flat_id / ${grid_groups.virtual_strides[vdim]}) % ${virtual_grid_size[vdim]};
        %endif

        %endif
    }
    %endfor

    return 0;
}

WITHIN_KERNEL VSIZE_T virtual_num_groups(unsigned int dim)
{
    %for vdim in range(len(virtual_grid_size)):
    if (dim == ${vdim_inverse(vdim)})
    {
        return ${virtual_grid_size[vdim]};
    }
    %endfor

    return 1;
}

WITHIN_KERNEL VSIZE_T virtual_global_id(unsigned int dim)
{
    return virtual_local_id(dim) + virtual_group_id(dim) * virtual_local_size(dim);
}

WITHIN_KERNEL VSIZE_T virtual_global_size(unsigned int dim)
{
    %for vdim in range(len(virtual_global_size)):
    if(dim == ${vdim_inverse(vdim)})
    {
        return ${virtual_global_size[vdim]};
    }
    %endfor

    return 1;
}

WITHIN_KERNEL VSIZE_T virtual_global_flat_id()
{
    return
    %for vdim in range(len(virtual_global_size)):
        virtual_global_id(${vdim_inverse(vdim)}) * ${product(virtual_global_size[:vdim])} +
    %endfor
        0;
}

WITHIN_KERNEL VSIZE_T virtual_global_flat_size()
{
    return
    %for vdim in range(len(virtual_global_size)):
        virtual_global_size(${vdim_inverse(vdim)}) *
    %endfor
        1;
}


WITHIN_KERNEL bool virtual_skip_local_threads()
{
    %for threshold, strides in local_groups.skip_thresholds:
    {
        VSIZE_T flat_id =
        %for rdim, stride in strides:
            get_local_id(${rdim}) * ${stride} +
        %endfor
            0;

        if (flat_id >= ${threshold})
            return true;
    }
    %endfor

    return false;
}

WITHIN_KERNEL bool virtual_skip_groups()
{
    %for threshold, strides in grid_groups.skip_thresholds:
    {
        VSIZE_T flat_id =
        %for rdim, stride in strides:
            get_group_id(${rdim}) * ${stride} +
        %endfor
            0;

        if (flat_id >= ${threshold})
            return true;
    }
    %endfor

    return false;
}

WITHIN_KERNEL bool virtual_skip_global_threads()
{
    %for vdim in range(len(virtual_global_size)):
    %if virtual_global_size[vdim] < bounding_global_size[vdim]:
    if (virtual_global_id(${vdim_inverse(vdim)}) >= ${virtual_global_size[vdim]})
        return true;
    %endif
    %endfor

    return false;
}

<%
    virtual_func_names = [
        'num_groups(0)',
        'global_flat_id()',
        'global_flat_size()',
        ]

    mark_used = "; ".join(("(void)(virtual_" + name + ")") for name in virtual_func_names)
%>

## Avoid warnings about unused functions in computation kernels.
## TODO: this will not be necessary if virtual_* functions are included as Modules
#ifndef CUDA
#define MARK_VIRTUAL_FUNCTIONS_AS_USED ${mark_used}
#else
#define MARK_VIRTUAL_FUNCTIONS_AS_USED
#endif

#define VIRTUAL_SKIP_THREADS MARK_VIRTUAL_FUNCTIONS_AS_USED; if(virtual_skip_local_threads() || virtual_skip_groups() || virtual_skip_global_threads()) return
