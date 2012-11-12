<%def name="prelude()">

%if ctx_fast_math:
#define CTX_FAST_MATH
%endif

%if api == 'cuda':
    #define CUDA
    // taken from pycuda._cluda
    #define LOCAL_BARRIER __syncthreads()

    #define WITHIN_KERNEL __device__
    #define KERNEL extern "C" __global__
    #define GLOBAL_MEM /* empty */
    #define LOCAL_MEM __shared__
    #define LOCAL_MEM_DYNAMIC extern __shared__
    #define LOCAL_MEM_ARG /* empty */
    #define INLINE __forceinline__

    <%
        dimnames = ['x', 'y', 'z']
    %>

    WITHIN_KERNEL int get_local_id(int dim)
    {
    %for n in range(3):
        if(dim == ${n}) return threadIdx.${dimnames[n]};
    %endfor
        return 0;
    }

    WITHIN_KERNEL int get_group_id(int dim)
    {
    %for n in range(3):
        if(dim == ${n}) return blockIdx.${dimnames[n]};
    %endfor
        return 0;
    }

    WITHIN_KERNEL int get_local_size(int dim)
    {
    %for n in range(3):
        if(dim == ${n}) return blockDim.${dimnames[n]};
    %endfor
        return 1;
    }

    WITHIN_KERNEL int get_num_groups(int dim)
    {
    %for n in range(3):
        if(dim == ${n}) return gridDim.${dimnames[n]};
    %endfor
        return 1;
    }

    WITHIN_KERNEL int get_global_size(int dim)
    {
        return get_num_groups(dim) * get_local_size(dim);
    }

    WITHIN_KERNEL int get_global_id(int dim)
    {
        return get_local_id(dim) + get_group_id(dim) * get_local_size(dim);
    }

%elif api == 'ocl':
    // taken from pyopencl._cluda
    #define LOCAL_BARRIER barrier(CLK_LOCAL_MEM_FENCE)

    // 'static' helps to avoid the "no previous prototype for function" warning
    #if PYOPENCL_CL_VERSION >= 0x1020
    #define WITHIN_KERNEL static
    #else
    #define WITHIN_KERNEL
    #endif

    #define KERNEL __kernel
    #define GLOBAL_MEM __global
    #define LOCAL_MEM __local
    #define LOCAL_MEM_DYNAMIC __local
    #define LOCAL_MEM_ARG __local
    #define INLINE inline

    #if defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
    #elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64: enable
    #endif

%endif

%if api == 'cuda':
    #define COMPLEX_CTR(T) make_##T
%elif api == 'ocl':
    #define COMPLEX_CTR(T) (T)
%endif

## These operators are supported by OpenCL
%if api == 'cuda':
%for tp in ('float2', 'double2'):
    WITHIN_KERNEL ${tp} operator+(${tp} a, ${tp} b)
    {
        return COMPLEX_CTR(${tp})(a.x + b.x, a.y + b.y);
    }
    WITHIN_KERNEL ${tp} operator-(${tp} a, ${tp} b)
    {
        return COMPLEX_CTR(${tp})(a.x - b.x, a.y - b.y);
    }
    WITHIN_KERNEL ${tp} operator+(${tp} a) { return a; }
    WITHIN_KERNEL ${tp} operator-(${tp} a) { return COMPLEX_CTR(${tp})(-a.x, -a.y); }
%endfor
%endif

</%def>

<%def name="mul(name, out_dtype, dtype1, dtype2)">
WITHIN_KERNEL ${dtypes.ctype(out_dtype)} ${name}(
    ${dtypes.ctype(dtype1)} a, ${dtypes.ctype(dtype2)} b)
{
<%
    c1 = dtypes.is_complex(dtype1)
    c2 = dtypes.is_complex(dtype2)
    if dtypes.is_complex(out_dtype):
        out_ctr = dtypes.complex_ctr(out_dtype)

    if not c1 and not c2:
        result = "a * b"
    elif c1 and not c2:
        result = out_ctr + "(a.x * b, a.y * b)"
    elif not c1 and c2:
        result = out_ctr + "(b.x * a, b.y * a)"
    else:
        result = out_ctr + "(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)"
%>
    return ${result};
}
</%def>

<%def name="div(name, out_dtype, dtype1, dtype2)">
WITHIN_KERNEL ${dtypes.ctype(out_dtype)} ${name}(
    ${dtypes.ctype(dtype1)} a, ${dtypes.ctype(dtype2)} b)
{
<%
    c1 = dtypes.is_complex(dtype1)
    c2 = dtypes.is_complex(dtype2)
    if dtypes.is_complex(out_dtype):
        out_ctr = dtypes.complex_ctr(out_dtype)

    if not c1 and not c2:
        result = "a / b"
    elif c1 and not c2:
        result = out_ctr + "(a.x / b, a.y / b)"
    elif not c1 and c2:
        result = out_ctr + "(a * b.x / (b.x * b.x + b.y * b.y), -a * b.y / (b.x * b.x + b.y * b.y))"
    else:
        result = out_ctr + "((a.x * b.x + a.y * b.y) / (b.x * b.x + b.y * b.y), " + \
            "(-a.x * b.y + a.y * b.x) / (b.x * b.x + b.y * b.y))"
%>
    return ${result};
}
</%def>

<%def name="cast(name, out_dtype, in_dtype)">
WITHIN_KERNEL ${dtypes.ctype(out_dtype)} ${name}(${dtypes.ctype(in_dtype)} x)
{
<%
    if dtypes.is_complex(out_dtype) and not dtypes.is_complex(in_dtype):
        result = dtypes.complex_ctr(out_dtype) + "(x, 0)"
    elif dtypes.is_complex(out_dtype) == dtypes.is_complex(in_dtype):
        result = "(" + dtypes.ctype(out_dtype) + ")x"
    else:
        raise NotImplementedError("Cast from " + str(in_dtype) + " to " + str(out_dtype) +
            " is not supported")
%>
    return ${result};
}
</%def>
