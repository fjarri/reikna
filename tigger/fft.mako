<%def name="insertBaseKernels()">

## TODO: replace by intrinsincs if necessary

#ifndef mad24
#define mad24(x, y, z) ((x) * (y) + (z))
#endif

#ifndef mad
#define mad(x, y, z) ((x) * (y) + (z))
#endif

// integer multiplication
#ifndef mul24
#ifdef __mul24
#define mul24(x, y) __mul24(x, y)
#endif
#ifndef __mul24
#define mul24(x, y) ((x) * (y))
#endif
#endif

#ifdef sincosf
#define complex_exp(res, ang) ${"sincos" + ("" if dtypes.is_double(basis.dtype) else "f")}(ang, &((res).y), &((res).x))
#endif
#ifndef sincosf
#define complex_exp(res, ang) (res).x = cos(ang); (res).y = sin(ang)
#endif

#define complex_ctr COMPLEX_CTR(${dtypes.ctype(basis.dtype)})
#define complex_mul ${func.mul(basis.dtype, basis.dtype)}
#define complex_div_scalar ${func.div(basis.dtype, dtypes.real_for(basis.dtype))}
#define conj(a) complex_ctr((a).x, -(a).y)
#define conj_transp(a) complex_ctr(-(a).y, (a).x)
#define conj_transp_and_mul(a, b) complex_ctr(-(a).y * (b), (a).x * (b))

typedef ${dtypes.ctype(basis.dtype)} complex_t;
typedef ${dtypes.ctype(dtypes.real_for(basis.dtype))} real_t;


WITHIN_KERNEL void swap(complex_t *a, complex_t *b)
{
    complex_t c = *a;
    *a = *b;
    *b = c;
}

// shifts the sequence (a1, a2, a3, a4, a5) transforming it to
// (a5, a1, a2, a3, a4)
WITHIN_KERNEL void shift32(
    complex_t *a1, complex_t *a2, complex_t *a3, complex_t *a4, complex_t *a5)
{
    complex_t c1, c2;
    c1 = *a2;
    *a2 = *a1;
    c2 = *a3;
    *a3 = c1;
    c1 = *a4;
    *a4 = c2;
    c2 = *a5;
    *a5 = c1;
    *a1 = c2;
}

WITHIN_KERNEL void _fftKernel2(complex_t *a)
{
    complex_t c = a[0];
    a[0] = c + a[1];
    a[1] = c - a[1];
}
#define fftKernel2(a, direction) _fftKernel2(a)

WITHIN_KERNEL void _fftKernel2S(complex_t *d1, complex_t *d2)
{
    complex_t c = *d1;
    *d1 = c + *d2;
    *d2 = c - *d2;
}
#define fftKernel2S(d1, d2, direction) _fftKernel2S(d1, d2)

WITHIN_KERNEL void fftKernel4(complex_t *a, const int direction)
{
    fftKernel2S(a + 0, a + 2, direction);
    fftKernel2S(a + 1, a + 3, direction);
    fftKernel2S(a + 0, a + 1, direction);
    a[3] = conj_transp_and_mul(a[3], direction);
    fftKernel2S(a + 2, a + 3, direction);
    swap(a + 1, a + 2);
}

WITHIN_KERNEL void fftKernel4s(complex_t *a0, complex_t *a1,
    complex_t *a2, complex_t *a3, const int direction)
{
    fftKernel2S(a0, a2, direction);
    fftKernel2S(a1, a3, direction);
    fftKernel2S(a0, a1, direction);
    *a3 = conj_transp_and_mul(*a3, direction);
    fftKernel2S(a2, a3, direction);
    swap(a1, a2);
}

WITHIN_KERNEL void bitreverse8(complex_t *a)
{
    swap(a + 1, a + 4);
    swap(a + 3, a + 6);
}

WITHIN_KERNEL void fftKernel8(complex_t *a, const int direction)
{
    const complex_t w1  = complex_ctr(
        ${wrap_const(numpy.sin(numpy.pi / 4))},
        ${wrap_const(numpy.sin(numpy.pi / 4))} * direction);
    const complex_t w3  = complex_ctr(
        ${wrap_const(-numpy.sin(numpy.pi / 4))},
        ${wrap_const(numpy.sin(numpy.pi / 4))} * direction);
    fftKernel2S(a + 0, a + 4, direction);
    fftKernel2S(a + 1, a + 5, direction);
    fftKernel2S(a + 2, a + 6, direction);
    fftKernel2S(a + 3, a + 7, direction);
    a[5] = complex_mul(w1, a[5]);
    a[6] = conj_transp_and_mul(a[6], direction);
    a[7] = complex_mul(w3, a[7]);
    fftKernel2S(a + 0, a + 2, direction);
    fftKernel2S(a + 1, a + 3, direction);
    fftKernel2S(a + 4, a + 6, direction);
    fftKernel2S(a + 5, a + 7, direction);
    a[3] = conj_transp_and_mul(a[3], direction);
    a[7] = conj_transp_and_mul(a[7], direction);
    fftKernel2S(a + 0, a + 1, direction);
    fftKernel2S(a + 2, a + 3, direction);
    fftKernel2S(a + 4, a + 5, direction);
    fftKernel2S(a + 6, a + 7, direction);
    bitreverse8(a);
}

WITHIN_KERNEL void bitreverse4x4(complex_t *a)
{
    swap(a + 1, a + 4);
    swap(a + 2, a + 8);
    swap(a + 3, a + 12);
    swap(a + 6, a + 9);
    swap(a + 7, a + 13);
    swap(a + 11, a + 14);
}

WITHIN_KERNEL void fftKernel16(complex_t *a, const int direction)
{
    complex_t temp;
    const real_t w0 = ${wrap_const(numpy.cos(numpy.pi / 8))};
    const real_t w1 = ${wrap_const(numpy.sin(numpy.pi / 8))};
    const real_t w2 = ${wrap_const(numpy.sin(numpy.pi / 4))};
    fftKernel4s(a + 0, a + 4, a + 8,  a + 12, direction);
    fftKernel4s(a + 1, a + 5, a + 9,  a + 13, direction);
    fftKernel4s(a + 2, a + 6, a + 10, a + 14, direction);
    fftKernel4s(a + 3, a + 7, a + 11, a + 15, direction);

    temp  = complex_ctr(w0, direction * w1);
    a[5]  = complex_mul(a[5], temp);
    temp  = complex_ctr(w1, direction * w0);
    a[7]  = complex_mul(a[7], temp);
    temp  = complex_ctr(w2, direction * w2);
    a[6]  = complex_mul(a[6], temp);
    a[9]  = complex_mul(a[9], temp);

    a[10] = conj_transp_and_mul(a[10], direction);

    temp  = complex_ctr(-w2, direction * w2);
    a[11] = complex_mul(a[11], temp);
    a[14] = complex_mul(a[14], temp);
    temp  = complex_ctr(w1, direction * w0);
    a[13] = complex_mul(a[13], temp);
    temp  = complex_ctr(-w0, -direction * w1);
    a[15] = complex_mul(a[15], temp);

    fftKernel4(a, direction);
    fftKernel4(a + 4, direction);
    fftKernel4(a + 8, direction);
    fftKernel4(a + 12, direction);
    bitreverse4x4(a);
}

WITHIN_KERNEL void bitreverse32(complex_t *a)
{
    shift32(a + 1, a + 2, a + 4, a + 8, a + 16);
    shift32(a + 3, a + 6, a + 12, a + 24, a + 17);
    shift32(a + 5, a + 10, a + 20, a + 9, a + 18);
    shift32(a + 7, a + 14, a + 28, a + 25, a + 19);
    shift32(a + 11, a + 22, a + 13, a + 26, a + 21);
    shift32(a + 15, a + 30, a + 29, a + 27, a + 23);
}

WITHIN_KERNEL void fftKernel32(complex_t *a, const int direction)
{
    complex_t temp;
    %for i in range(16):
        fftKernel2S(a + ${i}, a + ${i + 16}, direction);
    %endfor

    %for i in range(1, 16):
        temp = complex_ctr(
            ${wrap_const(numpy.cos(i * numpy.pi / 16))},
            ${wrap_const(numpy.sin(i * numpy.pi / 16))}
        );
        a[${i + 16}] = complex_mul(a[${i + 16}], temp);
    %endfor

    fftKernel16(a, direction);
    fftKernel16(a + 16, direction);
    bitreverse32(a);
}

</%def>

<%def name="insertGlobalLoad(input, a_index, g_index)">
    a[${a_index}] = ${input.load}(${g_index} + global_mem_offset);
</%def>

<%def name="insertGlobalStore(output, a_index, g_index)">
    ${output.store}(${g_index} + global_mem_offset, complex_div_scalar(a[${a_index}], norm_coeff));
</%def>

<%def name="insertGlobalLoadsAndTranspose(input, n, threads_per_xform, xforms_per_workgroup, radix, mem_coalesce_width)">

    <%
        local_size = threads_per_xform * xforms_per_workgroup
        s = outer_batch % xforms_per_workgroup
    %>

    %if threads_per_xform >= mem_coalesce_width:
        %if xforms_per_workgroup > 1:
            ii = thread_id % ${threads_per_xform};
            jj = thread_id / ${threads_per_xform};

            if(${s} == 0 || (group_id < num_groups - 1) || (jj < ${s}))
            {
                global_mem_offset = mad24(mad24(group_id, ${xforms_per_workgroup}, jj), ${n}, ii);
                %for i in range(radix):
                    ${insertGlobalLoad(input, i, i * threads_per_xform)}
                %endfor
            }
        %else:
            ii = thread_id;
            global_mem_offset = mad24(group_id, ${n}, ii);
            %for i in range(radix):
                ${insertGlobalLoad(input, i, i * threads_per_xform)}
            %endfor
        %endif

    %elif n >= mem_coalesce_width:
        <%
            num_inner_iter = n / mem_coalesce_width
            num_outer_iter = xforms_per_workgroup / (local_size / mem_coalesce_width)
        %>

        ii = thread_id % ${mem_coalesce_width};
        jj = thread_id / ${mem_coalesce_width};
        lmem_store_index = mad24(jj, ${n + threads_per_xform}, ii);

        global_mem_offset = mad24(mad24(group_id, ${xforms_per_workgroup}, jj), ${n}, ii);

        if((group_id == num_groups - 1) && ${s} != 0)
        {
        %for i in range(num_outer_iter):
            if(jj < ${s})
            {
            %for j in range(num_inner_iter):
                ${insertGlobalLoad(input, i * num_inner_iter + j, \
                    j * mem_coalesce_width + i * (local_size / mem_coalesce_width) * n)}
            %endfor
            }
            %if i != num_outer_iter - 1:
                jj += ${local_size / mem_coalesce_width};
            %endif
        %endfor
        }
        else
        {
        %for i in range(num_outer_iter):
            %for j in range(num_inner_iter):
                ${insertGlobalLoad(input, i * num_inner_iter + j, \
                    j * mem_coalesce_width + i * (local_size / mem_coalesce_width) * n)}
            %endfor
        %endfor
        }

        ii = thread_id % ${threads_per_xform};
        jj = thread_id / ${threads_per_xform};
        lmem_load_index = mad24(jj, ${n + threads_per_xform}, ii);

        %for comp in ('x', 'y'):
            %for i in range(num_outer_iter):
                %for j in range(num_inner_iter):
                    lmem[lmem_store_index + ${j * mem_coalesce_width + \
                        i * (local_size / mem_coalesce_width) * (n + threads_per_xform)}] =
                        a[${i * num_inner_iter + j}].${comp};
                %endfor
            %endfor
            LOCAL_BARRIER;

            %for i in range(radix):
                a[${i}].${comp} = lmem[lmem_load_index + ${i * threads_per_xform}];
            %endfor
            LOCAL_BARRIER;
        %endfor
    %else:
        global_mem_offset = mad24(group_id, ${n * xforms_per_workgroup}, thread_id);
        ii = thread_id % ${n};
        jj = thread_id / ${n};
        lmem_store_index = mad24(jj, ${n + threads_per_xform}, ii);

        if((group_id == num_groups - 1) && ${s} != 0)
        {
        %for i in range(radix):
            if(jj < ${s})
            {
                ${insertGlobalLoad(input, i, i * local_size)}
            }
            %if i != radix - 1:
                jj += ${local_size / n};
            %endif
        %endfor
        }
        else
        {
        %for i in range(radix):
            ${insertGlobalLoad(input, i, i*local_size)}
        %endfor
        }

        %if threads_per_xform > 1:
            ii = thread_id % ${threads_per_xform};
            jj = thread_id / ${threads_per_xform};
            lmem_load_index = mad24(jj, ${n + threads_per_xform}, ii);
        %else:
            ii = 0;
            jj = thread_id;
            lmem_load_index = mul24(jj, ${n + threads_per_xform});
        %endif

        %for comp in ('x', 'y'):
            %for i in range(radix):
                lmem[lmem_store_index + ${i * (local_size / n) * (n + threads_per_xform)}] = a[${i}].${comp};
            %endfor
            LOCAL_BARRIER;

            %for i in range(radix):
                a[${i}].${comp} = lmem[lmem_load_index + ${i * threads_per_xform}];
            %endfor
            LOCAL_BARRIER;
        %endfor
    %endif
</%def>

<%def name="insertGlobalStoresAndTranspose(output, n, max_radix, radix, threads_per_xform, xforms_per_workgroup, mem_coalesce_width)">

    <%
        local_size = threads_per_xform * xforms_per_workgroup
        num_iter = max_radix / radix
        s = outer_batch % xforms_per_workgroup
    %>

    %if threads_per_xform >= mem_coalesce_width:
        %if xforms_per_workgroup > 1:
            if(${s} == 0 || (group_id < num_groups - 1) || (jj < ${s}))
            {
        %endif

        %for i in range(max_radix):
            <%
                j = i % num_iter
                k = i / num_iter
                ind = j * radix + k
            %>
            ${insertGlobalStore(output, ind, i * threads_per_xform)}
        %endfor

        %if xforms_per_workgroup > 1:
            }
        %endif

    %elif n >= mem_coalesce_width:
        <%
            num_inner_iter = n / mem_coalesce_width
            num_outer_iter = xforms_per_workgroup / (local_size / mem_coalesce_width)
        %>
        lmem_load_index  = mad24(jj, ${n + threads_per_xform}, ii);
        ii = thread_id % ${mem_coalesce_width};
        jj = thread_id / ${mem_coalesce_width};
        lmem_store_index = mad24(jj, ${n + threads_per_xform}, ii);

        %for comp in ('x', 'y'):
            %for i in range(max_radix):
                <%
                    j = i % num_iter
                    k = i / num_iter
                    ind = j * radix + k
                %>
                lmem[lmem_load_index + ${i * threads_per_xform}] = a[${ind}].${comp};
            %endfor
            LOCAL_BARRIER;

            %for i in range(num_outer_iter):
                %for j in range(num_inner_iter):
                    a[${i*num_inner_iter + j}].${comp} = lmem[lmem_store_index + ${j * mem_coalesce_width + \
                        i * (local_size / mem_coalesce_width) * (n + threads_per_xform)}];
                %endfor
            %endfor
            LOCAL_BARRIER;
        %endfor

        if((group_id == num_groups - 1) && ${s} != 0)
        {
        %for i in range(num_outer_iter):
            if(jj < ${s})
            {
            %for j in range(num_inner_iter):
                ${insertGlobalStore(output, i * num_inner_iter + j, \
                    j * mem_coalesce_width + i * (local_size / mem_coalesce_width) * n)}
            %endfor
            }
            %if i != num_outer_iter - 1:
                jj += ${local_size / mem_coalesce_width};
            %endif
        %endfor
        }
        else
        {
        %for i in range(num_outer_iter):
            %for j in range(num_inner_iter):
                ${insertGlobalStore(output, i * num_inner_iter + j, \
                    j * mem_coalesce_width + i * (local_size / mem_coalesce_width) * n)}
            %endfor
        %endfor
        }
    %else:
        lmem_load_index = mad24(jj, ${n + threads_per_xform}, ii);
        ii = thread_id % ${n};
        jj = thread_id / ${n};
        lmem_store_index = mad24(jj, ${n + threads_per_xform}, ii);

        %for comp in ('x', 'y'):
            %for i in range(max_radix):
                <%
                    j = i % num_iter
                    k = i / num_iter
                    ind = j * radix + k
                %>
                lmem[lmem_load_index + ${i * threads_per_xform}] = a[${ind}].${comp};
            %endfor
            LOCAL_BARRIER;

            %for i in range(max_radix):
                a[${i}].${comp} = lmem[lmem_store_index + ${i * (local_size / n) * (n + threads_per_xform)}];
            %endfor
            LOCAL_BARRIER;
        %endfor

        if((group_id == num_groups - 1) && ${s} != 0)
        {
        %for i in range(max_radix):
            if(jj < ${s})
            {
                ${insertGlobalStore(output, i, i * local_size)}
            }
            %if i != max_radix - 1:
                jj += ${local_size / n};
            %endif
        %endfor
        }
        else
        {
            %for i in range(max_radix):
                ${insertGlobalStore(output, i, i * local_size)}
            %endfor
        }
    %endif
</%def>

<%def name="insertTwiddleKernel(radix, num_iter, radix_prev, data_len, threads_per_xform)">

    {
        // Twiddle kernel
        real_t angf, ang;
        complex_t w;

    %for z in range(num_iter):
        %if z == 0:
            %if radix_prev > 1:
                angf = (real_t)(ii / ${radix_prev});
            %else:
                angf = (real_t)ii;
            %endif
        %else:
            %if radix_prev > 1:
                angf = (real_t)((${z * threads_per_xform} + ii) / ${radix_prev});
            %else:
                ## TODO: find out which conditions are necessary to execute this code
                angf = (real_t)(${z * threads_per_xform} + ii);
            %endif
        %endif

        %for k in range(1, radix):
            <% ind = z * radix + k %>
            ang = ${wrap_const(2 * numpy.pi * k / data_len)} * angf * direction;
            complex_exp(w, ang);
            a[${ind}] = complex_mul(a[${ind}], w);
        %endfor
    %endfor
    }
</%def>

<%def name="insertLocalStores(num_iter, radix, threads_per_xform, threads_req, offset, comp)">
    %for z in range(num_iter):
        %for k in range(radix):
            <% index = k * (threads_req + offset) + z * threads_per_xform %>
            lmem[lmem_store_index + ${index}] = a[${z * radix + k}].${comp};
        %endfor
    %endfor
    LOCAL_BARRIER;
</%def>

<%def name="insertLocalLoads(n, radix, radix_next, radix_prev, radix_curr, threads_per_xform, threads_req, offset, comp)">
    <%
        threads_req_next = n / radix_next
        inter_block_hnum = max(radix_prev / threads_per_xform, 1)
        inter_block_hstride = threads_per_xform
        vert_width = max(threads_per_xform / radix_prev, 1)
        vert_width = min(vert_width, radix)
        vert_num = radix / vert_width
        vert_stride = (n / radix + offset) * vert_width
        iter = max(threads_req_next / threads_per_xform, 1)
        intra_block_hstride = max(threads_per_xform / (radix_prev * radix), 1)
        intra_block_hstride *= radix_prev

        stride = threads_req / radix_next
    %>

    %for i in range(iter):
        <%
            ii = i / (inter_block_hnum * vert_num)
            zz = i % (inter_block_hnum * vert_num)
            jj = zz % inter_block_hnum
            kk = zz / inter_block_hnum
        %>

        %for z in range(radix_next):
            <% st = kk * vert_stride + jj * inter_block_hstride + ii * intra_block_hstride + z * stride %>
            a[${i * radix_next + z}].${comp} = lmem[lmem_load_index + ${st}];
        %endfor
    %endfor
    LOCAL_BARRIER;
</%def>

<%def name="insertLocalLoadIndexArithmetic(radix_prev, radix, threads_req, threads_per_xform, xforms_per_workgroup, offset, mid_pad)">
    <%
        radix_curr = radix_prev * radix
        incr = (threads_req + offset) * radix + mid_pad
    %>

    %if radix_curr < threads_per_xform:
        %if radix_prev == 1:
            j = ii % ${radix_curr};
        %else:
            j = (ii % ${radix_curr}) / ${radix_prev};
        %endif

        %if radix_prev == 1:
            i = ii / ${radix_curr};
        %else:
            i = mad24(ii / ${radix_curr}, ${radix_prev}, ii % ${radix_prev});
        %endif
    %else:
        %if radix_prev == 1:
            j = ii;
        %else:
            j = ii / ${radix_prev};
        %endif

        %if radix_prev == 1:
            i = 0;
        %else:
            i = ii % ${radix_prev};
        %endif
    %endif

    %if xforms_per_workgroup > 1:
        i = mad24(jj, ${incr}, i);
    %endif

    lmem_load_index = mad24(j, ${threads_req + offset}, i);
</%def>

<%def name="insertLocalStoreIndexArithmetic(threads_req, xforms_per_workgroup, radix, offset, mid_pad)">
    %if xforms_per_workgroup == 1:
        lmem_store_index = ii;
    %else:
        lmem_store_index = mad24(jj, ${(threads_req + offset) * radix + mid_pad}, ii);
    %endif
</%def>

<%def name="insertVariableDefinitions(direction, lmem_size, temp_array_size)">

    %if lmem_size > 0:
        LOCAL_MEM real_t lmem[${lmem_size}];
        size_t lmem_store_index, lmem_load_index;
    %endif

    complex_t a[${temp_array_size}];

    int thread_id = get_local_id(0);
    int group_id = get_group_id(0);

    ## makes it easier to use it inside other definitions
    int direction = ${direction};

    int norm_coeff = direction == 1 ? ${norm_coeff if normalize else 1} : 1;
</%def>

<%def name="fft_local(output, input, direction)">

<%
    max_radix = radix_arr[0]
    num_radix = len(radix_arr)
%>

${insertBaseKernels()}

${kernel_definition}
{
    VIRTUAL_SKIP_THREADS;

    ${insertVariableDefinitions(direction, lmem_size, max_radix)}
    int global_mem_offset = 0;
    int ii;
    %if num_radix > 1:
        int i, j;
    %endif

    %if not (threads_per_xform >= min_mem_coalesce_width and xforms_per_workgroup == 1):
        int jj;
        int num_groups = get_num_groups(0);
    %endif

    ${insertGlobalLoadsAndTranspose(input, n, threads_per_xform, xforms_per_workgroup, max_radix,
        min_mem_coalesce_width)}

    <%
        radix_prev = 1
        data_len = n
    %>

    %for r in range(num_radix):
        <%
            num_iter = radix_arr[0] / radix_arr[r]
            threads_req = n / radix_arr[r]
            radix_curr = radix_prev * radix_arr[r]
        %>

        %for i in range(num_iter):
            fftKernel${radix_arr[r]}(a + ${i * radix_arr[r]}, direction);
        %endfor

        %if r < num_radix - 1:
            ${insertTwiddleKernel(radix_arr[r], num_iter, radix_prev, data_len, threads_per_xform)}
            <%
                lMemSize, offset, mid_pad = get_padding(threads_per_xform, radix_prev, threads_req,
                    xforms_per_workgroup, radix_arr[r], local_mem_banks)
            %>
            ${insertLocalStoreIndexArithmetic(threads_req, xforms_per_workgroup, radix_arr[r], offset, mid_pad)}
            ${insertLocalLoadIndexArithmetic(radix_prev, radix_arr[r], threads_req, threads_per_xform, xforms_per_workgroup, offset, mid_pad)}
            %for comp in ('x', 'y'):
                ${insertLocalStores(num_iter, radix_arr[r], threads_per_xform, threads_req, offset, comp)}
                ${insertLocalLoads(n, radix_arr[r], radix_arr[r+1], radix_prev, radix_curr, threads_per_xform, threads_req, offset, comp)}
            %endfor
            <%
                radix_prev = radix_curr
                data_len = data_len / radix_arr[r]
            %>
        %endif
    %endfor

    ${insertGlobalStoresAndTranspose(output, n, max_radix, radix_arr[num_radix - 1], threads_per_xform,
        xforms_per_workgroup, min_mem_coalesce_width)}
}

</%def>

<%def name="fft_global(output, input, direction)">

${insertBaseKernels()}

<%
    num_iter = radix1 / radix2
    input_multiplier = local_size / local_batch
    groups_per_xform = min_blocks(stride_in, local_batch)
%>

${kernel_definition}
{
    VIRTUAL_SKIP_THREADS;

    ${insertVariableDefinitions(direction, lmem_size, radix1)}

    int xform_global = group_id / ${groups_per_xform};
    int group_in_xform = group_id % ${groups_per_xform};
    int thread_in_xform = thread_id % ${local_batch};
    int xform_local = thread_id / ${local_batch};

    int index_in, index_out;
    {
        int tid = mul24(group_in_xform, ${local_batch});

        index_in = mad24(xform_local, ${stride_in},
            thread_in_xform + tid + xform_global * ${n * inner_batch});
        index_out = mad24(tid / ${stride_out}, ${stride},
            tid % ${stride_out} + xform_global * ${n * inner_batch}
        %if stride_out == 1:
            + thread_id);
        %else:
            + mad24(xform_local, ${num_iter * stride_out}, thread_in_xform));
        %endif
    }

    ## Load Data
    %for j in range(radix1):
        a[${j}] = ${input.load}(${j * input_multiplier * stride_in} + index_in);
    %endfor

    fftKernel${radix1}(a, direction);

    %if radix2 > 1:
        ## twiddle
        {
            real_t ang;
            complex_t w;

        %for k in range(1, radix1):
            ## TODO: for some reason, writing it in form
            ## (real_t)${2 * numpy.pi / radix} * (real_t)${k} gives slightly better precision
            ## have to try it with double precision
            ang = ${wrap_const(2 * numpy.pi * k / radix)} * xform_local * direction;
            complex_exp(w, ang);
            a[${k}] = complex_mul(a[${k}], w);
        %endfor
        }

        ## shuffle
        index_in = mad24(xform_local, ${local_size * num_iter}, thread_in_xform);
        lmem_store_index = thread_id;
        lmem_load_index = index_in;

        %for comp in ('x', 'y'):
            %for k in range(radix1):
                lmem[lmem_store_index + ${k * local_size}] = a[${k}].${comp};
            %endfor
            LOCAL_BARRIER;

            %for k in range(num_iter):
                %for t in range(radix2):
                    a[${k * radix2 + t}].${comp} =
                        lmem[lmem_load_index + ${t * local_batch + k * local_size}];
                %endfor
            %endfor
            LOCAL_BARRIER;
        %endfor

        %for j in range(num_iter):
            fftKernel${radix2}(a + ${j * radix2}, direction);
        %endfor
    %endif

    ## twiddle
    %if not last_pass:
    {
        real_t ang1, ang;
        complex_t w;

        int l = (group_in_xform * ${local_batch} + thread_in_xform) / ${stride_out};
        int k = xform_local * ${radix1 / radix2};
        ang1 = ${wrap_const(2 * numpy.pi / curr_n)} * l * direction;
        %for t in range(radix1):
            ang = ang1 * (k + ${(t % radix2) * radix1 + (t / radix2)});
            complex_exp(w, ang);
            a[${t}] = complex_mul(a[${t}], w);
        %endfor
    }
    %endif

    ## Store Data
    %if stride_out == 1:
        lmem_store_index = mad24(thread_in_xform, ${radix + 1}, xform_local * ${radix1 / radix2});
        lmem_load_index = mad24(thread_id / ${radix}, ${radix + 1}, thread_id % ${radix});

        %for comp in ('x', 'y'):
            %for i in range(radix1 / radix2):
                %for j in range(radix2):
                    lmem[lmem_store_index + ${i + j * radix1}] = a[${i * radix2 + j}].${comp};
                %endfor
            %endfor
            LOCAL_BARRIER;

            %if local_size >= radix:
                %for i in range(radix1):
                    a[${i}].${comp} = lmem[lmem_load_index + ${i * (radix + 1) * (local_size / radix)}];
                %endfor
            %else:
                <%
                    inner_iter = radix / local_size
                    outer_iter = radix1 / inner_iter
                %>
                %for i in range(outer_iter):
                    %for j in range(inner_iter):
                        a[${i * inner_iter + j}].${comp} = lmem[lmem_load_index + ${j * local_size + i * (radix + 1)}];
                    %endfor
                %endfor
            %endif
            LOCAL_BARRIER;
        %endfor

        %for k in range(radix1):
            ${output.store}(${k * local_size} + index_out,
                complex_div_scalar(a[${k}], norm_coeff));
        %endfor
    %else:
        %for k in range(radix1):
            ${output.store}(${((k % radix2) * radix1 + (k / radix2)) * stride_out} + index_out,
                complex_div_scalar(a[${k}], norm_coeff));
        %endfor
    %endif
}

</%def>
