<%def name="insertBaseKernels()">

#ifdef CUDA
// On some platforms nvcc will pick up the NAN constant from math.h
#ifndef NAN
// Using the native CUDA NaN constant
#include <math_constants.h>
%if dtypes.is_double(dtype):
#define NAN CUDART_NAN
%else:
#define NAN CUDART_NAN_F
%endif
#endif
#endif

#define complex_ctr COMPLEX_CTR(${dtypes.ctype(dtype)})
#define conj_transp(a) complex_ctr(-(a).y, (a).x)
#define conj_transp_and_mul(a, b) complex_ctr(-(a).y * (b), (a).x * (b))

typedef ${dtypes.ctype(dtype)} complex_t;
typedef ${dtypes.ctype(dtypes.real_for(dtype))} real_t;


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
    a[5] = ${mul}(w1, a[5]);
    a[6] = conj_transp_and_mul(a[6], direction);
    a[7] = ${mul}(w3, a[7]);
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
    a[5]  = ${mul}(a[5], temp);
    temp  = complex_ctr(w1, direction * w0);
    a[7]  = ${mul}(a[7], temp);
    temp  = complex_ctr(w2, direction * w2);
    a[6]  = ${mul}(a[6], temp);
    a[9]  = ${mul}(a[9], temp);

    a[10] = conj_transp_and_mul(a[10], direction);

    temp  = complex_ctr(-w2, direction * w2);
    a[11] = ${mul}(a[11], temp);
    a[14] = ${mul}(a[14], temp);
    temp  = complex_ctr(w1, direction * w0);
    a[13] = ${mul}(a[13], temp);
    temp  = complex_ctr(-w0, -direction * w1);
    a[15] = ${mul}(a[15], temp);

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
        a[${i + 16}] = ${mul}(a[${i + 16}], temp);
    %endfor

    fftKernel16(a, direction);
    fftKernel16(a + 16, direction);
    bitreverse32(a);
}

// Calculates input and output weights for the Bluestein's algorithm
WITHIN_KERNEL complex_t xweight(int dir_coeff, VSIZE_T pos)
{
    // The modulo of 2 * fft_size_real does not change the result,
    // but greatly improves the precision by keeping the argument of sin()/cos() small.
    return ${polar_unit}(dir_coeff * ${wrap_const(numpy.pi / fft_size_real)} *
        ((pos * pos) % (2 * ${fft_size_real})) );
}

</%def>

<%def name="insertGlobalLoadsNoIf(input, kweights, a_indices, g_indices, pad=False, fft_index_offsets=None)">
    <%
        if fft_index_offsets is None:
            fft_index_offsets = [0] * len(a_indices)
    %>

    %for a_i, g_i, fft_index_offset in zip(a_indices, g_indices, fft_index_offsets):

    %if pad:
        a[${a_i}] = complex_ctr(0, 0);
    %else:
    {
        const VSIZE_T position_in_fft = fft_position_offset + ${g_i};

        a[${a_i}] = ${input.load_combined_idx(input_slices)}(
            fft_index + ${fft_index_offset}, position_in_fft, 0);

        %if pad_in:
        a[${a_i}] = ${mul}(a[${a_i}], xweight(direction, position_in_fft));
        %endif

        %if takes_kweights:
        a[${a_i}] = ${mul}(a[${a_i}],
            ${kweights.load_idx}((1 - direction) / 2, position_in_fft)
            );
        %endif
    }
    %endif

    %endfor
</%def>

<%def name="insertGlobalLoadsOuter(input, kweights, outer_list, num_inner_iter, outer_step, inner_step)">
    %for i in outer_list:
        <%
            loads = lambda indices, pad: insertGlobalLoadsNoIf(input, kweights,
                i * num_inner_iter + numpy.array(indices),
                inner_step * numpy.array(indices),
                pad=pad, fft_index_offsets=[i * outer_step] * len(indices))
            border = fft_size_real // inner_step
        %>
        %if pad_in:
            ${loads(range(border), False)}
            %if fft_size_real % inner_step > 0:
            if (fft_position_offset < ${fft_size_real % inner_step})
            {
                ${loads([border], False)}
            }
            else
            %endif
            {
                ${loads([border], True)}
            }
            ${loads(range(border + 1, num_inner_iter), True)}
        %else:
            ${loads(range(num_inner_iter), False)}
        %endif
    %endfor
</%def>

<%def name="insertGlobalStoresNoIf(output, kweights, a_indices, g_indices, fft_index_offsets=None)">
    <%
        if fft_index_offsets is None:
            fft_index_offsets = [0] * len(a_indices)
    %>

    %for a_i, g_i, fft_index_offset in zip(a_indices, g_indices, fft_index_offsets):
    {
        const VSIZE_T position_in_fft = fft_position_offset + ${g_i};

        %if unpad_out:
        a[${a_i}] = ${mul}(a[${a_i}], xweight(-direction, position_in_fft));
        %endif

        ${output.store_combined_idx(output_slices)}(
            fft_index + ${fft_index_offset}, position_in_fft, 0,
            ${cdivs}(a[${a_i}], norm_coeff));
    }
    %endfor
</%def>

<%def name="insertGlobalStoresOuter(output, kweights, outer_list, num_inner_iter, outer_step, inner_step)">
    %for i in outer_list:
        <%
            stores = lambda indices: insertGlobalStoresNoIf(output, kweights,
                i * num_inner_iter + numpy.array(indices),
                inner_step * numpy.array(indices),
                fft_index_offsets=[i * outer_step] * len(indices))
            border = fft_size_real // inner_step
        %>
        %if unpad_out:
            ${stores(range(border))}
            %if fft_size_real % inner_step > 0:
            if (fft_position_offset < ${fft_size_real % inner_step})
            {
                ${stores([border])}
            }
            %endif
        %else:
            ${stores(range(num_inner_iter))}
        %endif
    %endfor
</%def>


<%def name="insertGlobalLoadsAndTranspose(input, kweights, threads_per_xform, xforms_per_workgroup, radix, mem_coalesce_width)">

    <%
        local_size = threads_per_xform * xforms_per_workgroup
        xforms_remainder = outer_batch % xforms_per_workgroup
    %>

    %if threads_per_xform >= mem_coalesce_width:
        <%
            loads = lambda indices, pad: insertGlobalLoadsNoIf(input, kweights,
                indices, [j * threads_per_xform for j in indices], pad=pad)
            border = fft_size_real // threads_per_xform
        %>

        const VSIZE_T thread_in_xform = thread_id % ${threads_per_xform};
        const VSIZE_T xform_in_wg = thread_id / ${threads_per_xform};
        const VSIZE_T fft_index = group_id * ${xforms_per_workgroup} + xform_in_wg;
        const VSIZE_T fft_position_offset = thread_in_xform;

        %if xforms_remainder != 0:
        if((group_id < num_groups - 1) || (xform_in_wg < ${xforms_remainder}))
        %endif
        {
        %if pad_in:
            ${loads(range(border), False)}
            %if fft_size_real % threads_per_xform > 0:
            if (fft_position_offset < ${fft_size_real % threads_per_xform})
            {
                ${loads([border], False)}
            }
            else
            %endif
            {
                ${loads([border], True)}
            }
            ${loads(range(border + 1, radix), True)}
        %else:
            ${loads(range(radix), False)}
        %endif
        }
    %elif fft_size >= mem_coalesce_width:
        <%
            num_inner_iter = fft_size // mem_coalesce_width
            num_outer_iter = xforms_per_workgroup // (local_size // mem_coalesce_width)
        %>

        const VSIZE_T thread_in_coalesce = thread_id % ${mem_coalesce_width};
        const VSIZE_T coalesce_in_wg = thread_id / ${mem_coalesce_width};
        const VSIZE_T fft_index = group_id * ${xforms_per_workgroup} + coalesce_in_wg;
        const VSIZE_T fft_position_offset = thread_in_coalesce;

        const VSIZE_T thread_in_xform = thread_id % ${threads_per_xform};
        const VSIZE_T xform_in_wg = thread_id / ${threads_per_xform};

        %if xforms_remainder != 0:
        if(group_id == num_groups - 1)
        {
            ${insertGlobalLoadsOuter(input, kweights,
                range(xforms_remainder // (local_size // mem_coalesce_width)),
                num_inner_iter, (local_size // mem_coalesce_width), mem_coalesce_width)}

            %if xforms_remainder % (local_size // mem_coalesce_width) > 0:
            if (coalesce_in_wg < ${xforms_remainder % (local_size // mem_coalesce_width)})
            {
                ${insertGlobalLoadsOuter(input, kweights,
                    [xforms_remainder // (local_size // mem_coalesce_width)],
                    num_inner_iter, (local_size // mem_coalesce_width), mem_coalesce_width)}
            }
            %endif
        }
        else
        %endif
        {
            ${insertGlobalLoadsOuter(input, kweights, range(num_outer_iter),
                num_inner_iter, (local_size // mem_coalesce_width), mem_coalesce_width)}
        }

        {
        const VSIZE_T lmem_store_idx = coalesce_in_wg * ${fft_size + threads_per_xform} + thread_in_coalesce;
        const VSIZE_T lmem_load_idx = xform_in_wg * ${fft_size + threads_per_xform} + thread_in_xform;

        %for comp in ('x', 'y'):
            %for i in range(num_outer_iter):
                %for j in range(num_inner_iter):
                    lmem[lmem_store_idx + ${j * mem_coalesce_width + \
                        i * (local_size // mem_coalesce_width) * (fft_size + threads_per_xform)}] =
                        a[${i * num_inner_iter + j}].${comp};
                %endfor
            %endfor
            LOCAL_BARRIER;

            %for i in range(radix):
                a[${i}].${comp} = lmem[lmem_load_idx + ${i * threads_per_xform}];
            %endfor

            LOCAL_BARRIER;
        %endfor
        }

    %else:
        const VSIZE_T thread_in_fft = thread_id % ${fft_size};
        const VSIZE_T fft_in_wg = thread_id / ${fft_size};
        const VSIZE_T fft_index = group_id * ${xforms_per_workgroup} + fft_in_wg;
        const VSIZE_T fft_position_offset = thread_in_fft;

        const VSIZE_T thread_in_xform = thread_id % ${threads_per_xform};
        const VSIZE_T xform_in_wg = thread_id / ${threads_per_xform};

        <%
            loads = lambda indices, pad: insertGlobalLoadsNoIf(input, kweights,
                indices, [0] * len(indices), pad=pad,
                fft_index_offsets = [i * local_size // fft_size for i in indices])
            border = xforms_remainder // (local_size // fft_size)
        %>

        %if xforms_remainder != 0:
        if(group_id == num_groups - 1)
        {
            %if pad_in:
            if (fft_position_offset < ${fft_size_real})
            {
            %endif
                ${loads(range(border), False)}
                %if xforms_remainder % (local_size // fft_size) > 0:
                if (fft_in_wg < ${xforms_remainder % (local_size // fft_size)})
                {
                    ${loads([border], False)}
                }
                %endif
            %if pad_in:
            }
            else
            {
                ${loads(range(radix), True)}
            }
            %endif
        }
        else
        %endif
        {
            %if pad_in:
            if (fft_position_offset < ${fft_size_real})
            {
            %endif
                ${loads(range(radix), False)}
            %if pad_in:
            }
            else
            {
                ${loads(range(radix), True)}
            }
            %endif
        }

        {
        const VSIZE_T lmem_store_idx = fft_in_wg * ${fft_size + threads_per_xform} + thread_in_fft;
        const VSIZE_T lmem_load_idx = xform_in_wg * ${fft_size + threads_per_xform} + thread_in_xform;

        %for comp in ('x', 'y'):
            %for i in range(radix):
                lmem[lmem_store_idx + ${i * (local_size // fft_size) * (fft_size + threads_per_xform)}] = a[${i}].${comp};
            %endfor
            LOCAL_BARRIER;

            %for i in range(radix):
                a[${i}].${comp} = lmem[lmem_load_idx + ${i * threads_per_xform}];
            %endfor

            LOCAL_BARRIER;
        %endfor
        }
    %endif
</%def>

<%def name="insertGlobalStoresAndTranspose(output, kweights, max_radix, radix, threads_per_xform, xforms_per_workgroup, mem_coalesce_width)">

    <%
        local_size = threads_per_xform * xforms_per_workgroup
        num_iter = max_radix // radix
        xforms_remainder = outer_batch % xforms_per_workgroup
    %>

    %if threads_per_xform >= mem_coalesce_width:
        <%
            stores = lambda indices: insertGlobalStoresNoIf(output, kweights,
                [((j % num_iter) * radix + (j // num_iter)) for j in indices],
                [j * threads_per_xform for j in indices])
            border = fft_size_real // threads_per_xform
        %>

        %if xforms_remainder != 0:
        if((group_id < num_groups - 1) || xform_in_wg < ${xforms_remainder})
        %endif
        {
        %if unpad_out:
            ${stores(range(border))}
            %if fft_size_real % threads_per_xform > 0:
            if (fft_position_offset < ${fft_size_real % threads_per_xform})
            {
                ${stores([border])}
            }
            %endif
        %else:
            ${stores(range(max_radix))}
        %endif
        }

    %elif fft_size >= mem_coalesce_width:
        <%
            num_inner_iter = fft_size // mem_coalesce_width
            num_outer_iter = xforms_per_workgroup // (local_size // mem_coalesce_width)
        %>

        {
        const VSIZE_T lmem_load_idx = xform_in_wg * ${fft_size + threads_per_xform} + thread_in_xform;
        const VSIZE_T lmem_store_idx = coalesce_in_wg * ${fft_size + threads_per_xform} + thread_in_coalesce;

        %for comp in ('x', 'y'):
            %for i in range(max_radix):
                <%
                    j = i % num_iter
                    k = i // num_iter
                    ind = j * radix + k
                %>
                lmem[lmem_load_idx + ${i * threads_per_xform}] = a[${ind}].${comp};
            %endfor
            LOCAL_BARRIER;

            %for i in range(num_outer_iter):
                %for j in range(num_inner_iter):
                    a[${i*num_inner_iter + j}].${comp} = lmem[
                        lmem_store_idx + ${j * mem_coalesce_width + \
                        i * (local_size // mem_coalesce_width) * (fft_size + threads_per_xform)}];
                %endfor
            %endfor
            LOCAL_BARRIER;
        %endfor
        }

        <%
            stores = lambda indices: insertGlobalStoresOuter(output, kweights,
                indices, num_inner_iter, (local_size // mem_coalesce_width), mem_coalesce_width)
            border = xforms_remainder // (local_size // mem_coalesce_width)
        %>

        %if xforms_remainder != 0:
        if(group_id == num_groups - 1)
        {
            ${stores(range(border))}
            %if xforms_remainder % (local_size // mem_coalesce_width) > 0:
            if (coalesce_in_wg < ${xforms_remainder % (local_size // mem_coalesce_width)})
            {
                ${stores([border])}
            }
            %endif
        }
        else
        %endif
        {
            ${stores(range(num_outer_iter))}
        }

    %else:
        {
        const VSIZE_T lmem_load_idx = xform_in_wg * ${fft_size + threads_per_xform} + thread_in_xform;
        const VSIZE_T lmem_store_idx = fft_in_wg * ${fft_size + threads_per_xform} + thread_in_fft;

        %for comp in ('x', 'y'):
            %for i in range(max_radix):
                <%
                    j = i % num_iter
                    k = i // num_iter
                    ind = j * radix + k
                %>
                lmem[lmem_load_idx + ${i * threads_per_xform}] = a[${ind}].${comp};
            %endfor
            LOCAL_BARRIER;

            %for i in range(max_radix):
                a[${i}].${comp} = lmem[lmem_store_idx + ${i * (local_size // fft_size) * (fft_size + threads_per_xform)}];
            %endfor
            LOCAL_BARRIER;
        %endfor
        }

        <%
            stores = lambda indices: insertGlobalStoresNoIf(output, kweights,
                indices, [0] * len(indices),
                fft_index_offsets = [i * local_size // fft_size for i in indices])
            border = xforms_remainder // (local_size // fft_size)
        %>

        %if xforms_remainder != 0:
        if(group_id == num_groups - 1)
        {
            %if unpad_out:
            if (fft_position_offset < ${fft_size_real})
            {
            %endif
                ${stores(range(border))}
                %if xforms_remainder % (local_size // fft_size) > 0:
                if (fft_in_wg < ${xforms_remainder % (local_size // fft_size)})
                {
                    ${stores([border])}
                }
                %endif
            %if unpad_out:
            }
            %endif
        }
        else
        %endif
        {
            %if unpad_out:
            if (fft_position_offset < ${fft_size_real})
            {
            %endif
                ${stores(range(radix))}
            %if unpad_out:
            }
            %endif
        }
    %endif
</%def>

<%def name="insertTwiddleKernel(radix, num_iter, radix_prev, data_len, threads_per_xform)">
    // Twiddle kernel
    %for z in range(num_iter):
    {
        const VSIZE_T angf = (${z * threads_per_xform} + thread_in_xform) / ${radix_prev};
        %for k in range(1, radix):
            <% ind = z * radix + k %>
            a[${ind}] = ${mul}(
                a[${ind}],
                ${polar_unit}(${wrap_const(2 * numpy.pi * k / data_len)} * angf * direction));
        %endfor
    }
    %endfor
</%def>

<%def name="insertLocalTranspose(num_iter, radix, radix_next, radix_prev, radix_curr, threads_per_xform, threads_req, offset, mid_pad, xforms_per_workgroup)">

    <%
        threads_req_next = fft_size // radix_next
        inter_block_hnum = max(radix_prev // threads_per_xform, 1)
        inter_block_hstride = threads_per_xform
        vert_width = max(threads_per_xform // radix_prev, 1)
        vert_width = min(vert_width, radix)
        vert_num = radix // vert_width
        vert_stride = (fft_size // radix + offset) * vert_width
        iter = max(threads_req_next // threads_per_xform, 1)
        intra_block_hstride = max(threads_per_xform // (radix_prev * radix), 1)
        intra_block_hstride *= radix_prev

        stride = threads_req // radix_next

        radix_curr = radix_prev * radix
        incr = (threads_req + offset) * radix + mid_pad
    %>
    {
    %if xforms_per_workgroup == 1:
        const VSIZE_T lmem_store_idx = thread_in_xform;
    %else:
        const VSIZE_T lmem_store_idx = xform_in_wg * ${(threads_req + offset) * radix + mid_pad} + thread_in_xform;
    %endif

        VSIZE_T i, j;
    %if radix_curr < threads_per_xform:
        j = (thread_in_xform % ${radix_curr}) / ${radix_prev};
        i = (thread_in_xform / ${radix_curr}) * ${radix_prev} + thread_in_xform % ${radix_prev};
    %else:
        j = thread_in_xform / ${radix_prev};
        i = thread_in_xform % ${radix_prev};
    %endif

    %if xforms_per_workgroup > 1:
        i = xform_in_wg * ${incr} + i;
    %endif

        const VSIZE_T lmem_load_idx = j * ${threads_req + offset} + i;

    %for comp in ('x', 'y'):
        %for z in range(num_iter):
            %for k in range(radix):
                <% index = k * (threads_req + offset) + z * threads_per_xform %>
                lmem[lmem_store_idx + ${index}] = a[${z * radix + k}].${comp};
            %endfor
        %endfor

        LOCAL_BARRIER;

        %for i in range(iter):
            <%
                ii = i // (inter_block_hnum * vert_num)
                zz = i % (inter_block_hnum * vert_num)
                jj = zz % inter_block_hnum
                kk = zz // inter_block_hnum
            %>

            %for z in range(radix_next):
                <% st = kk * vert_stride + jj * inter_block_hstride + ii * intra_block_hstride + z * stride %>
                a[${i * radix_next + z}].${comp} = lmem[lmem_load_idx + ${st}];
            %endfor
        %endfor

        LOCAL_BARRIER;
    %endfor

    }

</%def>

<%def name="insertVariableDefinitions(inverse, lmem_size, temp_array_size)">

    %if lmem_size > 0:
        LOCAL_MEM real_t lmem[${lmem_size}];
    %endif

    complex_t a[${temp_array_size}];

    // This does not change the performance, but
    // makes it much easier to catch bugs in addressing.
    %for i in range(temp_array_size):
    a[${i}] = complex_ctr(NAN, NAN);
    %endfor

    const VSIZE_T thread_id = virtual_local_id(0);
    const VSIZE_T group_id = virtual_group_id(0);

    ## makes it easier to use it inside other definitions
    %if reverse_direction:
    const int direction = (1 - ${inverse} * 2);
    %else:
    const int direction = -(1 - ${inverse} * 2);
    %endif

    const unsigned int norm_coeff = direction == 1 ? ${fft_size if normalize else 1} : 1;
</%def>

<%def name="fft_local(kernel_declaration, *args)">

<%
    if takes_kweights:
        output, input, kweights, inverse = args
    else:
        output, input, inverse = args
        kweights = None

    max_radix = radix_arr[0]
    num_radix = len(radix_arr)

    xforms_remainder = outer_batch % xforms_per_workgroup
%>

${insertBaseKernels()}

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    ${insertVariableDefinitions(inverse, lmem_size, max_radix)}

    %if xforms_remainder != 0:
    const VSIZE_T num_groups = virtual_num_groups(0);
    %endif

    ${insertGlobalLoadsAndTranspose(input, kweights, threads_per_xform, xforms_per_workgroup, max_radix,
        min_mem_coalesce_width)}

    <%
        radix_prev = 1
        data_len = fft_size
    %>

    %for r in range(num_radix):
        <%
            num_iter = radix_arr[0] // radix_arr[r]
            threads_req = fft_size // radix_arr[r]
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
            ${insertLocalTranspose(num_iter, radix_arr[r], radix_arr[r+1], radix_prev, radix_curr, threads_per_xform, threads_req, offset, mid_pad, xforms_per_workgroup)}
            <%
                radix_prev = radix_curr
                data_len = data_len // radix_arr[r]
            %>
        %endif
    %endfor

    ${insertGlobalStoresAndTranspose(output, kweights, max_radix, radix_arr[num_radix - 1], threads_per_xform,
        xforms_per_workgroup, min_mem_coalesce_width)}
}

</%def>

<%def name="fft_global(kernel_declaration, *args)">

<%
    if takes_kweights:
        output, input, kweights, direction = args
    else:
        output, input, direction = args
%>

${insertBaseKernels()}

<%
    num_iter = radix1 // radix2
    groups_per_xform = min_blocks(stride_in, local_batch)
%>

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    ${insertVariableDefinitions(direction, lmem_size, radix1)}

    const VSIZE_T xform_global = group_id / ${groups_per_xform};
    const VSIZE_T group_in_xform = group_id % ${groups_per_xform};
    const VSIZE_T xform_local = thread_id / ${local_batch};
    const VSIZE_T thread_in_xform = thread_id % ${local_batch};

    const VSIZE_T position_in_stride_in = thread_in_xform + group_in_xform * ${local_batch};

    ## Technically, if we're not inside a stride-in, we could return right now.
    ## But some OpenCL implementations have undefined behavior if local barriers
    ## are not reached by every single thread.
    ## Therefore we just skip global memory operations if we're not in a stride-in.
    %if stride_in % local_batch != 0:
    const int inside_stride_in = (position_in_stride_in < ${stride_in});
    %else:
    const int inside_stride_in = 1;
    %endif

    // Load data

    if (inside_stride_in)
    {
    %for j in range(radix1):
    {
        const VSIZE_T stride_in_number = xform_local + ${j * radix2};
        const VSIZE_T position = position_in_stride_in + ${stride_in} * stride_in_number;

        const VSIZE_T position_in_fft = position / ${inner_batch};
        const VSIZE_T position_in_inner_batch = position % ${inner_batch};

        %if pad_in:
        if (position_in_fft < ${fft_size_real})
        {
        %endif
            a[${j}] = ${input.load_combined_idx(input_slices)}(
                xform_global, position_in_fft, position_in_inner_batch);
        %if pad_in:
        }
        else
        {
            a[${j}] = complex_ctr(0, 0);
        }
        %endif

        %if pad_in:
            const complex_t xw = xweight(direction, position_in_fft);
            a[${j}] = ${mul}(a[${j}], xw);
        %endif

        %if takes_kweights:
        a[${j}] = ${mul}(a[${j}],
            ${kweights.load_idx}((1 - direction) / 2, position_in_fft)
            );
        %endif
    }
    %endfor
    }

    fftKernel${radix1}(a, direction);

    %if radix2 > 1:
        // Twiddle
        %for k in range(1, radix1):
        {
            const real_t ang = ${wrap_const(2 * numpy.pi * k / radix)} * xform_local * direction;
            const complex_t w = ${polar_unit}(ang);
            a[${k}] = ${mul}(a[${k}], w);
        }
        %endfor

        // Shuffle
        {
        const VSIZE_T lmem_store_idx = thread_id;
        const VSIZE_T lmem_load_idx = xform_local * ${local_size * num_iter} + thread_in_xform;

        %for comp in ('x', 'y'):
            %for k in range(radix1):
                lmem[lmem_store_idx + ${k * local_size}] = a[${k}].${comp};
            %endfor
            LOCAL_BARRIER;

            %for k in range(num_iter):
                %for t in range(radix2):
                    a[${k * radix2 + t}].${comp} =
                        lmem[lmem_load_idx + ${t * local_batch + k * local_size}];
                %endfor
            %endfor
            LOCAL_BARRIER;
        %endfor
        }

        %for j in range(num_iter):
            fftKernel${radix2}(a + ${j * radix2}, direction);
        %endfor
    %endif

    %if not last_pass:
    // Twiddle
    {
        const VSIZE_T l = (group_in_xform * ${local_batch} + thread_in_xform) / ${stride_out};
        const VSIZE_T k = xform_local * ${radix1 // radix2};
        const real_t ang1 = ${wrap_const(2 * numpy.pi / curr_size)} * l * direction;
        %for t in range(radix1):
        {
            const real_t ang = ang1 * (k + ${(t % radix2) * radix1 + (t // radix2)});
            const complex_t w = ${polar_unit}(ang);
            a[${t}] = ${mul}(a[${t}], w);
        }
        %endfor
    }
    %endif

    // Store Data
    %if stride_out == 1:
    {
        const VSIZE_T lmem_store_idx = thread_in_xform * ${radix + 1} + xform_local * ${radix1 // radix2};
        const VSIZE_T lmem_load_idx = (thread_id / ${radix}) * ${radix + 1} + thread_id % ${radix};

        %for comp in ('x', 'y'):
            %for i in range(radix1 // radix2):
                %for j in range(radix2):
                    lmem[lmem_store_idx + ${i + j * radix1}] = a[${i * radix2 + j}].${comp};
                %endfor
            %endfor
            LOCAL_BARRIER;

            %if local_size >= radix:
                %for i in range(radix1):
                    a[${i}].${comp} = lmem[lmem_load_idx + ${i * (radix + 1) * (local_size // radix)}];
                %endfor
            %else:
                <%
                    inner_iter = radix // local_size
                    outer_iter = radix1 // inner_iter
                %>
                %for i in range(outer_iter):
                    %for j in range(inner_iter):
                        a[${i * inner_iter + j}].${comp} = lmem[lmem_load_idx + ${j * local_size + i * (radix + 1)}];
                    %endfor
                %endfor
            %endif
            LOCAL_BARRIER;
        %endfor
    }

    if (inside_stride_in)
    {
        const VSIZE_T position_in_stride_out = (group_in_xform * ${local_batch}) % ${stride_out};
        const VSIZE_T stride_out_number = (group_in_xform * ${local_batch}) / ${stride_out};

        %for k in range(radix1):
        {
            const VSIZE_T position = stride_out_number * ${stride} + ${k * local_size} +
                position_in_stride_out + thread_id;
            const VSIZE_T position_in_fft = position / ${inner_batch};
            const VSIZE_T position_in_inner_batch = position % ${inner_batch};

            %if unpad_out:
            const complex_t xw = xweight(-direction, position_in_fft);
            a[${k}] = ${mul}(a[${k}], xw);

            if (position_in_fft < ${fft_size_real})
            %endif
                ${output.store_combined_idx(output_slices)}(
                    xform_global, position_in_fft, position_in_inner_batch,
                    ${cdivs}(a[${k}], norm_coeff));
        }
        %endfor
    }
    %else:
    if (inside_stride_in)
    {
        const VSIZE_T position_in_stride_out = (group_in_xform * ${local_batch} + thread_in_xform) % ${stride_out};
        const VSIZE_T stride_out_number = (group_in_xform * ${local_batch} + thread_in_xform) / ${stride_out};

        %for k in range(radix1):
        {
            const VSIZE_T position = ${((k % radix2) * radix1 + (k // radix2)) * stride_out} +
                ${stride_out} * (
                    stride_out_number * ${radix} +
                    xform_local * ${radix1 // radix2}) +
                position_in_stride_out;

            const VSIZE_T position_in_fft = position / ${inner_batch};
            const VSIZE_T position_in_inner_batch = position % ${inner_batch};

            %if unpad_out:
            const complex_t xw = xweight(-direction, position_in_fft);
            a[${k}] = ${mul}(a[${k}], xw);

            if (position_in_fft < ${fft_size_real})
            %endif
                ${output.store_combined_idx(output_slices)}(
                    xform_global, position_in_fft, position_in_inner_batch,
                    ${cdivs}(a[${k}], norm_coeff));
        }
        %endfor
    }
    %endif
}

</%def>
