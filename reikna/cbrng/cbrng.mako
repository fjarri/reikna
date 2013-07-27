<%def name="cbrng(kernel_declaration, new_counters, randoms, old_counters)">

<%
    uint32 = dtypes.ctype(numpy.uint32)
    uint64 = dtypes.ctype(numpy.uint64)
    rng_dtype = numpy.uint32 if rng_params.bitness == 32 else numpy.uint64
    rng_ctype = dtypes.ctype(rng_dtype)
    output_len = rng_params.words
    output_len_words = output_len * (1 if rng_params.bitness == 32 else 2)
    randoms_per_call = distribution.randoms_per_call
%>

typedef struct _CBRNG_ARGUMENT
{
    ${rng_ctype} v[${output_len}];
} CBRNG_ARGUMENT;

typedef struct _LOCAL_STATE
{
    CBRNG_ARGUMENT counter;
    union {
        CBRNG_ARGUMENT buffer;
        ${uint32} buffer_word[${output_len_words}];
    };
    int buffer_word_cursor;
} LOCAL_STATE;


WITHIN_KERNEL void bump_counter(LOCAL_STATE *state)
{
    %for i in range(output_len-1, 0, -1):
    state->counter.v[${i}] += 1;
    if (state->counter.v[${i}] == 0)
    {
    %endfor
    state->counter.v[0] += 1;
    %for i in range(output_len-1, 0, -1):
    }
    %endfor
}


WITHIN_KERNEL void refill_buffer(LOCAL_STATE *state)
{
    state->buffer = ${rng}(virtual_global_flat_id(), state->counter);
    bump_counter(state);
    state->buffer_word_cursor = 0;
}


WITHIN_KERNEL ${uint32} get_raw_uint32(LOCAL_STATE *state)
{
    if (state->buffer_word_cursor == ${output_len_words})
    {
        refill_buffer(state);
    }

    int cur = state->buffer_word_cursor;
    state->buffer_word_cursor += 1;
    return state->buffer_word[cur];
}


WITHIN_KERNEL ${uint64} get_raw_uint64(LOCAL_STATE *state)
{
    if (state->buffer_word_cursor >= ${output_len_words} - 1)
    {
        refill_buffer(state);
    }

    int cur = state->buffer_word_cursor;
    state->buffer_word_cursor += 2;
    %if rng_params.bitness == 64:
    return state->buffer.v[cur / 2];
    %else:
    ${uint32} hi = state->buffer_word[cur];
    ${uint32} lo = state->buffer_word[cur+1];
    return ((${uint64})hi << 32) + (${uint64})lo;
    %endif
}


${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    const int idx = virtual_global_flat_id();

    LOCAL_STATE local_state;
    %for i in range(output_len):
    local_state.counter.v[${i}] = ${old_counters.load_combined_idx(counters_slices)}(idx, ${i});
    %endfor
    refill_buffer(&local_state);

    %if randoms_per_call == 1 and dtypes.is_complex(randoms.dtype):
    for (int i = 0; i < ${batch}; i++)
    {
        ${randoms.store_combined_idx(randoms_slices)}(
            i, idx,
            COMPLEX_CTR(${randoms.ctype})(
                ${distribution}(&local_state),
                ${distribution}(&local_state)));
    }
    %elif randoms_per_call == 1 or (randoms_per_call == 2 and dtypes.is_complex(randoms.dtype)):
    for (int i = 0; i < ${batch}; i++)
    {
        ${randoms.store_combined_idx(randoms_slices)}(
            i, idx, ${distribution}(&local_state));
    }
    %elif randoms_per_call == 2:
    for (int i = 0; i < ${batch // 2}; i++)
    {
        ${randoms.ctype}2 r = ${distribution}(&local_state);
        ${randoms.store_combined_idx(randoms_slices)}(i * 2, idx, r.x);
        ${randoms.store_combined_idx(randoms_slices)}(i * 2 + 1, idx, r.y);
    }
    %elif batch % 2 != 0:
    {
        ${randoms.ctype}2 r = ${distribution}(&local_state);
        ${randoms.store_combined_idx(randoms_slices)}(i * 2, idx, r.x);
    }
    %else:
    <%
        raise NotImplementedError()
    %>
    %endif

    if (local_state.buffer_word_cursor > 0 && local_state.buffer_word_cursor < ${output_len})
        bump_counter(&local_state);

    %for i in range(output_len):
    ${new_counters.store_combined_idx(counters_slices)}(idx, ${i}, local_state.counter.v[${i}]);
    %endfor
}
</%def>
