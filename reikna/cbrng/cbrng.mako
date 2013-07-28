<%def name="cbrng(kernel_declaration, counters, randoms)">
<%
    bijection = sampler.bijection
    randoms_per_call = sampler.randoms_per_call
%>

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    const int idx = virtual_global_id(0);

    ${bijection.module}KEY key = ${keygen.module}(idx);

    ${bijection.module}COUNTER counter;
    %for i in range(bijection.counter_words):
    counter.v[${i}] = ${counters.load_combined_idx(counters_slices)}(idx, ${i});
    %endfor
    ${bijection.module}STATE state = ${bijection.module}make_state(key, counter);

    ${sampler.module}RESULT result;
    for (int i = 0; i < ${batch // randoms_per_call}; i++)
    {
        result = ${sampler.module}(&state);
        %for j in range(randoms_per_call):
        ${randoms.store_combined_idx(randoms_slices)}(
            i * ${randoms_per_call} + ${j}, idx, result.v[${j}]);
        %endfor
    }
    %if batch % randoms_per_call != 0:
    result = ${sampler.module}(&state);
    %for j in range(batch % randoms_per_call):
    ${randoms.store_combined_idx(randoms_slices)}(
        ${batch - batch % randoms_per_call + j}, idx, result.v[${j}]);
    %endfor
    %endif

    ${bijection.module}COUNTER next_ctr = ${bijection.module}get_next_unused_counter(state);
    %for i in range(bijection.counter_words):
    ${counters.store_combined_idx(counters_slices)}(idx, ${i}, state.counter.v[${i}]);
    %endfor
}
</%def>
