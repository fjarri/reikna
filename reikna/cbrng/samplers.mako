<%def name="result_struct(prefix, ctype, randoms_per_call)">
#define ${prefix}Value ${ctype};
#define ${prefix}RANDOMS_PER_CALL ${randoms_per_call}

typedef struct
{
    ${ctype} v[${randoms_per_call}];
} ${prefix}Result;
</%def>


<%def name="uniform_integer(prefix)">
${result_struct(prefix, ctype, 1)}

WITHIN_KERNEL ${prefix}Result ${prefix}sample(${bijection.module}State *state)
{
    ${prefix}Result result;
    ${raw_ctype} non_offset = 0;

    %if max_num % num == 0:
    ${raw_ctype} t = ${bijection.module}${raw_func}(state);
    non_offset = t / ${max_num // num};
    %else:
    while(1)
    {
        ${raw_ctype} t = ${bijection.module}${raw_func}(state);
        if (t < ${max_num - max_num % num})
        {
            non_offset = t / ${max_num // num};
            break;
        }
    }
    %endif

    result.v[0] = (${ctype})non_offset + (${ctype})${low};
    return result;
}
</%def>


<%def name="uniform_float(prefix)">
${result_struct(prefix, ctype, 1)}

WITHIN_KERNEL ${prefix}Result ${prefix}sample(${bijection.module}State *state)
{
    ${prefix}Result result;
    ${ctype} normalized = (${ctype})${bijection.module}${raw_func}(state) / ${raw_max};
    result.v[0] = normalized * (${size}) + (${low});
    return result;
}
</%def>


<%def name="normal_bm(prefix)">
${result_struct(prefix, ctype, 2)}

WITHIN_KERNEL ${prefix}Result ${prefix}sample(${bijection.module}State *state)
{
    ${prefix}Result result;
    ${uf.module}Result r1 = ${uf.module}sample(state);
    ${uf.module}Result r2 = ${uf.module}sample(state);
    ${ctype} u1 = r1.v[0];
    ${ctype} u2 = r2.v[0];

    ${ctype} ang = ${dtypes.c_constant(2.0 * numpy.pi, dtype)} * u2;
    ${ctype} c_ang = cos(ang);
    ${ctype} s_ang = sin(ang);
    ${ctype} coeff = sqrt(${dtypes.c_constant(-2.0, dtype)} * log(u1)) * (${std});

    result.v[0] = coeff * c_ang + (${mean});
    result.v[1] = coeff * s_ang + (${mean});
    return result;
}
</%def>


<%def name="gamma(prefix)">
${result_struct(prefix, ctype, 1)}

WITHIN_KERNEL ${prefix}Result ${prefix}sample(${bijection.module}State *state)
{
    <%
        d = shape - 1. / 3
        c = 1 / numpy.sqrt(9 * d)
    %>

    ${prefix}Result result;
    ${uf.module}Result rand_float;
    ${nbm.module}Result rand_normal;
    bool normals_need_regen = true;

    const ${ctype} d = ${dtypes.c_constant(d, dtype)};
    const ${ctype} c = ${dtypes.c_constant(c, dtype)};

    for (;;)
    {
        ${ctype} X, V;

        do
        {
            if (normals_need_regen)
            {
                rand_normal = ${nbm.module}sample(state);
                X = rand_normal.v[0];
                normals_need_regen = false;
            }
            else
            {
                X = rand_normal.v[1];
                normals_need_regen = true;
            }

            V = 1 + c * X;
        } while (V <= 0);

        V = V * V * V;
        rand_float = ${uf.module}sample(state);
        if (rand_float.v[0] < 1 - 0.0331f * (X * X) * (X * X))
        {
            result.v[0] = (d * V) * (${scale});
            break;
        }
        if (log(rand_float.v[0]) < 0.5f * X * X + d * (1 - V + log(V)))
        {
            result.v[0] = (d * V) * (${scale});
            break;
        }
    }

    return result;
}
</%def>
