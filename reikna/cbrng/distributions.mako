<%def name="distribution_uniform_integer(prefix)">
WITHIN_KERNEL ${ctype} ${prefix}(LOCAL_STATE *state)
{
    ${raw_ctype} non_offset = 0;

    %if max_num % num == 0:
    ${raw_ctype} t = ${raw_func}(state);
    non_offset = t / ${max_num // num};
    %else:
    while(1)
    {
        ${raw_ctype} t = ${raw_func}(state);
        if (t < ${max_num - max_num % num})
        {
            non_offset = t / ${max_num // num};
            break;
        }
    }
    %endif

    return (${ctype})non_offset + (${ctype})${low};
}
</%def>


<%def name="distribution_uniform_float(prefix)">
WITHIN_KERNEL ${ctype} ${prefix}(LOCAL_STATE *state)
{
    ${ctype} normalized = (${ctype})${raw_func}(state) / ${raw_max};
    return normalized * (${size}) + (${low});
}
</%def>


<%def name="normal_bm(prefix)">
WITHIN_KERNEL ${ctype2} ${prefix}(LOCAL_STATE *state)
{
    ${ctype} u1 = ${uniform_float}(state);
    ${ctype} u2 = ${uniform_float}(state);

    ${ctype} ang = ${dtypes.c_constant(2.0 * numpy.pi, dtype)} * u2;
    ${ctype} c_ang = cos(ang);
    ${ctype} s_ang = sin(ang);
    ${ctype} coeff = sqrt(${dtypes.c_constant(-2.0, dtype)} * log(u1)) * (${std});

    return COMPLEX_CTR(${ctype2})(coeff * c_ang + (${mean}), coeff * s_ang + (${mean}));
}
</%def>


<%def name="gamma(prefix)">
WITHIN_KERNEL ${ctype} ${prefix}(LOCAL_STATE *state)
{
    <%
        d = shape - 1. / 3
        c = 1 / numpy.sqrt(9 * d)
    %>

    ${ctype2} rand_normal;
    bool normals_need_regen = true;

    const ${ctype} d = ${dtypes.c_constant(d, dtype)};
    const ${ctype} c = ${dtypes.c_constant(c, dtype)};

    for (;;)
    {
        ${ctype} X, V, U;

        do
        {
            if (normals_need_regen)
            {
                rand_normal = ${normal_bm}(state);
                X = rand_normal.x;
            }
            else
            {
                X = rand_normal.y;
            }

            V = 1.0 + c * X;
        } while (V <= 0.0);

        V = V * V * V;
        U = ${uniform_float}(state);
        if (U < 1.0 - 0.0331 * (X * X) * (X * X)) return (d * V) * (${scale});
        if (log(U) < 0.5 * X * X + d * (1. - V + log(V))) return (d * V) * (${scale});
    }
}
</%def>
