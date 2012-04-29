<%def name="define_mul_func(name, dtype1, dtype2, out_dtype)">
WITHIN_KERNEL ${dtypes.ctype(out_dtype)} ${name}(
    ${dtypes.ctype(dtype1)} a, ${dtypes.ctype(dtype2)} b)
{
<%
    c1 = dtypes.is_complex(dtype1)
    c2 = dtypes.is_complex(dtype2)
    if dtypes.is_complex(out_dtype):
        out_ctr = dtypes.complex_ctr(out_dtype)

    if not c1 and not c2:
        result = 'a * b'
    elif c1 and not c2:
        result = out_ctr + '(a.x * b, a.y * b)'
    elif not c1 and c2:
        result = out_ctr + '(b.x * a, b.y * a)'
    else:
        result = out_ctr + '(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)'
%>
    return ${result};
}
</%def>

%for name in mul_functions:
<%
    dtype1, dtype2, out_dtype = mul_functions[name]
%>
${define_mul_func(name, dtype1, dtype2, out_dtype)}
%endfor
