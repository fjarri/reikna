KERNEL void dummy(${signature})
{
    int idx = LID_0 + LSIZE_0 * GID_0;
    if (idx < ${size})
    {
    	${ctype.A} a = ${load.A}(idx);
    	${ctype.B} b = ${load.B}(idx);
    	${ctype.C} c = a + ${func.mul(dtype.coeff, dtype.B)}(${param.coeff}, b);
    	${store.C}(idx, c);
    }
}
