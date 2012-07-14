KERNEL void dummy(${signature})
{
    int idx = GLOBAL_INDEX;
    if (idx < {size})
    {
    	${ctype.A} a = ${load.A}(idx);
    	${ctype.B} b = ${load.B}(idx);
    	${ctype.C} c = a + ${mul(dtype.coeff, dtype.B)}(${param.coeff}, b);
    	${store.C}(idx, c);
    }
}
