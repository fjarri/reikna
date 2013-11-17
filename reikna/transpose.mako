<%def name="transpose(kernel_declaration, output, input)">
<%
	fields = dtypes.flatten_dtype(output.dtype)
	paths = [('.' if len(path) > 0 else '') + '.'.join(path) for path, _ in fields]
	ctypes = [dtypes.ctype(dtype) for _, dtype in fields]
	suffixes = ['_' + '_'.join(path) for path, _ in fields]
%>
${kernel_declaration}
{
	VIRTUAL_SKIP_THREADS;

	// To prevent shared memory bank confilcts:
	// - Load each component into a different array. Since the array size is a
	//   multiple of the number of banks, each thread reads x and y from
	//   the same bank. If a single value_pair array is used, thread n would read
	//   x and y from banks n and n+1, and thread n+8 would read values from the
	//   same banks - causing a bank conflict.
	// - Use (half) warp size + 1 as the x size of the array. This way each row of the
	//   array starts in a different bank - so reading from shared memory
	//   doesn't cause bank conflicts when writing the transpose out to global
	//   memory.

	// We are using separate local memory arrays for each field of a nested dtype:
	// - to avoid bank conflicts for large structs
	// - to avoid crashes on some platforms that cannot handle local memory arrays of structs
	%for ctype, suffix in zip(ctypes, suffixes):
	LOCAL_MEM ${ctype} block${suffix}[(${block_width} + 1) * ${block_width}];
	%endfor

	VSIZE_T lid_x = virtual_local_id(2);
	VSIZE_T lid_y = virtual_local_id(1);

	VSIZE_T gid_x = virtual_group_id(2);
	VSIZE_T gid_y = virtual_group_id(1);

	VSIZE_T batch_num = virtual_global_id(0);

	VSIZE_T xBlock = ${block_width} * gid_x;
	VSIZE_T yBlock = ${block_width} * gid_y;
	VSIZE_T xIndex = xBlock + lid_x;
	VSIZE_T yIndex = yBlock + lid_y;
	VSIZE_T index_block = lid_y * (${block_width} + 1) + lid_x;
	VSIZE_T index_transpose = lid_x * (${block_width} + 1) + lid_y;

	if(xIndex < ${input_width} && yIndex < ${input_height})
	{
		const ${output.ctype} val =
			${input.load_combined_idx(input_slices)}(batch_num, yIndex, xIndex);
		%for path, suffix in zip(paths, suffixes):
		block${suffix}[index_block] = val${path};
		%endfor
	}

	LOCAL_BARRIER;

	if(xBlock + lid_y < ${input_width} && yBlock + lid_x < ${input_height})
	{
		${output.ctype} val;

		%for path, suffix in zip(paths, suffixes):
		val${path} = block${suffix}[index_transpose];
		%endfor

		${output.store_combined_idx(output_slices)}(
			batch_num, xBlock + lid_y, yBlock + lid_x, val);
	}
}

</%def>
