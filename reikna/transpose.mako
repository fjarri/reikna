<%def name="transpose(output, input)">
${kernel_definition}
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
	LOCAL_MEM ${output.ctype} block[(${block_width} + 1) * ${block_width}];

	unsigned int lid_x = virtual_local_id(0);
	unsigned int lid_y = virtual_local_id(1);

	unsigned int gid_x = virtual_group_id(0);
	unsigned int gid_y = virtual_group_id(1);

	//unsigned int batch_num = gid_y / ${blocks_per_matrix};
	//gid_y = gid_y % ${blocks_per_matrix};
	unsigned int batch_num = virtual_global_id(2);

	unsigned int xBlock = ${block_width} * gid_x;
	unsigned int yBlock = ${block_width} * gid_y;
	unsigned int xIndex = xBlock + lid_x;
	unsigned int yIndex = yBlock + lid_y;
	unsigned int index_block = lid_y * (${block_width} + 1) + lid_x;
	unsigned int index_transpose = lid_x * (${block_width} + 1) + lid_y;

	if(xIndex < ${input_width} && yIndex < ${input_height})
		block[index_block] = ${input.load_combined_idx(input_slices)}(
			batch_num, yIndex, xIndex);

	LOCAL_BARRIER;

	if(xBlock + lid_y < ${input_width} && yBlock + lid_x < ${input_height})
		${output.store_combined_idx(output_slices)}(
			batch_num, xBlock + lid_y, yBlock + lid_x,
			block[index_transpose]);
}

</%def>
