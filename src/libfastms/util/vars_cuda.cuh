/*
* This file is part of fastms.
*
* Copyright 2014 Evgeny Strekalovskiy <evgeny dot strekalovskiy at in dot tum dot de> (Technical University of Munich)
*
* fastms is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* fastms is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with fastms. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef UTIL_CUDA_VARS_H
#define UTIL_CUDA_VARS_H

#if !defined(DISABLE_CUDA) && defined(__CUDACC__)

#include <algorithm>  // min, max
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "image_access.h"



__device__ __forceinline__ int cuda_x() { return threadIdx.x + blockDim.x * blockIdx.x; }
__device__ __forceinline__ int cuda_y() { return threadIdx.y + blockDim.y * blockIdx.y; }


__device__ __host__ __forceinline__ bool is_active(int x, int y, const Dim2D &dim)
{
	return (x < dim.w && y < dim.h);
}


inline dim3 cuda_block_size(int w, int h)
{
	dim3 block(128, 2, 1);
	block.x = std::min((int)block.x, w);
	block.y = std::min((int)block.y, w);
	return block;
}


inline dim3 cuda_grid_size(dim3 block, int w, int h)
{
	return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);
}



#endif // !defined(DISABLE_CUDA) && defined(__CUDACC__)

#endif // UTIL_CUDA_VARS_H
