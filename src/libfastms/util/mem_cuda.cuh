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

#ifndef UTIL_MEM_CUDA_H
#define UTIL_MEM_CUDA_H

#if !defined(DISABLE_CUDA) && defined(__CUDACC__)

#include <cuda_runtime.h>



template<typename T> __device__ T* shmem_ptr();
// must specialize here for cuda to have everything in one place
#define SPECIALIZATION(TYPE) template<> __device__ __forceinline__ TYPE* shmem_ptr<TYPE>() { extern __shared__ TYPE shmem_##TYPE[]; return shmem_##TYPE; }
SPECIALIZATION(float)
SPECIALIZATION(double)
#undef SPECIALIZATION


template<typename T>
class ShMemArray
{
public:
	typedef T elem_type;

	// Since all CUDA shared memory arrays begin at the same address,
	// the following constructor should be used only for the first array.
	__device__ ShMemArray(int num_elem) : num_elem(num_elem)
	{
		index_thread = threadIdx.x + blockDim.x * threadIdx.y;
		num_threads = blockDim.x * blockDim.y;
		data_start = shmem_ptr<T>();
		data_end = data_start + num_elem * num_threads;
	}
	// The second and all subsequent arrays should use the following constructor,
	// where the second argument is the previously declared array.
	// Example:
	//   ShMemArray<float> myarray1(10);
	//   ShMemArray<float> myarray2(50, myarray1);
	//   ShMemArray<int> myarray3(2, myarray2);
	template<typename U> __device__ ShMemArray(int num_elem, const ShMemArray<U> &from_array) : num_elem(num_elem)
	{
		index_thread = from_array.index_thread;
		num_threads = from_array.num_threads;
		data_start = (T*)from_array.data_end;
		data_end = data_start + num_elem * num_threads;
	}
	__device__ T& get(int i)
	{
		return data_start[index(i)];
	}
	__device__ const T& get(int i) const
	{
		return data_start[index(i)];
	}
	__host__ static size_t size(int num_elem, dim3 block)
	{
		return block.x * block.y * block.z * sizeof(T) * num_elem;
	}
private:
	__device__ ShMemArray(const ShMemArray<T> &other_array);
	__device__ ShMemArray<T>& operator= (const ShMemArray<T> &other_array);
	__device__ int index(int i) const
	{
		return i + num_elem * index_thread;
		//return index_thread + num_threads * i;   // slower by factor 1.3
	}
	int num_elem;
	int index_thread;
	int num_threads;
	T *data_start;
	T *data_end;
};



#endif // !defined(DISABLE_CUDA) && defined(__CUDACC__)

#endif // UTIL_MEM_CUDA_H
