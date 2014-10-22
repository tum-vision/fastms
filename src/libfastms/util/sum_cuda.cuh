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

#ifndef UTIL_CUDA_SUM_H
#define UTIL_CUDA_SUM_H

#if !defined(DISABLE_CUDA) && defined(__CUDACC__)

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>



namespace
{

template<typename T>
T cuda_sum_on_cpu(const T *a, int num)
{
	T *host_copy = new T[num];
	cudaMemcpy(host_copy, a, num * sizeof(T), cudaMemcpyDeviceToHost);
	T sum = cpu_sum(host_copy, num);
	delete[] host_copy;
	return sum;
}

template<typename T>
T cuda_sum_with_existing_handle(const void *a, int num, const cublasHandle_t &handle)
{
	// sum on cpu for general T (except T=float and T=double, see specializations below)
	return cuda_sum_on_cpu<T>(a, num);
}
template<> __forceinline__ float cuda_sum_with_existing_handle<float>(const void *a, int num, const cublasHandle_t &handle)
{
	float sum = 0.0f;
	cublasSasum(handle, num, (float*)a, 1, &sum); // 1: sum every element (not only every 2nd, or every 3rd etc.)
	return sum;
}
template<> __forceinline__ double cuda_sum_with_existing_handle<double>(const void *a, int num, const cublasHandle_t &handle)
{
	double sum = 0.0;
	cublasDasum(handle, num, (double*)a, 1, &sum);
	return sum;
}

} // namespace



template<typename real>
class DeviceSummator
{
public:
	DeviceSummator() : cublas_handle(NULL) {}
	void alloc()
	{
		cublasCreate(&cublas_handle);
	}
	void free()
	{
	    cublasDestroy(cublas_handle);
	}
	real sum(const void *data, size_t num_bytes)
	{
		size_t elem_size = sizeof(real);
		if (num_bytes % elem_size == 0)
		{
			size_t effective_num_elem = num_bytes / elem_size;
			return cuda_sum_with_existing_handle<real>(data, effective_num_elem, cublas_handle);
		}
		else
		{
			std::cerr << "ERROR: DeviceSummator::sum(): array size (" << num_bytes << " B) not multiple of element size (" << elem_size << " B)." << std::endl;
			return real(0);
		}
	}
private:
	cublasHandle_t cublas_handle;
};


#endif // !defined(DISABLE_CUDA) && defined(__CUDACC__)

#endif // UTIL_CUDA_SUM_H
