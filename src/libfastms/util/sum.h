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

#ifndef UTIL_SUM_H
#define UTIL_SUM_H


#include "image_access.h"



template<typename T>
class KahanSummation
{
public:
	KahanSummation()
	{
		clear();
	}
	void clear()
	{
		s = T(0);
		addend_small = T(0);
	}
	void add(T a)
	{
		T addend_compensated = a - addend_small;
		T s_compensated = s + addend_compensated;
		addend_small = (s_compensated-s) - addend_compensated;
		s = s_compensated;
	}
	T sum()
	{
		return s;
	}
private:
	T s;
	T addend_small;
};


template<typename T>
T cpu_sum(const void *a, size_t num)
{
	const T *a_T = (const T*)a;
	KahanSummation<T> summation;
	for (size_t i = 0; i < num; i++)
	{
		summation.add(a_T[i]);
	}
	return summation.sum();
}


template<typename T>
T cpu_sum2d(const void *a, size_t pitch, int w, int h)
{
	KahanSummation<T> summation;
	for (int y = 0; y < h; y++)
	{
		const T *y_ptr = (const T*)((char*)a + pitch * y);
		for (int x = 0; x < w; x++)
		{
			summation.add(y_ptr[x]);
		}
	}
	return summation.sum();
}


template<typename TImageAccess>
typename TImageAccess::elem_t cpu_sum_reduce(TImageAccess aux_reduce)
{
	typedef typename TImageAccess::elem_t real;

	real sum = real(0);
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp parallel default(none) shared(sum) firstprivate(aux_reduce)
	{
#endif
	const Dim2D &dim2d = aux_reduce.dim().dim2d();
	KahanSummation<real> summation;
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp for
#endif
	for (int y = 0; y < dim2d.h; y++)
	{
		for (int x = 0; x < dim2d.w; x++)
		{
			summation.add(aux_reduce.get(x, y, 0));
		}
	}
	real sub_sum = summation.sum();
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp atomic
#endif
	sum += sub_sum;
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    }
#endif
    return sum;
}



#endif // UTIL_SUM_H
