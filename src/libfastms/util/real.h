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

#ifndef UTIL_REAL_H
#define UTIL_REAL_H

#include <algorithm>  // for min, max
#include <cmath>
#include <cfloat>

#if !defined(DISABLE_CUDA) && defined(__CUDACC__)
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#define FORCEINLINE __forceinline__
#else
#define HOST_DEVICE
#define FORCEINLINE inline
#endif // !defined(DISABLE_CUDA) && defined(__CUDACC__)



#if !defined(DISABLE_CUDA) && defined(__CUDACC__)
template<typename real> bool gpu_supports_real();
template<> inline bool gpu_supports_real<float>() { return true; }
template<> inline bool gpu_supports_real<double>()
{
	// check gpu for double support
	int device = 0;
	cudaGetDevice(&device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	return (prop.major >= 2 || (prop.major == 1 && prop.minor >= 3));
}
#endif // !defined(DISABLE_CUDA) && defined(__CUDACC__)


template<typename real> HOST_DEVICE FORCEINLINE real realabs(real x);
template<> HOST_DEVICE FORCEINLINE float realabs<float> (float x)
{
#ifdef __CUDA_ARCH__
	// device code
	return fabsf(x);
#else
	// host code
	return std::fabs(x);
#endif
}
template<> HOST_DEVICE FORCEINLINE double realabs<double> (double x)
{
#ifdef __CUDACC__
	return fabs(x);
#else
	return std::fabs(x);
#endif
}


template<typename real> HOST_DEVICE FORCEINLINE real realsqrt(real x);
template<> HOST_DEVICE FORCEINLINE float realsqrt<float> (float x)
{
#ifdef __CUDACC__
	return sqrtf(x);
#else
	return std::sqrt(x);
#endif
}
template<> HOST_DEVICE FORCEINLINE double realsqrt<double> (double x)
{
#ifdef __CUDACC__
	return sqrt(x);
#else
	return std::sqrt(x);
#endif
}


template<typename real> HOST_DEVICE FORCEINLINE real realexp(real x);
template<> HOST_DEVICE FORCEINLINE float realexp<float> (float x)
{
#ifdef __CUDACC__
	return expf(x);
#else
	return std::exp(x);
#endif
}
template<> HOST_DEVICE FORCEINLINE double realexp<double> (double x)
{
#ifdef __CUDACC__
	return exp(x);
#else
	return std::exp(x);
#endif
}


template<typename real> HOST_DEVICE FORCEINLINE real reallog(real x);
template<> HOST_DEVICE FORCEINLINE float reallog<float> (float x)
{
#ifdef __CUDACC__
	return logf(x);
#else
	return std::log(x);
#endif
}
template<> HOST_DEVICE FORCEINLINE double reallog<double> (double x)
{
#ifdef __CUDACC__
	return log(x);
#else
	return std::log(x);
#endif
}


template<typename real> HOST_DEVICE FORCEINLINE real realfloor(real x);
template<> HOST_DEVICE FORCEINLINE float realfloor<float> (float x)
{
#ifdef __CUDACC__
	return floorf(x);
#else
	return std::floor(x);
#endif
}
template<> HOST_DEVICE FORCEINLINE double realfloor<double> (double x)
{
#ifdef __CUDACC__
	return floor(x);
#else
	return std::floor(x);
#endif
}


template<typename real> HOST_DEVICE FORCEINLINE real realmax(real x, real y)
{
#ifdef __CUDACC__
	return max(x, y);
#else
	return std::max(x, y);
#endif
}


template<typename real> HOST_DEVICE FORCEINLINE real realmin(real x, real y)
{
#ifdef __CUDACC__
	return min(x, y);
#else
	return std::min(x, y);
#endif
}


template<typename real> HOST_DEVICE FORCEINLINE real realmax();
// divide max values by 4 to adjust for ancient GPUs which couldn't represent all CPU float values
template<> HOST_DEVICE FORCEINLINE float realmax<float> () { return FLT_MAX / 4.0f; }
template<> HOST_DEVICE FORCEINLINE double realmax<double> () { return DBL_MAX / 4.0; }


template<typename Array1D> HOST_DEVICE typename Array1D::elem_type vec_norm_squared(const Array1D &a, int num)
{
	typedef typename Array1D::elem_type real;
	real result = real(0);
	for (int i = 0; i < num; i++)
	{
		real val = a.get(i);
		result += val * val;
	}
	return result;
}


template<typename Array1D> HOST_DEVICE typename Array1D::elem_type vec_norm(const Array1D &a, int num)
{
	return realsqrt(vec_norm_squared<Array1D>(a, num));
}


template<typename Array1D> HOST_DEVICE void vec_scale_eq(Array1D &a, int num, typename Array1D::elem_type mult)
{
	for (int i = 0; i < num; i++)
	{
		a.get(i) *= mult;
	}
}


template<typename Array1D> HOST_DEVICE typename Array1D::elem_type vec_diff_l1(const Array1D &a, const Array1D &b, int num)
{
	typedef typename Array1D::elem_type real;
	real result = real(0);
	for (int i = 0; i < num; i++)
	{
		real val_a = a(i);
		real val_b = b(i);
		real diff = realabs(val_a - val_b);
		result += diff;
	}
	return result;
}


template<typename Tout, typename Tin> HOST_DEVICE FORCEINLINE Tout convert_type(Tin in);
template<> HOST_DEVICE FORCEINLINE float convert_type<float, float>(float in) { return in; }
template<> HOST_DEVICE FORCEINLINE float convert_type<float, double>(double in) { return (float)in; }
template<> HOST_DEVICE FORCEINLINE float convert_type<float, unsigned char>(unsigned char in) { return (float)in / 255.0f; }
template<> HOST_DEVICE FORCEINLINE double convert_type<double, float>(float in) { return (double)in; }
template<> HOST_DEVICE FORCEINLINE double convert_type<double, double>(double in) { return in; }
template<> HOST_DEVICE FORCEINLINE double convert_type<double, unsigned char>(unsigned char in) { return (double)in / 255.0; }
template<> HOST_DEVICE FORCEINLINE unsigned char convert_type<unsigned char, float>(float in) { return (unsigned char)realmax(0, realmin(255, (int)realfloor(in * 255.0f))); }
template<> HOST_DEVICE FORCEINLINE unsigned char convert_type<unsigned char, double>(double in) { return (unsigned char)realmax(0, realmin(255, (int)realfloor(in * 255.0))); }


enum ElemKind
{
	elem_kind_uchar = 0,
	elem_kind_float,
	elem_kind_double
};


template<typename T> struct ElemType2Kind;
template<> struct ElemType2Kind<unsigned char> { static const ElemKind value = elem_kind_uchar; };
template<> struct ElemType2Kind<float> { static const ElemKind value = elem_kind_float; };
template<> struct ElemType2Kind<double> { static const ElemKind value = elem_kind_double; };


struct ElemKindGeneral
{
	HOST_DEVICE static size_t size(ElemKind elem_kind)
	{
		switch (elem_kind)
		{
			case elem_kind_uchar: { return sizeof(unsigned char); }
			case elem_kind_float: { return sizeof(float); }
			case elem_kind_double: { return sizeof(double); }
			default: { return 0; }
		}
	}
};


HOST_DEVICE FORCEINLINE void convert_type(ElemKind out_kind, ElemKind in_kind, void *out, const void *in)
{
	if (out_kind == elem_kind_float && in_kind == elem_kind_float) { *(float*)out = convert_type<float, float>(*(const float*)in); }
	if (out_kind == elem_kind_float && in_kind == elem_kind_double) { *(float*)out = convert_type<float, double>(*(const double*)in); }
	if (out_kind == elem_kind_float && in_kind == elem_kind_uchar) { *(float*)out = convert_type<float, unsigned char>(*(const unsigned char*)in); }
	if (out_kind == elem_kind_double && in_kind == elem_kind_float) { *(double*)out = convert_type<double, float>(*(const float*)in); }
	if (out_kind == elem_kind_double && in_kind == elem_kind_double) { *(double*)out = convert_type<double, double>(*(const double*)in); }
	if (out_kind == elem_kind_double && in_kind == elem_kind_uchar) { *(double*)out = convert_type<double, unsigned char>(*(const unsigned char*)in); }
	if (out_kind == elem_kind_uchar && in_kind == elem_kind_float) { *(unsigned char*)out = convert_type<unsigned char, float>(*(const float*)in); }
	if (out_kind == elem_kind_uchar && in_kind == elem_kind_double) { *(unsigned char*)out = convert_type<unsigned char, double>(*(const double*)in); }
}



#undef HOST_DEVICE
#undef FORCEINLINE

#endif // AUX_REAL_H
