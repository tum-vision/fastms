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

#ifndef UTIL_IMAGE_ACCESS_H
#define UTIL_IMAGE_ACCESS_H

#include <cstring>  // for memset, memcpy
#include "real.h"
#include "types_equal.h"
#include <ostream>

#ifndef DISABLE_CUDA
#include <cuda_runtime.h>
#endif // not DISABLE_CUDA

#if !defined(DISABLE_CUDA) && defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#define FORCEINLINE __forceinline__
#else
#define HOST_DEVICE
#define FORCEINLINE inline
#endif // !defined(DISABLE_CUDA) && defined(__CUDACC__)



struct DataIndex
{
	HOST_DEVICE DataIndex() : x(0), y(0) {}
	HOST_DEVICE DataIndex(size_t x, size_t y) : x(x), y(y) {}
	size_t x;
	size_t y;
};


struct DataDim
{
	HOST_DEVICE DataDim() : pitch(0), height(0) {}
	HOST_DEVICE DataDim(size_t pitch, size_t height) : pitch(pitch), height(height) {}
	HOST_DEVICE size_t num_bytes() const { return pitch * height; }
	size_t pitch;
	size_t height;
};


struct Dim2D
{
	HOST_DEVICE Dim2D() : w(0), h(0) {}
	HOST_DEVICE Dim2D(int w, int h) : w(w), h(h) {}
	HOST_DEVICE size_t num_elem() const { return (size_t)w * h; }
	HOST_DEVICE bool operator== (const Dim2D &other_dim) { return w == other_dim.w && h == other_dim.h; }
	HOST_DEVICE bool operator!= (const Dim2D &other_dim) { return !(operator==(other_dim)); }

	int w;
	int h;
};


struct ArrayDim
{
	HOST_DEVICE ArrayDim() : w(0), h(0), num_channels(0) {}
	HOST_DEVICE ArrayDim(int w, int h, int num_channels) : w(w), h(h), num_channels(num_channels) {}
	HOST_DEVICE Dim2D dim2d() const { return Dim2D(w, h); }
	HOST_DEVICE size_t num_elem() const { return (size_t)w * h * num_channels; }
	HOST_DEVICE bool operator== (const ArrayDim &other_dim) { return w == other_dim.w && h == other_dim.h && num_channels == other_dim.num_channels; }
	HOST_DEVICE bool operator!= (const ArrayDim &other_dim) { return !(operator==(other_dim)); }

	int w;
	int h;
	int num_channels;
};
inline std::ostream& operator<< (std::ostream &out, const ArrayDim &dim)
{
	out << dim.w << " x " << dim.h << " x " << dim.num_channels;
	return out;
}


class HostAllocator
{
public:
	static void free(void *&ptr) { delete[] (char*)ptr; ptr = NULL; }
	static void setzero(void *ptr, size_t num_bytes) { memset(ptr, 0, num_bytes); }
	static void* alloc2d(DataDim *used_data_dim)
	{
		return (void*)(new char[used_data_dim->num_bytes()]);
	}
	static void copy2d(void *out, size_t out_pitch, const void *in, size_t in_pitch, size_t w_bytes, size_t h_lines)
	{
		if (out_pitch == in_pitch)
		{
			size_t num_bytes = in_pitch * h_lines;
			memcpy(out, in, num_bytes);
		}
		else
		{
			for (size_t y = 0; y < h_lines; y++)
			{
				void *out_y = (void*)((char*)out + out_pitch * y);
				const void *in_y = (const void*)((const char*)in + in_pitch * y);
				memcpy(out_y, in_y, w_bytes);
			}
		}
	}
};


#ifndef DISABLE_CUDA
class DeviceAllocator
{
public:
	static void free(void *&ptr_cuda) { cudaFree(ptr_cuda); ptr_cuda = NULL; }
	static void setzero(void *ptr_cuda, size_t num_bytes)
	{
		cudaMemset(ptr_cuda, 0, num_bytes);
	}
	static void* alloc2d(DataDim *used_data_dim)
	{
		void *ptr_cuda = NULL;
		size_t pitch0 = 0;
		cudaMallocPitch(&ptr_cuda, &pitch0, used_data_dim->pitch, used_data_dim->height);
		used_data_dim->pitch = pitch0;
		return ptr_cuda;
	}
	static void copy2d(void *out_cuda, size_t out_pitch, const void *in_cuda, size_t in_pitch, size_t w_bytes, size_t h_lines)
	{
		cudaMemcpy2D(out_cuda, out_pitch, in_cuda, in_pitch, w_bytes, h_lines, cudaMemcpyDeviceToDevice);
	}
	static void copy2d_h2d(void *out_cuda, size_t out_pitch, const void *in_host, size_t in_pitch, size_t w_bytes, size_t h_lines)
	{
		cudaMemcpy2D(out_cuda, out_pitch, in_host, in_pitch, w_bytes, h_lines, cudaMemcpyHostToDevice);
	}
	static void copy2d_d2h(void *out_host, size_t out_pitch, const void *in_cuda, size_t in_pitch, size_t w_bytes, size_t h_lines)
	{
		cudaMemcpy2D(out_host, out_pitch, in_cuda, in_pitch, w_bytes, h_lines, cudaMemcpyDeviceToHost);
	}
};
#endif // not DISABLE_CUDA


struct ImageData
{
	HOST_DEVICE ImageData() : data_(NULL), data_pitch_(0) {}
	HOST_DEVICE ImageData(void *data, const ArrayDim &dim, size_t data_pitch) : data_(data), dim_(dim), data_pitch_(data_pitch) {}
	void *data_;
	ArrayDim dim_;
	size_t data_pitch_;
};


template<typename TUntypedAccess>
TUntypedAccess alloc_untyped_access(const ArrayDim &dim, ElemKind elem_kind, bool on_host)
{
	DataDim data_dim = TUntypedAccess::data_interpretation_t::used_data_dim(dim, ElemKindGeneral::size(elem_kind));
#ifndef DISABLE_CUDA
	void *data = (on_host? HostAllocator::alloc2d(&data_dim) : DeviceAllocator::alloc2d(&data_dim));
#else
	void *data = HostAllocator::alloc2d(&data_dim);
#endif // not DISABLE_CUDA
	return TUntypedAccess(ImageData(data, dim, data_dim.pitch), elem_kind, on_host);
}


template<typename DataInterpretation> struct ImageUntypedAccess;

template<typename T, typename DataInterpretation>
struct ImageAccess
{
	typedef T elem_t;
	typedef DataInterpretation data_interpretation_t;
	typedef ImageUntypedAccess<data_interpretation_t> image_untyped_access_t;

	HOST_DEVICE ImageAccess() : is_on_host_(true) {}
	HOST_DEVICE ImageAccess(void *data, const ArrayDim &dim, bool is_on_host) :
			image_data_(data, dim, data_interpretation_t::used_data_dim(dim, sizeof(T)).pitch), is_on_host_(is_on_host) {}
	HOST_DEVICE ImageAccess(const ImageData &image_data, bool is_on_host) :
			image_data_(image_data), is_on_host_(is_on_host) {}

	HOST_DEVICE image_untyped_access_t get_untyped_access() const { return image_untyped_access_t(image_data_, ElemType2Kind<T>::value, is_on_host_); }

	HOST_DEVICE T& get(int x, int y, int i) { return *(T*)((char*)image_data_.data_ + offset(x, y, i)); }
	HOST_DEVICE const T& get(int x, int y, int i) const { return *(T*)((char*)image_data_.data_ + offset(x, y, i)); }
	HOST_DEVICE ArrayDim dim() const { return image_data_.dim_; }
	HOST_DEVICE bool is_valid() const { return image_data_.data_ != NULL; }
	HOST_DEVICE bool is_on_host() const { return is_on_host_; }

	HOST_DEVICE void*& data() { return image_data_.data_; }
	HOST_DEVICE const void* const_data() const { return image_data_.data_; }
	HOST_DEVICE size_t data_pitch() const { return image_data_.data_pitch_; }
	HOST_DEVICE size_t data_height() const { return used_data_dim().height; }
	HOST_DEVICE size_t data_width_in_bytes() const { return used_data_dim().pitch; }
	HOST_DEVICE size_t num_bytes() const { return data_pitch() * data_height(); }

private:
	HOST_DEVICE size_t offset(int x, int y, int i) const
	{
		const DataIndex &data_index = data_interpretation_t::get(x, y, i, image_data_.dim_);
		return data_index.x * sizeof(T) + image_data_.data_pitch_ * data_index.y;
	}
	HOST_DEVICE DataDim used_data_dim() const { return data_interpretation_t::used_data_dim(image_data_.dim_, sizeof(T)); }

	ImageData image_data_;
	bool is_on_host_;
};


template<typename DataInterpretation>
struct ImageUntypedAccess
{
	typedef DataInterpretation data_interpretation_t;
	template<typename T> struct image_access_t { typedef ImageAccess<T, data_interpretation_t> type; };

	HOST_DEVICE ImageUntypedAccess() : elem_kind_(elem_kind_uchar), is_on_host_(true) {}
	HOST_DEVICE ImageUntypedAccess(void *data, const ArrayDim &dim, ElemKind elem_kind, bool is_on_host) :
			image_data_(data, dim, data_interpretation_t::used_data_dim(dim, ElemKindGeneral::size(elem_kind)).pitch),
			elem_kind_(elem_kind), is_on_host_(is_on_host) {}
	HOST_DEVICE ImageUntypedAccess(const ImageData &image_data, ElemKind elem_kind, bool is_on_host) :
			image_data_(image_data),
			elem_kind_(elem_kind), is_on_host_(is_on_host) {}

	template<typename T> HOST_DEVICE typename image_access_t<T>::type get_access() const { return typename image_access_t<T>::type(image_data_, is_on_host_); }
	HOST_DEVICE ElemKind elem_kind() const { return elem_kind_; }

	HOST_DEVICE void* get_address(int x, int y, int i) { return (void*)((char*)image_data_.data_ + offset_address(x, y, i)); }
	HOST_DEVICE const void* get_address(int x, int y, int i) const { return (const void*)((const char*)image_data_.data_ + offset_address(x, y, i)); }
	HOST_DEVICE ArrayDim dim() const { return image_data_.dim_; }
	HOST_DEVICE bool is_valid() const { return image_data_.data_ != NULL; }
	HOST_DEVICE bool is_on_host() const { return is_on_host_; }

	HOST_DEVICE void*& data() { return image_data_.data_; }
	HOST_DEVICE const void* const_data() const { return image_data_.data_; }
	HOST_DEVICE size_t data_pitch() const { return image_data_.data_pitch_; }
	HOST_DEVICE size_t data_height() const { return used_data_dim().height; }
	HOST_DEVICE size_t data_width_in_bytes() const { return used_data_dim().pitch; }
	HOST_DEVICE size_t num_bytes() const { return data_pitch() * data_height(); }

private:
	HOST_DEVICE size_t offset_address(int x, int y, int i) const
	{
		const DataIndex &data_index = data_interpretation_t::get(x, y, i, image_data_.dim_);
		return data_index.x * elem_size() + image_data_.data_pitch_ * data_index.y;
	}
	HOST_DEVICE DataDim used_data_dim() const { return data_interpretation_t::used_data_dim(image_data_.dim_, elem_size()); }
	HOST_DEVICE size_t elem_size() const { return ElemKindGeneral::size(elem_kind_); }

	ImageData image_data_;
	ElemKind elem_kind_;
	bool is_on_host_;
};


struct DataInterpretationLayered
{
	HOST_DEVICE static DataIndex get(int x, int y, int i, const ArrayDim &dim)
	{
		return DataIndex(x, y + (size_t)dim.h * i);
	}
	HOST_DEVICE static DataDim used_data_dim (const ArrayDim &dim, size_t elem_size)
	{
		return DataDim(dim.w * elem_size, (size_t)dim.h * dim.num_channels);
	}
};


struct DataInterpretationLayeredTransposed
{
	HOST_DEVICE static DataIndex get(int x, int y, int i, const ArrayDim &dim)
	{
		return DataIndex(y, x + (size_t)dim.w * i);
	}

	HOST_DEVICE static DataDim used_data_dim (const ArrayDim &dim, size_t elem_size)
	{
		return DataDim(dim.h * elem_size, (size_t)dim.w * dim.num_channels);
	}
};


struct DataInterpretationInterlaced
{
	HOST_DEVICE static DataIndex get(int x, int y, int i, const ArrayDim &dim)
	{
		return DataIndex(i + (size_t)dim.num_channels * x, y);
	}
	HOST_DEVICE static DataDim used_data_dim (const ArrayDim &dim, size_t elem_size)
	{
		return DataDim((size_t)dim.num_channels * dim.w * elem_size, dim.h);
	}
};


struct DataInterpretationInterlacedReversed
{
	HOST_DEVICE static DataIndex get(int x, int y, int i, const ArrayDim &dim)
	{
		return DataIndex((dim.num_channels - 1 - i) + (size_t)dim.num_channels * x, y);
	}
	HOST_DEVICE static DataDim used_data_dim (const ArrayDim &dim, size_t elem_size)
	{
		return DataDim((size_t)dim.num_channels * dim.w * elem_size, dim.h);
	}
};



#undef HOST_DEVICE
#undef FORCEINLINE

#endif // UTIL_IMAGE_ACCESS_H
