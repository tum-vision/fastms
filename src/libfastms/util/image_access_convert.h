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

#ifndef UTIL_IMAGE_ACCESS_CONVERT_H
#define UTIL_IMAGE_ACCESS_CONVERT_H

#include "image_access.h"


template<typename TUntypedAccessOut, typename TUntypedAccessIn>
void copy_image_h2h_base(TUntypedAccessOut out, TUntypedAccessIn in)
{
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
	#pragma omp parallel default(none) firstprivate(out, in)
	{
#endif
	const ElemKind out_kind = out.elem_kind();
	const ElemKind in_kind = in.elem_kind();
	const Dim2D &dim2d = in.dim().dim2d();
	const int num_channels = in.dim().num_channels;
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
	#pragma omp for
#endif
	for (int y = 0; y < dim2d.h; y++)
	{
		for (int i = 0; i < num_channels; i++)
		{
			for (int x = 0; x < dim2d.w; x++)
			{
				convert_type(out_kind, in_kind, out.get_address(x, y, i), in.get_address(x, y, i));
			}
		}
	}
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
	}
#endif
}


template<typename TUntypedAccessOut, typename TUntypedAccessIn>
void copy_image_h2h(TUntypedAccessOut out, TUntypedAccessIn in)
{
	if (out.elem_kind() == in.elem_kind() &&
		types_equal<typename TUntypedAccessOut::data_interpretation_t, typename TUntypedAccessIn::data_interpretation_t>::value)
	{
		HostAllocator::copy2d(out.data(), out.data_pitch(), in.const_data(), in.data_pitch(), in.data_width_in_bytes(), in.data_height());
	}
	else
	{
		copy_image_h2h_base(out, in);
	}
}


#ifndef DISABLE_CUDA
template<typename TUntypedAccessOut, typename TUntypedAccessIn>
void copy_image_d2d_base(TUntypedAccessOut out, TUntypedAccessIn in);


template<typename TUntypedAccessOut, typename TUntypedAccessIn>
void copy_image_d2d(TUntypedAccessOut out, TUntypedAccessIn in)
{
	if (out.elem_kind() == in.elem_kind() &&
		types_equal<typename TUntypedAccessOut::data_interpretation_t, typename TUntypedAccessIn::data_interpretation_t>::value)
	{
		DeviceAllocator::copy2d(out.data(), out.data_pitch(), in.data(), in.data_pitch(), in.data_width_in_bytes(), in.data_height());
	}
	else
	{
		copy_image_d2d_base(out, in);
	}
}


template<typename TUntypedAccessOut, typename TUntypedAccessIn>
void copy_image_h2d(TUntypedAccessOut out, TUntypedAccessIn in)
{
	if (out.elem_kind() == in.elem_kind() &&
		types_equal<typename TUntypedAccessOut::data_interpretation_t, typename TUntypedAccessIn::data_interpretation_t>::value)
	{
		DeviceAllocator::copy2d_h2d(out.data(), out.data_pitch(), in.data(), in.data_pitch(), in.data_width_in_bytes(), in.data_height());
	}
	else
	{
		TUntypedAccessIn in_device = alloc_untyped_access<TUntypedAccessIn>(in.dim(), in.elem_kind(), false);  // false: not on_host = on_device
		DeviceAllocator::copy2d_h2d(in_device.data(), in_device.data_pitch(), in.data(), in.data_pitch(), in.data_width_in_bytes(), in.data_height());
		copy_image_d2d(out, in_device);
		DeviceAllocator::free(in_device.data());
	}
}


template<typename TUntypedAccessOut, typename TUntypedAccessIn>
void copy_image_d2h(TUntypedAccessOut out, TUntypedAccessIn in)
{
	if (out.elem_kind() == in.elem_kind() &&
		types_equal<typename TUntypedAccessOut::data_interpretation_t, typename TUntypedAccessIn::data_interpretation_t>::value)
	{
		DeviceAllocator::copy2d_d2h(out.data(), out.data_pitch(), in.data(), in.data_pitch(), in.data_width_in_bytes(), in.data_height());
	}
	else
	{
		TUntypedAccessOut out_device = alloc_untyped_access<TUntypedAccessOut>(out.dim(), out.elem_kind(), false);  // false: not on_host = on_device
	    copy_image_d2d(out_device, in);
		DeviceAllocator::copy2d_d2h(out.data(), out.data_pitch(), out_device.data(), out_device.data_pitch(), out.data_width_in_bytes(), out.data_height());
		DeviceAllocator::free(out_device.data());
	}
}
#endif // not DISABLE_CUDA

#include <iostream>
template<typename TUntypedAccessOut, typename TUntypedAccessIn>
void copy_image(TUntypedAccessOut out, TUntypedAccessIn in)
{
#ifndef DISABLE_CUDA
	if (in.is_on_host() && out.is_on_host())
	{
		copy_image_h2h(out, in);
	}
	else if (!in.is_on_host() && out.is_on_host())
	{
		copy_image_d2h(out, in);
	}
	else if (in.is_on_host() && !out.is_on_host())
	{
		copy_image_h2d(out, in);
	}
	else
	{
		copy_image_d2d(out, in);
	}
#else
	copy_image_h2h(out, in);
#endif // not DISABLE_CUDA
}


#undef HOST_DEVICE
#undef FORCEINLINE

#endif // UTIL_IMAGE_ACCESS_CONVERT_H
