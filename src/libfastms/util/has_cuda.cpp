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

#include "has_cuda.h"

#ifndef DISABLE_CUDA
#include <cuda_runtime.h>
#endif // not DISABLE_CUDA

#include <sstream>



bool has_cuda(std::string *error_str)
{
#ifndef DISABLE_CUDA
	int num_dev = 0;
	cudaError_t e = cudaGetDeviceCount(&num_dev);
	if (e == cudaSuccess)
	{
		if (num_dev > 0)
		{
			if (error_str) { *error_str = ""; }
			return true;
		}
		else
		{
			if (error_str) { *error_str = "No CUDA capable devices detected"; }
			return false;
		}
	}
	else
	{
		if (error_str)
		{
			std::stringstream s;
			s << "cuda error code " << e << ": " << cudaGetErrorString(e);
			*error_str = s.str();
		}
		return false;
	}
#else
	if (error_str) { *error_str = "CUDA was disabled during compilation"; }
	return false;
#endif // not DISABLE_CUDA
}
