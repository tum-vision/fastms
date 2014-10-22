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

#include "check_cuda.cuh"

#if !defined(DISABLE_CUDA) && defined(__CUDACC__)

#include <cuda_runtime.h>
#include <iostream>



namespace
{

std::string prev_file = "";
int prev_line = 0;

} // namespace



void cuda_check(std::string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        std::cerr << std::endl << file.c_str() << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << std::endl;
        if (prev_line>0) std::cerr << "Previous CUDA call:" << std::endl << prev_file.c_str() << ", line " << prev_line << std::endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}



#endif // !defined(DISABLE_CUDA) && defined(__CUDACC__)
