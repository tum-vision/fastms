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

#ifndef UTIL_CUDA_TIMER_H
#define UTIL_CUDA_TIMER_H

#if !defined(DISABLE_CUDA) && defined(__CUDACC__)

#include <cuda_runtime.h>



class DeviceTimer
{
public:
	DeviceTimer() : running(false), sec(0.0)
	{
		cudaEventCreate(&event_start);
		cudaEventCreate(&event_stop);
	}
	~DeviceTimer()
	{
		cudaEventDestroy(event_start);
		cudaEventDestroy(event_stop);
	}
	void start()
	{
		cudaEventRecord(event_start,0);
		cudaEventSynchronize(event_start);
		running = true;
	}
	void end()
	{
		if (!running)
		{
			sec = 0;
			return;
		}
		cudaEventRecord(event_stop,0);
		cudaEventSynchronize(event_stop);
		float cuda_duration;
		cudaEventElapsedTime(&cuda_duration, event_start, event_stop);
		sec = (double)cuda_duration / 1000.0;
		running = false;
	}
	double get()
	{
		if (running) end();
		return sec;
	}
private:
	cudaEvent_t event_start;
	cudaEvent_t event_stop;
	bool running;
	double sec;
};



#endif // !defined(DISABLE_CUDA) && defined(__CUDACC__)

#endif // UTIL_CUDA_TIMER_H
