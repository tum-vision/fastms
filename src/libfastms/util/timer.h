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

#ifndef UTIL_TIMER_H
#define UTIL_TIMER_H

//#include <ctime>
#include <cstddef>
#include <sys/time.h>


class Timer
{
public:
	Timer() : time_start(0.0), running(false), seconds(0.0)
	{
	}
	void start()
	{
		time_start = get_cur_seconds();
		running = true;
	}
	void end()
	{
		if (!running) { seconds = 0.0; return; }
		seconds = get_cur_seconds() - time_start;
		running = false;
	}
	double get()
	{
		if (running) end();
		return seconds;
	}
private:
	// this method accumulates the time spent in all threads as if they were running in parallel,
	// so we can't use this if we compute something with OpenMP
	// double get_cur_seconds() { return (double)clock() / CLOCKS_PER_SEC; }

	// this gives the wall clock time, and works as expected with OpenMP
	double get_cur_seconds()
	{
	    struct timeval cur_time;
	    if (gettimeofday(&cur_time, NULL)) { return 0.0; }
	    return (double)cur_time.tv_sec + 1e-6 * (double)cur_time.tv_usec;
	}
	double time_start;
	bool running;
	double seconds;
};


#endif // UTIL_TIMER_H
