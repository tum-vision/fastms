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

#ifndef SOLVER_DEVICE_H
#define SOLVER_DEVICE_H

#ifndef DISABLE_CUDA

#include "solver.h"



template<typename real> class SolverDeviceImplementation;

template<typename real>
class SolverDevice
{
public:
	typedef real real_t;

	SolverDevice();
	~SolverDevice();

	BaseImage* run(const BaseImage *image, const Par &par);

private:
	SolverDevice(const SolverDevice<real> &other_solver);  // disable
	SolverDevice<real>& operator= (const SolverDevice<real> &other_solver);  // disable

	SolverDeviceImplementation<real> *implementation;
};


#endif // not DISABLE_CUDA

#endif // SOLVER_DEVICE_H
