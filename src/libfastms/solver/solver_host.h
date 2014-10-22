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

#ifndef SOLVER_HOST_H
#define SOLVER_HOST_H

#include "solver.h"


template<typename real> class SolverHostImplementation;

template<typename real>
class SolverHost
{
public:
	typedef real real_t;

	SolverHost();
	~SolverHost();

	BaseImage* run(const BaseImage *in_image, const Par &par);

private:
	SolverHost(const SolverHost<real> &other_solver);  // disable
	SolverHost<real>& operator= (const SolverHost<real> &other_solver);  // disable

	SolverHostImplementation<real> *implementation;
};



#endif // SOLVER_HOST_H
