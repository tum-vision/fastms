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

#include "solver.h"

#include "solver_host.h"
#ifndef DISABLE_CUDA
#include "solver_device.h"
#endif // not DISABLE_CUDA
#ifndef DISABLE_OPENCV
#include "util/image_mat.h"
#endif // not DISABLE_OPENCV
#include "util/image.h"
#include "util/types_equal.h"
#include "util/has_cuda.h"
#include <iostream>



class SolverImplementation
{
public:
	virtual ~SolverImplementation() {}

	// general
	virtual BaseImage* run(const BaseImage *in, const Par &par) = 0;

	// layered real
	virtual void run(float *&out_image, const float *in_image, const ArrayDim &dim, const Par &par) = 0;
	virtual void run(double *&out_image, const double *in_image, const ArrayDim &dim, const Par &par) = 0;

	// interlaced char
	virtual void run(unsigned char *&out_image, const unsigned char *in_image, const ArrayDim &dim, const Par &par) = 0;

	// cv::Mat
#ifndef DISABLE_OPENCV
	virtual cv::Mat run(const cv::Mat in_image, const Par &par) = 0;
#endif // not DISABLE_OPENCV

	virtual int get_class_type() = 0; // for is_instance_of()
};



namespace
{

template<typename Solver>
class SolverImplementationConcrete: public SolverImplementation
{
public:
	virtual ~SolverImplementationConcrete() {};

	// general
	virtual BaseImage* run(const BaseImage *in, const Par &par)
	{
		return solver.run(in, par);
	}

	// layered real
	virtual void run(float *&out_image, const float *in_image, const ArrayDim &dim, const Par &par)
	{
		run_real(out_image, in_image, dim, par);
	}
	virtual void run(double *&out_image, const double *in_image, const ArrayDim &dim, const Par &par)
	{
		run_real(out_image, in_image, dim, par);
	}
	template<typename real> void run_real(real *&out_image, const real *in_image, const ArrayDim &dim, const Par &par)
	{
		typedef ManagedImage<real, DataInterpretationLayered> managed_image_t;

		managed_image_t in_managed(const_cast<real*>(in_image), dim);
		managed_image_t *out_managed = static_cast<managed_image_t*>(solver.run(&in_managed, par));
		if (out_image)
		{
			// copy
			managed_image_t outimage_managed(out_image, dim);
			outimage_managed.copy_from_samekind(out_managed);
		}
		else
		{
			// move
			out_image = out_managed->release_data();
		}
		delete out_managed;
	}

	// interlaced char
	virtual void run(unsigned char *&out_image, const unsigned char *in_image, const ArrayDim &dim, const Par &par)
	{
		typedef ManagedImage<unsigned char, DataInterpretationInterlaced> managed_image_t;

		managed_image_t in_managed(const_cast<unsigned char*>(in_image), dim);
		managed_image_t *out_managed = static_cast<managed_image_t*>(solver.run(&in_managed, par));
		if (out_image)
		{
			// copy
			managed_image_t outimage_managed(out_image, dim);
			outimage_managed.copy_from_samekind(out_managed);
		}
		else
		{
			// move
			out_image = out_managed->release_data();
		}
		delete out_managed;
	}

	// cv::Mat
#ifndef DISABLE_OPENCV
	cv::Mat run(const cv::Mat in_mat, const Par &par)
	{
		MatImage in_matimage(in_mat);
		MatImage *out_matimage = static_cast<MatImage*>(solver.run(&in_matimage, par));
		cv::Mat out_mat = out_matimage->get_mat();
		delete out_matimage;
		return out_mat;
	}
#endif // not DISABLE_OPENCV

	int get_class_type() { return class_type; }
	static int static_get_class_type() { return class_type; }

private:
	Solver solver;

	static const int class_type =
#ifndef DISABLE_CUDA
	(types_equal<Solver, SolverHost<float> >::value? 0 : \
     types_equal<Solver, SolverHost<double> >::value? 1 : \
     types_equal<Solver, SolverDevice<float> >::value? 2 : \
     types_equal<Solver, SolverDevice<double> >::value? 3 : -1);
#else
	(types_equal<Solver, SolverHost<float> >::value? 0 : \
     types_equal<Solver, SolverHost<double> >::value? 1 : -1);
#endif // not DISABLE_CUDA
};


template<typename T_class, typename T_object> bool is_instance_of(T_object *object)
{
	return (T_class::static_get_class_type() == object->get_class_type());
}
template<typename Implementation> void set_implementation_concrete(SolverImplementation *&implementation, const Par &par)
{
	if (implementation && !is_instance_of<Implementation>(implementation))
	{
		// allocated, but wrong class
		delete implementation;
		implementation = NULL;
	}
	if (!implementation)
	{
		implementation = new Implementation();
	}
}
template<typename real> void set_implementation_real(SolverImplementation *&implementation, const Par &par)
{
	switch (par.engine)
	{
		case Par::engine_cpu:
		{
			set_implementation_concrete<SolverImplementationConcrete<SolverHost<real> > >(implementation, par);
			return;
		}
		case Par::engine_cuda:
		{
			std::string error_str;
			bool cuda_ok = has_cuda(&error_str);
#ifndef DISABLE_CUDA
			if (cuda_ok) { set_implementation_concrete<SolverImplementationConcrete<SolverDevice<real> > >(implementation, par); }
#endif // not DISABLE_CUDA
			if (!cuda_ok)
			{
				std::cerr << "ERROR: Solver::run(): Could not select CUDA engine, USING CPU VERSION INSTEAD (" << error_str.c_str() << ")." << std::endl;
				Par par_cpu = par;
				par_cpu.engine = Par::engine_cpu;
				set_implementation_real<real>(implementation, par_cpu);
			}
			break;
		}
		default:
		{
			std::cerr << "ERROR: Solver::run(): Unexpected engine " << par.engine << ", USING CPU VERSION INSTEAD" << std::endl;
			Par par_cpu = par;
			par_cpu.engine = Par::engine_cpu;
			set_implementation_real<real>(implementation, par_cpu);
		}
	}
}
void set_implementation(SolverImplementation *&implementation, const Par &par)
{
	if (par.use_double)
	{
		set_implementation_real<double>(implementation, par);
	}
	else
	{
		set_implementation_real<float>(implementation, par);
	}
}

} // namespace



Solver::Solver() : implementation(NULL) {}
Solver::~Solver() { if (implementation) { delete implementation; } }
BaseImage* Solver::run(const BaseImage *in, const Par &par)
{
	set_implementation(implementation, par); if (!implementation) { return NULL; }
	return implementation->run(in, par);
}
void Solver::run(float *&out_image, const float *in_image, const ArrayDim &dim, const Par &par)
{
	set_implementation_real<float>(implementation, par); if (!implementation) { return; }
	return implementation->run(out_image, in_image, dim, par);
}
void Solver::run(double *&out_image, const double *in_image, const ArrayDim &dim, const Par &par)
{
	set_implementation_real<double>(implementation, par); if (!implementation) { return; }
	return implementation->run(out_image, in_image, dim, par);
}
void Solver::run(unsigned char *&out_image, const unsigned char *in_image, const ArrayDim &dim, const Par &par)
{
	set_implementation(implementation, par); if (!implementation) { return; }
	return implementation->run(out_image, in_image, dim, par);
}
#ifndef DISABLE_OPENCV
cv::Mat Solver::run(const cv::Mat in_image, const Par &par)
{
	set_implementation(implementation, par); if (!implementation) { return cv::Mat(); }
	return implementation->run(in_image, par);
}
#endif // not DISABLE_OPENCV


