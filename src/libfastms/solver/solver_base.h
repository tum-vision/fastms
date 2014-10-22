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

#ifndef SOLVER_BASE_H
#define SOLVER_BASE_H

#include "solver_common_operators.h"
#include "util/image.h"



template<typename real>
class Engine
{
public:
	typedef ImageAccess<real, DataInterpretationLayered> image_access_t;
	typedef typename image_access_t::data_interpretation_t data_interpretation_t;
	typedef LinearOperator<real> linear_operator_t;
	typedef Regularizer<image_access_t> regularizer_t;
	typedef Dataterm<image_access_t> dataterm_t;
	typedef ImageManagerBase<real, data_interpretation_t> image_manager_base_t;

	virtual ~Engine() {}
	virtual std::string str() { return ""; };
	virtual void alloc(const ArrayDim &dim_u) = 0;
	virtual void free() = 0;
	virtual bool is_valid() = 0;
	virtual image_manager_base_t* image_manager() = 0;
	virtual real get_sum(image_access_t a) = 0;
	virtual void timer_start() = 0;
	virtual void timer_end() = 0;
	virtual double timer_get() = 0;
	virtual void synchronize() = 0;

	virtual void run_dual_p(image_access_t p, image_access_t u, linear_operator_t linear_operator, regularizer_t regularizer, real dt) = 0;
	virtual void run_prim_u(image_access_t u, image_access_t ubar, image_access_t p, linear_operator_t linear_operator, dataterm_t dataterm, real theta_bar, real dt) = 0;
	virtual void energy_base(image_access_t u, image_access_t aux_reduce, linear_operator_t linear_operator, dataterm_t dataterm, regularizer_t regularizer) = 0;
	virtual void add_edges(image_access_t cur_result, linear_operator_t linear_operator, regularizer_t regularizer) = 0;
	virtual void set_regularizer_weight_from__normgrad(image_access_t regularizer_weight, image_access_t image, linear_operator_t linear_operator) = 0;
	virtual void set_regularizer_weight_from__exp(image_access_t regularizer_weight, real coeff) = 0;
	virtual void diff_l1_base(image_access_t a, image_access_t b, image_access_t aux_reduce) = 0;
};


template<typename real>
class SolverBase
{
public:
	SolverBase();
	virtual ~SolverBase();

	BaseImage* run(const BaseImage *image, const Par &par_const);

protected:
	void set_engine(Engine<real> *engine);

private:
	typedef typename Engine<real>::image_access_t image_access_t;
	typedef typename Engine<real>::linear_operator_t linear_operator_t;

	size_t alloc(const ArrayDim &dim_u);
	void free();
	void init(const BaseImage *image);
	void set_regularizer_weight_from(image_access_t image);
	real energy();
	real diff_l1(image_access_t a, image_access_t b);
	bool is_converged(int iteration);
	void print_stats();
	BaseImage* get_solution(const BaseImage *image);

	Engine<real> *engine;
	Par par;
	PrimalDualVars<image_access_t> pd_vars;
	bool u_is_computed;

	struct Arrays
	{
		size_t alloc(Engine<real> *engine, const ArrayDim &dim_u, const ArrayDim &dim_p)
		{
			ArrayDim dim_scalar(dim_u.w, dim_u.h, 1);
			size_t mem = 0;
			mem += engine->image_manager()->alloc(u, dim_u);
			mem += engine->image_manager()->alloc(ubar, dim_u);
			mem += engine->image_manager()->alloc(f, dim_u);
			mem += engine->image_manager()->alloc(p, dim_p);
			mem += engine->image_manager()->alloc(regularizer_weight, dim_scalar);
			mem += engine->image_manager()->alloc(prev_u, dim_u);
			mem += engine->image_manager()->alloc(aux_result, dim_u);
			mem += engine->image_manager()->alloc(aux_reduce, dim_scalar);
			return mem;
		}
		void free(Engine<real> *engine)
		{
			engine->image_manager()->free(u);
			engine->image_manager()->free(ubar);
			engine->image_manager()->free(f);
			engine->image_manager()->free(p);
			engine->image_manager()->free(regularizer_weight);
			engine->image_manager()->free(prev_u);
			engine->image_manager()->free(aux_result);
			engine->image_manager()->free(aux_reduce);
		}
		image_access_t u;
		image_access_t ubar;
		image_access_t f;
		image_access_t p;
		image_access_t regularizer_weight;
		image_access_t prev_u;
		image_access_t aux_result;
		image_access_t aux_reduce;
	} arr;

	struct ResultStats
	{
		ResultStats()
		{
			mem = 0;
			stop_iteration = -1;
			time_compute = 0.0;
			time_compute_sum = 0.0;
			time = 0.0;
			time_sum = 0.0;
			num_runs = 0;
			energy = real(0);
		}
		ArrayDim dim_u;
		ArrayDim dim_p;
		size_t mem;
		int stop_iteration;
		double time_compute;      // actual computation without alloc and image conversions and copying, in seconds
		double time_compute_sum;  // accumulation for averaging
		double time;              // overall time, including alloc, image conversions and copying, in second
		double time_sum;          // accumulation for averaging
		int num_runs;
		real energy;
	} stats;
};


#endif // SOLVER_BASE_H
