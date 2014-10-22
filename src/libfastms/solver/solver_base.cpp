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

#include "solver_base.h"
#include <cstdio>  // for snprintf
#include "util/timer.h"



template<typename real>
SolverBase<real>::SolverBase()
{
	engine = NULL;
	u_is_computed = false;
}


template<typename real>
SolverBase<real>::~SolverBase()
{
}


template<typename real>
void SolverBase<real>::set_engine(Engine<real> *engine)
{
	this->engine = engine;
}


template<typename real>
size_t SolverBase<real>::alloc(const ArrayDim &dim_u)
{
	engine->alloc(dim_u);
	const ArrayDim &dim_p = pd_vars.linear_operator.dim_range(dim_u);
	size_t mem = arr.alloc(engine, dim_u, dim_p);
	if (mem > 0) { u_is_computed = false; }
	return mem;
}


template<typename real>
void SolverBase<real>::free()
{
	arr.free(engine);
	engine->free();
}


template<typename real>
void SolverBase<real>::init(const BaseImage *image)
{
	image->copy_to_layered(arr.f.get_untyped_access());
	if (par.temporal == real(0)) { u_is_computed = false; }
	if (u_is_computed)
	{
		engine->image_manager()->copy_from_samekind(arr.prev_u, arr.u);
	}
	engine->image_manager()->copy_from_samekind(arr.u, arr.f);
	engine->image_manager()->copy_from_samekind(arr.ubar, arr.u);
	engine->image_manager()->setzero(arr.p);
    if (par.weight)
    {
	    set_regularizer_weight_from(arr.f);
    }
    pd_vars.init(par, arr.f, arr.regularizer_weight, (u_is_computed? arr.prev_u : image_access_t()));
}


template<typename real>
void SolverBase<real>::set_regularizer_weight_from(image_access_t image)
{
	linear_operator_t linear_operator;
	const Dim2D &dim2d = image.dim().dim2d();

	// real gamma = real(1);
    engine->set_regularizer_weight_from__normgrad(arr.regularizer_weight, image, linear_operator);
	real sigma = engine->get_sum(arr.regularizer_weight) / (real(dim2d.w) * real(dim2d.h));

    real coeff = (sigma > real(0)? real(2) / sigma : real(0));  // 2 = dim_image_domain
    engine->set_regularizer_weight_from__exp (arr.regularizer_weight, coeff);
}


template<typename real>
real SolverBase<real>::energy()
{
	engine->energy_base(arr.u, arr.aux_reduce, pd_vars.linear_operator, pd_vars.dataterm, pd_vars.regularizer);
	real energy = engine->get_sum(arr.aux_reduce);
    real mult = real(1) / (pd_vars.scale_omega * pd_vars.scale_omega);
	energy *= mult;
    return energy;
}


template<typename real>
real SolverBase<real>::diff_l1(image_access_t a, image_access_t b)
{
	engine->diff_l1_base(a, b, arr.aux_reduce);
	real diff = engine->get_sum(arr.aux_reduce);
	const Dim2D &dim2d = a.dim().dim2d();
	diff /= (size_t)dim2d.w * dim2d.h;
	return diff;
}
template<typename real>
bool SolverBase<real>::is_converged(int iteration)
{
	if (par.stop_k <= 0 || (iteration + 1) % par.stop_k != 0)
	{
		return false;
	}
	real diff_to_prev = diff_l1(arr.u, arr.ubar) / pd_vars.theta_bar;
	return (diff_to_prev <= par.stop_eps);
}


template<typename real>
BaseImage* SolverBase<real>::get_solution(const BaseImage *image)
{
	engine->image_manager()->copy_from_samekind(arr.aux_result, arr.u);
	if (par.edges)
	{
		engine->add_edges(arr.aux_result, pd_vars.linear_operator, pd_vars.regularizer);
	}
    BaseImage* out_image = image->new_of_same_type_and_size();
	out_image->copy_from_layered(arr.aux_result.get_untyped_access());
    return out_image;
}


template<typename real>
void SolverBase<real>::print_stats()
{
	if (stats.mem > 0)
	{
		std::cout << "alloc " << (stats.mem + (1<<20) - 1) / (1<<20) << " MB for ";
		std::cout << stats.dim_u << ",  ";
	}
	std::string str_from_engine = engine->str();
	if (str_from_engine != "") { std::cout << str_from_engine.c_str() << ", "; }
	char buffer[100];
	snprintf(buffer, sizeof(buffer), "%2.4f s compute / %2.4f s all (+ %2.4f)", stats.time_compute, stats.time, stats.time - stats.time_compute); std::cout << buffer;
	if (stats.num_runs > 1)
	{
		snprintf(buffer, sizeof(buffer), ", average %2.4f s / %2.4f s (+ %2.4f)", stats.time_compute_sum / stats.num_runs, stats.time_sum / stats.num_runs, (stats.time_sum - stats.time_compute_sum) / stats.num_runs); std::cout << buffer;
	}
	if (stats.stop_iteration != -1)
	{
		std::cout << ", " << (stats.stop_iteration + 1) << " iterations";
	}
	else
	{
		std::cout << ", did not stop after " << par.iterations << " iterations";
	}
	std::cout << ", lambda " << par.lambda;
	if (par.adapt_params) { std::cout << " (adapted " << pd_vars.regularizer.lambda << ")"; }
	std::cout << ", alpha " << par.alpha;
	if (par.adapt_params) { std::cout << " (adapted " << pd_vars.regularizer.alpha << ")"; }
	if (par.temporal > 0)
	{
		std::cout << ", temporal " << par.temporal;
	}
	if (par.weight)
	{
		std::cout << ", weighting";
	}
	std::cout << ", energy ";
	snprintf(buffer, sizeof(buffer), "%4.4f", stats.energy); std::cout << buffer;
	std::cout << std::endl;
}


template<typename real>
BaseImage* SolverBase<real>::run(const BaseImage *image, const Par &par_const)
{
	if (!engine->is_valid()) { BaseImage *out_image = image->new_of_same_type_and_size(); return out_image; }
	Timer timer_all;
	timer_all.start();

	// allocate (only if not already allocated)
	this->par = par_const;
	stats.dim_u = image->dim();
	stats.dim_p = linear_operator_t::dim_range(stats.dim_u);
    stats.mem = alloc(stats.dim_u);


    // initialize
	init(image);


	// compute
	engine->timer_start();
    stats.stop_iteration = -1;
    for (int iteration = 0; iteration < par.iterations; iteration++)
    {
    	pd_vars.update_vars();
    	engine->run_dual_p(arr.p, arr.ubar, pd_vars.linear_operator, pd_vars.regularizer, pd_vars.dt_d);
    	engine->run_prim_u(arr.u, arr.ubar, arr.p, pd_vars.linear_operator, pd_vars.dataterm, pd_vars.theta_bar, pd_vars.dt_p);
    	if (is_converged(iteration)) { stats.stop_iteration = iteration; break; }
    }
    engine->timer_end();
    u_is_computed = true;
    stats.time_compute = engine->timer_get();
    stats.time_compute_sum += stats.time_compute;
    stats.num_runs++;
    stats.energy = energy();


    // get solution
    BaseImage *result = get_solution(image);
    engine->synchronize();
    timer_all.end();
    stats.time = timer_all.get();
    stats.time_sum += stats.time;
    if (par.verbose) { print_stats(); }
    return result;
}


template class SolverBase<float>;
template class SolverBase<double>;
