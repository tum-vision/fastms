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

#include "solver_host.h"
#include "solver_base.h"
#include "util/mem.h"
#include "util/sum.h"
#include "util/timer.h"



template<typename real>
class HostEngine: public Engine<real>
{
public:
	typedef Engine<real> Base;
	typedef typename Base::image_access_t image_access_t;
	typedef typename Base::linear_operator_t linear_operator_t;
	typedef typename Base::regularizer_t regularizer_t;
	typedef typename Base::dataterm_t dataterm_t;
	typedef HostAllocator allocator_t;
	typedef ImageManager<real, typename image_access_t::data_interpretation_t, allocator_t> image_manager_t;

	HostEngine() {}
	virtual ~HostEngine() {}
	virtual std::string str();
	virtual void alloc(const ArrayDim &dim_u) {}
	virtual void free() {}
	virtual bool is_valid() { return true; }
	virtual typename Base::image_manager_base_t* image_manager() { return &image_manager_; }
	virtual real get_sum(image_access_t a) { return cpu_sum_reduce(a); }
	virtual void timer_start() { timer.start(); }
	virtual void timer_end() { timer.end(); }
	virtual double timer_get() { return timer.get(); }
	virtual void synchronize() {}

	virtual void run_dual_p(image_access_t p, image_access_t u, linear_operator_t linear_operator, regularizer_t regularizer, real dt);
	virtual void run_prim_u(image_access_t u, image_access_t ubar, image_access_t p, linear_operator_t linear_operator, dataterm_t dataterm, real theta_bar, real dt);
	virtual void energy_base(image_access_t u, image_access_t aux_reduce, linear_operator_t linear_operator, dataterm_t dataterm, regularizer_t regularizer);
	virtual void add_edges(image_access_t cur_result, linear_operator_t linear_operator, regularizer_t regularizer);
	virtual void set_regularizer_weight_from__normgrad(image_access_t regularizer_weight, image_access_t image, linear_operator_t linear_operator);
	virtual void set_regularizer_weight_from__exp(image_access_t regularizer_weight, real coeff);
	virtual void diff_l1_base(image_access_t a, image_access_t b, image_access_t aux_reduce);

	image_manager_t image_manager_;
	Timer timer;
};


template<typename real>
std::string HostEngine<real>::str()
{
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
	return "cpu with openmp";
#else
	return "cpu";
#endif
}


template<typename real>
void HostEngine<real>::run_dual_p(image_access_t p, image_access_t u, linear_operator_t linear_operator, regularizer_t regularizer, real dt)
{
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp parallel default(none) firstprivate(p, u, linear_operator, regularizer, dt)
	{
#endif
	const Dim2D &dim2d = u.dim().dim2d();
	const int u_num_channels = u.dim().num_channels;
	const int p_num_channels = linear_operator.num_channels_range(u_num_channels);
	HeapArray<real> p_sh(p_num_channels);
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp for
#endif
	for (int y = 0; y < dim2d.h; y++)
	{
		for (int x = 0; x < dim2d.w; x++)
		{
			linear_operator.apply(p_sh, u, x, y, dim2d, u_num_channels);

			for(int i = 0; i < p_num_channels; i++)
			{
				p_sh.get(i) = p.get(x, y, i) + p_sh.get(i) * dt;
			}

			regularizer.prox_star(p_sh, dt, x, y, dim2d, p_num_channels);

			for(int i = 0; i < p_num_channels; i++)
			{
				p.get(x, y, i) = p_sh.get(i);
			}
		}
	}
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
	}
#endif
}


template<typename real>
void HostEngine<real>::run_prim_u(image_access_t u, image_access_t ubar, image_access_t p, linear_operator_t linear_operator, dataterm_t dataterm, real theta_bar, real dt)
{
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp parallel default(none) firstprivate(u, ubar, p, linear_operator, dataterm, theta_bar, dt)
	{
#endif
	const Dim2D &dim2d = u.dim().dim2d();
	const int u_num_channels = u.dim().num_channels;
	HeapArray<real> u_sh(u_num_channels);
	HeapArray<real> valold_sh(u_num_channels);
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp for
#endif
	for (int y = 0; y < dim2d.h; y++)
	{
		for (int x = 0; x < dim2d.w; x++)
		{
			linear_operator.apply_transpose(u_sh, p, x, y, dim2d, u_num_channels);

			for(int i = 0; i < u_num_channels; i++)
			{
				real valold = u.get(x, y, i);
				u_sh.get(i) = valold - u_sh.get(i) * dt;
				valold_sh.get(i) = valold;
			}

			dataterm.prox(u_sh, dt, x, y, dim2d, u_num_channels);

			for(int i = 0; i < u_num_channels; i++)
			{
				real valnew = u_sh.get(i);
				u.get(x, y, i) = valnew;
				real valold = valold_sh.get(i);
				ubar.get(x, y, i) = valnew + (valnew - valold) * theta_bar;
			}
		}
	}
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
	}
#endif
}


template<typename real>
void HostEngine<real>::energy_base(image_access_t u, image_access_t aux_reduce, linear_operator_t linear_operator, dataterm_t dataterm, regularizer_t regularizer)
{
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp parallel default(none) firstprivate(u, aux_reduce, linear_operator, dataterm, regularizer)
	{
#endif
	const Dim2D &dim2d = u.dim().dim2d();
	const int u_num_channels = u.dim().num_channels;
	const int p_num_channels = linear_operator.num_channels_range(u_num_channels);
	HeapArray<real> u_sh(u_num_channels);
	HeapArray<real> p_sh(p_num_channels);
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp for
#endif
	for (int y = 0; y < dim2d.h; y++)
	{
		for (int x = 0; x < dim2d.w; x++)
		{
			real energy = real(0);
			linear_operator.apply(p_sh, u, x, y, dim2d, u_num_channels);
			energy += regularizer.value(p_sh, x, y, dim2d, p_num_channels);

			for(int i = 0; i < u_num_channels; i++)
			{
				u_sh.get(i) = u.get(x, y, i);
			}
			energy += dataterm.value(u_sh, x, y, dim2d, u_num_channels);

			aux_reduce.get(x, y, 0) = energy;
		}
	}
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    }
#endif
}


template<typename real>
void HostEngine<real>::add_edges(image_access_t image, linear_operator_t linear_operator, regularizer_t regularizer)
{
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp parallel default(none) firstprivate(image, linear_operator, regularizer)
	{
#endif
	const Dim2D &dim2d = image.dim().dim2d();
	const int u_num_channels = image.dim().num_channels;
	const int p_num_channels = linear_operator.num_channels_range(u_num_channels);
	HeapArray<real> p_sh(p_num_channels);
	const real max_range_norm = linear_operator.maximal_possible_range_norm(u_num_channels);
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp for
#endif
	for (int y = 0; y < dim2d.h; y++)
	{
		for (int x = 0; x < dim2d.w; x++)
		{
			linear_operator.apply(p_sh, image, x, y, dim2d, u_num_channels);
			real val_edge_indicator = regularizer.edge_indicator(p_sh, max_range_norm, x, y, dim2d, p_num_channels);
			real mult = real(1) - val_edge_indicator;
			for (int i = 0; i < u_num_channels; i++)
			{
				image.get(x, y, i) *= mult;
			}
		}
	}
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    }
#endif
}


template<typename real>
void HostEngine<real>::set_regularizer_weight_from__normgrad(image_access_t regularizer_weight, image_access_t image, linear_operator_t linear_operator)
{
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp parallel default(none) firstprivate(regularizer_weight, image, linear_operator)
	{
#endif
	const Dim2D &dim2d = image.dim().dim2d();
	const int u_num_channels = image.dim().num_channels;
	const int p_num_channels = linear_operator.num_channels_range(u_num_channels);
	HeapArray<real> gradient_sh(p_num_channels);
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp for
#endif
	for (int y = 0; y < dim2d.h; y++)
	{
		for (int x = 0; x < dim2d.w; x++)
		{
			linear_operator.apply(gradient_sh, image, x, y, dim2d, u_num_channels);
			real val = vec_norm(gradient_sh, p_num_channels);
			regularizer_weight.get(x, y, 0) = val;
		}
	}
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    }
#endif
}


template<typename real>
void HostEngine<real>::set_regularizer_weight_from__exp(image_access_t regularizer_weight, real coeff)
{
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp parallel default(none) firstprivate(regularizer_weight, coeff)
	{
#endif
	const Dim2D &dim2d = regularizer_weight.dim().dim2d();
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp for
#endif
	for (int y = 0; y < dim2d.h; y++)
	{
		for (int x = 0; x < dim2d.w; x++)
		{
		    static const real eps = real(1e-6);
			regularizer_weight.get(x, y, 0) = realmax(eps, realexp(-coeff * regularizer_weight.get(x, y, 0)));
		}
	}
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    }
#endif
}


template<typename real>
void HostEngine<real>::diff_l1_base(image_access_t a, image_access_t b, image_access_t aux_reduce)
{
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp parallel firstprivate(a, b, aux_reduce)
	{
#endif
	const Dim2D &dim2d = a.dim().dim2d();
	int a_num_channels = a.dim().num_channels;
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #pragma omp for
#endif
	for (int y = 0; y < dim2d.h; y++)
	{
		for (int x = 0; x < dim2d.w; x++)
		{
			real diff = real(0);
			for (int i = 0; i < a_num_channels; i++)
			{
				real val_a = a.get(x, y, i);
				real val_b = b.get(x, y, i);
				diff += realabs(val_a - val_b);
			}
			aux_reduce.get(x, y, 0) = diff;
		}
	}
#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    }
#endif
}




template<typename real>
class SolverHostImplementation: public SolverBase<real>
{
public:
	SolverHostImplementation() { SolverBase<real>::set_engine(&engine);	}
private:
	HostEngine<real> engine;
};


template<typename real> SolverHost<real>::SolverHost() : implementation(NULL) {	implementation = new SolverHostImplementation<real>(); }
template<typename real> SolverHost<real>::~SolverHost() { delete implementation; }
template<typename real> BaseImage* SolverHost<real>::run(const BaseImage *image, const Par &par) { return implementation->run(image, par); }

template class SolverHost<float>;
template class SolverHost<double>;
