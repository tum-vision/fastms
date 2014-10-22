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

#ifndef DISABLE_MEX

#include "mex.h"
#include "mex_util.h"
#include "solver/solver.h"



Par parse_par(const mxArray *matrix)
{
	Par par;
	matlab_get_scalar_field("lambda", par.lambda, matrix);
	matlab_get_scalar_field("alpha", par.alpha, matrix);
	matlab_get_scalar_field("temporal", par.temporal, matrix);
	matlab_get_scalar_field("iterations", par.iterations, matrix);
	matlab_get_scalar_field("stop_eps", par.stop_eps, matrix);
	matlab_get_scalar_field("stop_k", par.stop_k, matrix);
	matlab_get_scalar_field("adapt_params", par.adapt_params, matrix);
	matlab_get_scalar_field("weight", par.weight, matrix);
	matlab_get_scalar_field("use_double", par.use_double, matrix);
	matlab_get_scalar_field("engine", par.engine, matrix);
	matlab_get_scalar_field("edges", par.edges, matrix);
	matlab_get_scalar_field("verbose", par.verbose, matrix);
	return par;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// 1st argument: input image
	// 2nd argument (optional): parameters
	if (!(nrhs == 1 || nrhs == 2)) { mexPrintf("Syntax: out_image = ms(in_image, par)\n"); return; }
	if (nlhs == 0) { return; }


	// get parameters
    Par par;
    if (nrhs >= 2) { par = parse_par(prhs[1]); }
    if (par.verbose) par.print();


    // compute
    Solver solver;
    MatlabImage in_matlabimage(prhs[0]);
    MatlabImage *out_matlabimage = static_cast<MatlabImage*>(solver.run(&in_matlabimage, par));
	plhs[0] = out_matlabimage->get_matrix();
    delete out_matlabimage;


    // set other outputs to an empty matrix
    for (int i = 1; i < nlhs; i++) { plhs[i] = MatlabImage::empty_matrix(); }
}

#endif // not DISABLE_MEX
