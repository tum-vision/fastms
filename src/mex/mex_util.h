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

#ifndef MEX_MEX_UTIL_H
#define MEX_MEX_UTIL_H

#ifndef DISABLE_MEX

#include "mex.h"
#include "util/image.h"
#include <vector>
#include <string>

//#include "util/real.h"

#if !defined(DISABLE_CUDA) && defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#define FORCEINLINE __forceinline__
#else
#define HOST_DEVICE
#define FORCEINLINE inline
#endif // !defined(DISABLE_CUDA) && defined(__CUDACC__)



template<typename T> struct MatlabClass;
template<> struct MatlabClass<unsigned char> { static const int value = mxUINT8_CLASS; };
template<> struct MatlabClass<float> { static const int value = mxSINGLE_CLASS; };
template<> struct MatlabClass<double> { static const int value = mxDOUBLE_CLASS; };


class MatlabImage: public BaseImage
{
public:
	MatlabImage() : matrix(NULL) {}
	MatlabImage(const mxArray *matrix) : matrix(const_cast<mxArray*>(matrix))
	{
		if (!matrix) { mexPrintf("ERROR: matrix is NULL\n"); return; }
		if (mxIsSparse(matrix)) { mexPrintf("ERROR: matrix must be dense\n"); return; }

		int num_dims = mxGetNumberOfDimensions(matrix);
		const mwSize *matrix_dims = mxGetDimensions(matrix);
		dims.resize(num_dims);
		for (int i = 0; i < num_dims; i++)
		{
			dims[i] = matrix_dims[i];
		}
	}
	MatlabImage(const std::vector<mwSize> &dims, mxClassID matrix_class) : matrix(NULL), dims(dims)
	{
		matrix = mxCreateNumericArray(dims.size(), dims.data(), matrix_class, mxREAL);
	}
	virtual ~MatlabImage() {}
	static mxArray* empty_matrix() { return empty_matrix(mxDOUBLE_CLASS); }
	static mxArray* empty_matrix(mxClassID matrix_class)
	{
		std::vector<mwSize> dims;
		MatlabImage matlab_image(dims, matrix_class);
		return matlab_image.get_matrix();
	}

	virtual BaseImage* new_of_same_type_and_size() const { return new MatlabImage(get_dims(), get_class()); }
	virtual ArrayDim dim() const
	{
		ArrayDim d;
		d.w = (dims.size() > 1? (int)dims[1] : 1);
		d.h = (dims.size() > 0? (int)dims[0] : 0);
		size_t num_ch = 1;
		for (int i = 2; i < (int)dims.size(); i++)
		{
			num_ch *= dims[i];
		}
		d.num_channels = num_ch;
		return d;
	}
	virtual void copy_from_layered(const ImageUntypedAccess<DataInterpretationLayered> &in) { copy_image(this->get_untyped_access(), in); }
	virtual void copy_to_layered(ImageUntypedAccess<DataInterpretationLayered> out) const { copy_image(out, this->get_untyped_access()); }

	mxArray* get_matrix() const { return matrix; }
	std::vector<mwSize> get_dims() const { return dims; }

private:
	typedef ImageUntypedAccess<DataInterpretationLayeredTransposed> image_untyped_access_t;
	image_untyped_access_t get_untyped_access() const
	{
		return image_untyped_access_t(get_data(), dim(), elem_kind(), true);  // true = on_host
	}

	mxClassID get_class() const { return mxGetClassID(matrix); }
	void* get_data() const { return (void*)mxGetPr(matrix); }
	ElemKind elem_kind() const
	{
		mxClassID matlab_class = get_class();
		switch (matlab_class)
		{
			case MatlabClass<unsigned char>::value: { return elem_kind_uchar; }
			case MatlabClass<float>::value: { return elem_kind_float; }
			case MatlabClass<double>::value: { return elem_kind_double; }
			default: { std::cerr << "ERROR: MatlabImage::elem_kind(): Unexpected matlab class " << matlab_class << std::endl; return elem_kind_uchar; }
		}
	}

	mxArray *matrix;
	std::vector<mwSize> dims;
};


template<typename T> void set_var(double value, T &var);
template<> void set_var<float>(double value, float &var) { var = (float)value; }
template<> void set_var<double>(double value, double &var) { var = value; }
template<> void set_var<bool>(double value, bool &var) { var = (value != 0.0); }
template<> void set_var<int>(double value, int &var) { var = (int)value; }


template<typename T>
void matlab_get_scalar_field(std::string name, T &var, const mxArray* matrix)
{
	if (!mxIsStruct(matrix))
	{
		std::cerr << "ERROR: matlab_get_scalar_field(" << name.c_str() << "): matrix must be a struct" << std::endl;
		return;
	}
	int index = 0;
	const mxArray* matrix_field = mxGetField(matrix, index, name.c_str());
	if (!matrix_field)
	{
		std::cout << "WARNING: matlab_get_scalar_field(" << name.c_str() << "): matrix does not contain this field" << std::endl;
		return;
	}

	if (mxIsSparse(matrix_field))
	{
		std::cerr << "ERROR: matlab_get_scalar_field(" << name.c_str() << "): matrix field must be dense" << std::endl;
		return;
	}
	if (mxGetNumberOfDimensions(matrix_field) == 2 && (mxGetM(matrix_field) == 0 || mxGetN(matrix_field) == 0))
	{
		return;
	}
	if (!(mxGetNumberOfDimensions(matrix_field) == 2 &&
		  mxGetM(matrix_field) == 1 && mxGetN(matrix_field) == 1 &&
		  !mxIsComplex(matrix_field)))
	{
		std::cerr << "ERROR: matlab_get_scalar_field(" << name.c_str() << "): matrix field must be a scalar";
		std::cerr << " (#dims = " << mxGetNumberOfDimensions(matrix_field);
		std::cerr << ", " << mxGetM(matrix_field) << " x " << mxGetN(matrix_field);
		std::cerr << ", is_complex = " << (mxIsComplex(matrix_field)? "true" : "false");
		std::cerr << std::endl;
		return;
	}

	double result = mxGetScalar(matrix_field);
	set_var(result, var);
}



#endif // not DISABLE_MEX

#endif // MEX_MEX_UTIL_H
