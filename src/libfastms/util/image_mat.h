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

#ifndef UTIL_IMAGE_MAT_H
#define UTIL_IMAGE_MAT_H

#ifndef DISABLE_OPENCV

#include "image.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



void show_image(std::string title, const cv::Mat &mat, int x_window, int y_window);
cv::Mat image1d_to_graph(const cv::Mat &mat, int graph_height, int channel_id = -1, double thresh_jump = -1.0);
cv::Mat extract_row(const cv::Mat mat, int row);  // extract a row from a 2d image, to process this row as a single 1d image



template<typename T> struct MatDepth { static const int value = cv::DataDepth<T>::value; };

class MatImage: public BaseImage
{
public:
	MatImage(cv::Mat image) { mat = image; }
	MatImage(const ArrayDim &dim, int depth) { mat = cv::Mat(dim.h, dim.w, CV_MAKETYPE(depth, dim.num_channels), cv::Scalar::all(0)); }
	virtual ~MatImage() {}

	virtual BaseImage* new_of_same_type_and_size() const { return new MatImage(dim(), mat.depth()); }
	virtual ArrayDim dim() const { return ArrayDim(mat.cols, mat.rows, mat.channels()); }
	virtual void copy_from_layered(const ImageUntypedAccess<DataInterpretationLayered> &in) { copy_image(this->get_untyped_access(), in); }
	virtual void copy_to_layered(ImageUntypedAccess<DataInterpretationLayered> out) const { copy_image(out, this->get_untyped_access()); }

	cv::Mat get_mat() const { return mat; }

private:
	typedef ImageUntypedAccess<DataInterpretationInterlacedReversed> image_untyped_access_t;
	image_untyped_access_t get_untyped_access() const
	{
		return image_untyped_access_t(get_data(), dim(), elem_kind(), true);  // true = on_host
	}

	void* get_data() const { return (void*)mat.data; }
	ElemKind elem_kind() const
	{
		int mat_depth = mat.depth();
		switch (mat_depth)
		{
			case MatDepth<unsigned char>::value: { return elem_kind_uchar; }
			case MatDepth<float>::value: { return elem_kind_float; }
			case MatDepth<double>::value: { return elem_kind_double; }
			default: { std::cerr << "ERROR: MatImage::elem_kind(): Unexpected cv::Mat depth " << mat_depth << std::endl; return elem_kind_uchar; }
		}
	}

	cv::Mat mat;
};



#endif // not DISABLE_OPENCV

#endif // UTIL_IMAGE_MAT_H
