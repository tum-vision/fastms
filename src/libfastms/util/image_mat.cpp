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

#ifndef DISABLE_OPENCV

#include "image_mat.h"
#include <iostream>



void show_image(std::string title, const cv::Mat &mat, int x_window, int y_window)
{
	if (mat.cols == 0 || mat.rows == 0 || mat.data == NULL)
	{
		std::cerr << "ERROR: show_image(): empty input cv::Mat" << std::endl;
		return;
	}
    const char *wTitle = title.c_str();
    cv::namedWindow(wTitle, CV_WINDOW_NORMAL);
    cvMoveWindow(wTitle, x_window, y_window);
    cv::imshow(wTitle, mat);
}


cv::Mat image1d_to_graph(const cv::Mat &mat, int graph_height, int channel_id, double thresh_jump)
{
	if (mat.cols == 0 || mat.rows == 0 || mat.data == NULL)
	{
		std::cerr << "ERROR: image1d_to_graph: empty input cv::Mat" << std::endl;
		return cv::Mat();
	}
	typedef float real;
	typedef ManagedImage<real, DataInterpretationLayered> managed_image_t;
	typedef managed_image_t::image_access_t image_access_t;
	managed_image_t image(MatImage(mat).dim());
	MatImage(mat).copy_to_layered(image.get_untyped_access());
	image_access_t &image_access = image.get_access();

	int w = image.dim().w;
	int num_channels = image.dim().num_channels;
	managed_image_t image_graph(ArrayDim(w, graph_height, 3));
	image_access_t &image_graph_access = image_graph.get_access();

	real val_white = real(1);
	real val_grey = real(0.15);

	for (int x = 0; x < w; x++)
	{
		real g_abs = real(0);
		for (int i = 0; i < num_channels; i++)
		{
			real diff = (x + 1 < w? image_access.get(x + 1, 0, i) - image_access.get(x, 0, i) : real(0));
			g_abs += diff * diff;
		}
		g_abs = std::sqrt(g_abs);
		bool is_jump = (real(thresh_jump) > real(0) && g_abs > real(thresh_jump));

		for (int i = 0; i < std::min(num_channels, 3); i++)
		{
			real val_cur = image_access.get(x, 0, i);
			real val_next = (x + 1 < w? image_access.get(x + 1, 0, i) : val_cur);
			int y_cur = graph_height - 1 - std::max(0, std::min(graph_height - 1, (int)std::floor(val_cur * real(graph_height))));
			int y_next = graph_height - 1 - std::max(0, std::min(graph_height - 1, (int)std::floor(val_next * real(graph_height))));
			int y0 = std::min(y_cur, y_next);
			int y1 = std::max(y_cur, y_next);
			for (int y = y0; y <= y1; y++)
			{
				real val = val_white;
				if (is_jump && y0 < y && y < y1) { val = val_grey; }
				image_graph_access.get(x, y, i) = val;
			}
		}
	}

	if (channel_id >= 0)
	{
		for (int x = 0; x < image_graph_access.dim().w; x++)
		{
			for (int y = 0; y < image_graph_access.dim().h; y++)
			{
				real val = image_graph_access.get(x, y, channel_id);
				val = real(1) - val;
				for (int i = 0; i < image_graph_access.dim().num_channels; i++)
				{
					image_graph_access.get(x, y, i) = val;
				}
			}
		}
	}

	MatImage matimage_result(image_graph.dim(), CV_8UC3);
	matimage_result.copy_from_layered(image_graph_access.get_untyped_access());
	return matimage_result.get_mat();
}


cv::Mat extract_row(const cv::Mat in_mat, int row)
{
	// row image will have the same number of channels as the input image
	int w = in_mat.cols;
	int h = in_mat.rows;
	if (row < 0 || row >= h)
	{
		int row_new = std::max(0, std::min(h - 1, row));
		std::cerr << "WARNING: extract_row: " << row << " is not a valid row (0 .. " << h - 1 << "), using row = " << row_new << std::endl;
		row = row_new;
	}
	cv::Mat mat_row(1, w, in_mat.type(), cv::Scalar::all(0));
	memcpy(mat_row.data, in_mat.ptr(row), (size_t)w * in_mat.elemSize());
	return mat_row;
}


#endif // not DISABLE_OPENCV
