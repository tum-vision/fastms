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

#include "example_gui.h"

#include "solver/solver.h"
#include "param.h"
#ifndef DISABLE_OPENCV
#include "util/image_mat.h"
#endif // not DISABLE_OPENCV
#include "util.h"
#include <cmath>
#include <iostream>
#include <ctime>
#include <algorithm>




#ifndef DISABLE_OPENCV
class SliderExpTransform
{
public:
	SliderExpTransform() : min_val_exp(0.001), max_val_exp(100.0), min_val(0.0), max_val(100.0), pos_max(100), pos(0) {}
	void set_val_range(double min_val_exp, double max_val_exp, double min_val, double max_val, int pos_max)
	{
		this->min_val_exp = min_val_exp;
		this->max_val_exp = max_val_exp;
		this->min_val = min_val;
		this->max_val = max_val;
		this->pos_max = pos_max;
	}
    double pos2val(int pos)
    {
    	if (pos == 0) { return min_val; }
    	else if (pos == pos_max) { return max_val; }
    	else
    	{
        	double val = min_val_exp * pow(max_val_exp / min_val_exp, (double)pos / (double)pos_max);
        	double pwr10 = pow(10.0, floor(log(val) / log(10.0)));
        	val = floor(val / pwr10 * 100.0 + 0.5) / 100.0 * pwr10;
        	return val;
    	}
    }
    int val2pos(double val)
    {
    	if (val == min_val) { return 0; }
    	else if (val == max_val) { return pos_max; }
    	else
    	{
        	val = std::max(min_val_exp, std::min(max_val_exp, val));
        	double pos_double = log(val / min_val_exp) / log(max_val_exp / min_val_exp) * (double)pos_max;
        	int pos = (int)floor(pos_double + 0.5);
        	pos = std::max(0, std::min(pos_max, pos));
        	return pos;
    	}
    }
	double min_val_exp;
	double max_val_exp;
	double min_val;
	double max_val;
	int pos_max;
	int pos;
};


void slider_callback_lambda(int pos, void *data_void);
void slider_callback_alpha(int pos, void *data_void);
void slider_callback_temporal(int pos, void *data_void);


class Data
{
public:
	Data() : windows_created(false), windows_positioned(false), recompute_on_param_change(true), show_temporal(false), channel_id(-1) {}

	void compute()
	{
		result_image = solver.run(input_image, par);
        if (input_image.rows == 1)
        {
        	int graph_height = 200;
        	double thresh_jump = (par.alpha > 0.0? std::sqrt(par.lambda / par.alpha) : -1.0);
			shown_input_image = image1d_to_graph(input_image, graph_height, channel_id, -1.0);
			shown_result_image = image1d_to_graph(result_image, graph_height, channel_id, thresh_jump);
        }
        else
        {
        	shown_input_image = input_image;
        	shown_result_image = result_image;
        }
	}
	void compute_and_show_results()
	{
		compute();

        std::string title_input = "Input";
		std::string title_result = "Result";
		if (input_image.rows == 1)
		{
			std::string to_add = "";
			if (channel_id == 0) { to_add = " (R of RGB)"; }
			else if (channel_id == 1) { to_add = " (G of RGB)"; }
			else if (channel_id == 2) { to_add = " (B of RGB)"; }
			else { to_add = " (RGB)"; }
			title_input += to_add;
			title_result += to_add;
		}
		std::string title_params = "Parameters";
		if (!windows_created)
		{
		    cv::namedWindow(title_params.c_str(), CV_WINDOW_NORMAL);
		    slider_lambda.set_val_range(0.001, 1000, 0.0, -1.0, 500);
			slider_lambda.pos = slider_lambda.val2pos(par.lambda);
			cv::createTrackbar("Lambda", title_params.c_str(), &slider_lambda.pos, slider_lambda.pos_max, slider_callback_lambda, (void*)this);
			slider_alpha.set_val_range(0.1, 10000, 0.0, -1.0, 500);
			slider_alpha.pos = slider_alpha.val2pos(par.alpha);
			cv::createTrackbar("Alpha", title_params.c_str(), &slider_alpha.pos, slider_alpha.pos_max, slider_callback_alpha, (void*)this);
			if (show_temporal)
			{
				slider_temporal.set_val_range(0.01, 2, 0.0, 2.0, 500);
				slider_temporal.pos = slider_temporal.val2pos(par.temporal);
				cv::createTrackbar("Temporal", title_params.c_str(), &slider_temporal.pos, slider_temporal.pos_max, slider_callback_temporal, (void*)this);
			}

		    cv::namedWindow(title_input.c_str(), CV_WINDOW_NORMAL);

		    cv::namedWindow(title_result.c_str(), CV_WINDOW_NORMAL);

		    windows_created = true;
		}
		if (!windows_positioned)
		{
		    cvMoveWindow(title_input.c_str(), 100, 300);
		    cvMoveWindow(title_result.c_str(), 100 + shown_input_image.cols + 40, 300);
		    cvMoveWindow(title_params.c_str(), 100 + shown_input_image.cols + 40, 100);
			windows_positioned = true;
		}
		cv::imshow(title_input.c_str(), shown_input_image);
		cv::imshow(title_result.c_str(), shown_result_image);
	}

	Par par;
	Solver solver;
	cv::Mat input_image;
	cv::Mat result_image;
	cv::Mat shown_input_image;
	cv::Mat shown_result_image;

	SliderExpTransform slider_lambda;
	SliderExpTransform slider_alpha;
	SliderExpTransform slider_temporal;
	bool windows_created;
	bool windows_positioned;
	bool recompute_on_param_change;
	bool show_temporal;
	int channel_id;
};


void slider_callback_lambda(int pos, void *data_void)
{
	Data *data = (Data*)data_void;
	data->par.lambda = data->slider_lambda.pos2val(pos);
	if (data->recompute_on_param_change) { data->compute_and_show_results(); }
}
void slider_callback_alpha(int pos, void *data_void)
{
	Data *data = (Data*)data_void;
	data->par.alpha = data->slider_alpha.pos2val(pos);
	if (data->recompute_on_param_change) { data->compute_and_show_results(); }
}
void slider_callback_temporal(int pos, void *data_void)
{
	Data *data = (Data*)data_void;
	data->par.temporal = data->slider_temporal.pos2val(pos);
	if (data->recompute_on_param_change) { data->compute_and_show_results(); }
}


void print_keys()
{
	std::cout << "  e: edges highlighting    (on/off)\n";
	std::cout << "  v: info printing         (on/off)\n";
	std::cout << "  w: regularizer weighting (on/off)\n";
	std::cout << "  s: save result image\n";
	std::cout << "  escape: exit\n";
	std::cout << std::endl;
}


// returns: whether the result should be recomputed due to changed parameters
bool process_key(int key, Data &data, bool &running, bool &save)
{
	bool params_changed = false;
	save = false;
	if (key == 27)
	{
		running = false;
	}
	else if (key == 'e')
	{
		data.par.edges = !data.par.edges;
		params_changed = true;
	}
	else if (key == 'v')
	{
		data.par.verbose = !data.par.verbose;
		params_changed = true;
	}
	else if (key == 'w')
	{
		data.par.weight = !data.par.weight;
		params_changed = true;
	}
	else if (key == 's')
	{
		save = true;
	}
	else if (((int)('a') <= key && key <= (int)('z')) || ((int)('A') <= key && key <= (int)('Z')) || ((int)('0') <= key && key <= (int)('9')))
	{
		std::cout << "\nNo action set for key '" << (char)key << "'. You can use one of the following:" << std::endl;
		print_keys();
	}
	return params_changed;
}
#endif // not DISABLE_OPENCV


int example_gui(int argc, char **argv)
{
#ifndef DISABLE_OPENCV
	Data data;

    bool show_help = false;
	get_param("help", show_help, argc, argv);
	get_param("-help", show_help, argc, argv);
	get_param("h", show_help, argc, argv);
	get_param("-h", show_help, argc, argv);
    if (show_help) { std::cout << "Usage: " << argv[0] << " -i <inputfiles>" << std::endl; return 0; }

    // get params
    Par &par = data.par;
    get_param("verbose", par.verbose, argc, argv);
    if (par.verbose) std::cout << std::boolalpha;
    get_param("lambda", par.lambda, argc, argv);
    get_param("alpha", par.alpha, argc, argv);
    get_param("temporal", par.temporal, argc, argv);
    get_param("iterations", par.iterations, argc, argv);
    get_param("stop_eps", par.stop_eps, argc, argv);
    get_param("stop_k", par.stop_k, argc, argv);
    get_param("adapt_params", par.adapt_params, argc, argv);
    get_param("weight", par.weight, argc, argv);
    get_param("edges", par.edges, argc, argv);
    get_param("use_double", par.use_double, argc, argv);
    {
    	std::string s_engine = "";
        if (get_param("engine", s_engine, argc, argv))
        {
        	std::transform(s_engine.begin(), s_engine.end(), s_engine.begin(), ::tolower);
        	if (s_engine.find("cpu") == 0 || s_engine.find("host") == 0)
        	{
        		par.engine = Par::engine_cpu;
        	}
        	else if (s_engine.find("cuda") == 0 || s_engine.find("device") == 0)
        	{
        		par.engine = Par::engine_cuda;
        	}
        	else
        	{
        		get_param("engine", par.engine, argc, argv);
        	}
        }
    }
    if (par.verbose) { par.print(); }
    std::cout << std::endl;

    int row1d = -1;
    get_param("row1d", row1d, argc, argv);
    if (par.verbose) std::cout << "  row1d: "; if (row1d == -1) std::cout << "-1 (processing as 2d image)" << std::endl; else std::cout << "processing only row " << row1d << " as 1d image" << std::endl;

    bool show_result = true;
    get_param("show", show_result, argc, argv);
    if (par.verbose) std::cout << "  show: " << show_result << std::endl;

    std::string save_dir = "images_output";
    get_param("save", save_dir, argc, argv);
    if (par.verbose && save_dir != "") { std::cout << "  save (RESULTS DIRECTORY): " << save_dir.c_str() << std::endl; }
    else { std::cout << "  save (results directory): empty (result saving disabled))" << std::endl; }

	bool use_cam = false;
	get_param("cam", use_cam, argc, argv);
	std::cout << "  cam: " << use_cam << std::endl;

    std::cout << "\nKeys:\n";
	print_keys();

  	cv::VideoCapture *camera = NULL;
  	std::string in_dir;
	std::string in_basename;
	if (use_cam)
	{
	  	camera = new cv::VideoCapture(0);
	  	if(!camera->isOpened()) { delete camera; std::cerr << "ERROR: Could not open camera" << std::endl; return -1; }
	    int w_cam = 640;
	    int h_cam = 480;
	  	camera->set(CV_CAP_PROP_FRAME_WIDTH, w_cam);
	  	camera->set(CV_CAP_PROP_FRAME_HEIGHT, h_cam);
		data.recompute_on_param_change = false;
		data.show_temporal = true;
		in_dir = "camera";
		in_basename = "frame";
	}
	else
	{
		std::string inputfile;
	    bool has_i_param = get_param("i", inputfile, argc, argv);
	    if (!has_i_param)
	    {
	    	std::string default_file = "images/hepburn.png";
	    	inputfile = default_file;
	    }
		data.input_image = cv::imread(inputfile.c_str());
		if (data.input_image.data == NULL) { std::cerr << "ERROR: Could not load image " << inputfile.c_str() << std::endl; return -1; }
		data.recompute_on_param_change = true;
		data.show_temporal = false;
		FilesUtil::to_dir_basename(inputfile, in_dir, in_basename);
	}
	std::string out_dir = save_dir + '/' + in_dir;


	bool running = true;
	bool input_changed = true;
	bool params_changed = false;
	bool already_saved = false;
    while(running)
    {
    	if (use_cam) { *camera >> data.input_image; input_changed = true; }
    	if (input_changed)
    	{
    		if (row1d >= 0) { data.input_image = extract_row(data.input_image, row1d); }
    	}

    	if (input_changed || params_changed)
    	{
    		if (show_result) { data.compute_and_show_results(); } else { data.compute(); }
    		input_changed = false;
    		params_changed = false;
    	}

    	bool save_result = false;
    	if (show_result)
    	{
			int key = cv::waitKey( (use_cam? 15 : 0) );
			params_changed = process_key(key, data, running, save_result);
    	}
    	else
    	{
    		save_result = true;
    	}
    	if (!running && !already_saved) { save_result = true; }
		if (save_result && save_dir != "")
		{
			std::string basename = (use_cam? "frame_at_time_" + str_curtime() : in_basename);
			if (row1d >= 0) { std::stringstream s; s << "_row" << row1d; basename += s.str(); }
		    std::string out_file_input = out_dir + '/' + basename + "__input.png";
	    	std::string out_file_result = out_dir + '/' + basename + "__result" + par_to_string(data.par) + ".png";
	    	if (!FilesUtil::mkdir(out_dir)) { std::cerr << "ERROR: Could not create output directory " << out_dir.c_str() << std::endl; continue; }
	    	if (!cv::imwrite(out_file_input, data.shown_input_image)) { std::cerr << "ERROR: Could not save input image " << out_file_input.c_str() << std::endl; continue; }
	    	if (!cv::imwrite(out_file_result, data.shown_result_image)) { std::cerr << "ERROR: Could not save result image " << out_file_result.c_str() << std::endl; continue; }
	    	std::cout << "SAVED RESULT: " << out_file_result.c_str() << "  (SAVED INPUT: " << out_file_input.c_str() << ")" << std::endl;
	    	already_saved = true;
		}
		if (!show_result) { running = false; }
    }

    if (use_cam) { delete camera; }

	cvDestroyAllWindows();
#else
	std::cerr << "ERROR: " << __FILE__ << ": OpenCV disabled in compilation, but this example requires OpenCV." << std::endl;
#endif // DISABLE_OPENCV
	return 0;
}

