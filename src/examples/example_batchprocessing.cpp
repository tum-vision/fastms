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

#include "example_batchprocessing.h"

#include "solver/solver.h"
#include "param.h"
#ifndef DISABLE_OPENCV
#include "util/image_mat.h"
#endif // DISABLE_OPENCV
#include <iostream>
#include <vector>
#include "util.h"



int example_batchprocessing(int argc, char **argv)
{
    bool show_help = false;
	get_param("help", show_help, argc, argv);
	get_param("-help", show_help, argc, argv);
	get_param("h", show_help, argc, argv);
	get_param("-h", show_help, argc, argv);
    if (show_help) { std::cout << "Usage: " << argv[0] << " -i <inputfiles>" << std::endl; return 0; }

    // get params
    Par par;
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
    get_param("edges", par.edges, argc, argv);
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
    bool save_result = (save_dir != "");
    if (par.verbose && save_result) { std::cout << "  save (RESULTS DIRECTORY): " << save_dir.c_str() << std::endl; }
    else { std::cout << "  save (results directory): empty (result saving disabled))" << std::endl; }

    std::cout << std::endl;

#ifndef DISABLE_OPENCV
    // get input files
    std::vector<std::string> input_names;
    std::vector<cv::Mat> input_images;
    std::vector<std::string> inputfiles;
    bool has_i_param = get_param("i", inputfiles, argc, argv);
    if (!has_i_param)
    {
    	std::string default_file = "images/hepburn.png";
    	//std::cerr << "Using " << default_file << " (no option \"-i <inputfiles>\" given)" << std::endl;
    	inputfiles.push_back(default_file);
    }
	//if (par.verbose) std::cout << "loading input files" << std::endl;
	for (int i = 0; i < (int)inputfiles.size(); i++)
	{
		cv::Mat input_image = cv::imread(inputfiles[i].c_str());
		if (input_image.data == NULL) { std::cerr << "ERROR: Could not load image " << inputfiles[i].c_str() << std::endl; continue; }
		input_images.push_back(input_image);
		input_names.push_back(inputfiles[i]);
	}
    if (input_images.size() == 0)
    {
    	std::cerr << "No input files" << std::endl;
    	return -1;
    }

    // for 1d processing: extract rows
    if (row1d >= 0)
    {
        for (int i = 0; i < (int)input_images.size(); i++)
        {
        	input_images[i] = extract_row(input_images[i], row1d);
        }
    }

    // process
    std::vector<cv::Mat> result_images(input_images.size());
    Solver solver;
    for (int i = 0; i < (int)input_images.size(); i++)
    {
    	if (par.verbose) std::cout << input_names[i].c_str() << ":  ";
    	result_images[i] = solver.run(input_images[i], par);
    }

    // for 1d processing: replace 1d input and result with its graph visualization
    if (row1d >= 0)
    {
    	int graph_height = 200;
    	double thresh_jump = (par.alpha > 0.0? std::sqrt(par.lambda / par.alpha) : -1.0);
        for (int i = 0; i < (int)result_images.size(); i++)
        {
        	input_images[i] = image1d_to_graph(input_images[i], graph_height, -1.0);
        	result_images[i] = image1d_to_graph(result_images[i], graph_height, thresh_jump);
        }
    }


    // show results
    if (save_result)
    {
        for (int i = 0; i < (int)input_images.size(); i++)
        {
        	std::string dir;
			std::string basename;
			FilesUtil::to_dir_basename(input_names[i], dir, basename);
			if (row1d >= 0) { std::stringstream s; s << "_row" << row1d; basename += s.str(); }

			std::string out_dir = save_dir + '/' + dir;
			if (!FilesUtil::mkdir(out_dir)) { std::cerr << "ERROR: Could not create output directory " << out_dir.c_str() << std::endl; continue; }
			std::string out_file_input = out_dir + '/' + basename + "__input.png";
			std::string out_file_result = out_dir + '/' + basename + "__result" + par_to_string(par) + ".png";
			if (!cv::imwrite(out_file_input, input_images[i])) { std::cerr << "ERROR: Could not save input image " << out_file_input.c_str() << std::endl; continue; }
			if (!cv::imwrite(out_file_result, result_images[i])) { std::cerr << "ERROR: Could not save result image " << out_file_result.c_str() << std::endl; continue; }
			std::cout << "SAVED RESULT: " << out_file_result.c_str() << "  (SAVED INPUT: " << out_file_input.c_str() << ")" << std::endl;
        }
    }
    if (show_result)
    {
        for (int i = 0; i < (int)input_images.size(); i++)
        {
        	show_image("Input", input_images[i], 100, 100);
        	show_image("Output", result_images[i], 100 + input_images[i].cols + 40, 100);
        	cv::waitKey(0);
        }
    }
	cvDestroyAllWindows();
#else
	std::cerr << "ERROR: " << __FILE__ << ": OpenCV disabled in compilation, but this example requires OpenCV." << std::endl;
#endif // DISABLE_OPENCV

	return 0;
}
