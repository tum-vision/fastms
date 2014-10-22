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
#include "example_gui.h"
#include "param.h"
#include <iostream>



int main(int argc, char **argv)
{
	// example_gui: ability to change the parameters and see the result immediately
	//   --> if camera or only one input image (or no input image): example_gui
	// example_batchprocessing: set parameters once and apply them to all input images ("-i image1 image2 image3 ...")
	//   --> if multiple input images given

	bool use_cam = false;
	get_param("cam", use_cam, argc, argv);

    std::vector<std::string> inputfiles;
    get_param("i", inputfiles, argc, argv);

    if (use_cam || inputfiles.size() <= 1)
    {
		return example_gui(argc, argv);
    }
    else
    {
		return example_batchprocessing(argc, argv);
    }
}
