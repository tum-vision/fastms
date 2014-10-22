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

#ifndef EXAMPLES_UTIL_H
#define EXAMPLES_UTIL_H

#include <string>
#include <cstdlib>  // for system()
#include "solver/solver.h"



class FilesUtil
{
public:
	static void to_dir_basename(const std::string filename, std::string &dir, std::string &basename)
	{
		std::string ext = "";
		to_dir_basename_ext(filename, dir, basename, ext);
	}

	static void to_dir_basename_ext(const std::string filename, std::string &dir, std::string &basename, std::string &ext)
	{
		// filename = dir/name
		dir = ".";
		std::string name = filename;
		size_t ind_slash = filename.rfind('/');
		if (ind_slash != std::string::npos)
		{
			dir = filename.substr(0, ind_slash);
			name = filename.substr(ind_slash + 1, filename.length() - (ind_slash + 1));
		}

		// name = basename.ext
		basename = name;
		ext = "";
		size_t ind_dot = basename.rfind('.');
		if (ind_dot != std::string::npos)
		{
			basename = name.substr(0, ind_dot);
			ext = name.substr(ind_dot + 1, name.length() - (ind_dot + 1));
		}
	}

	static bool mkdir(const std::string dir)
	{
		return (system(("mkdir -p " + dir).c_str()) == 0);
	}

};


std::string inline par_to_string(const Par &par)
{
    std::stringstream str_params;
    str_params << "_alpha"; if (par.alpha >= 0.0) { str_params << par.alpha; } else { str_params << "Infinity"; }
    str_params << "_lambda"; if (par.lambda >= 0.0) { str_params << par.lambda; } else { str_params << "Infinity"; }
    if (par.temporal > 0.0 || par.temporal < 0.0)
    {
    	str_params << "_temporal"; if (par.temporal >= 0.0) { str_params << par.temporal; } else { str_params << "Infinity"; }
    }
	if (par.weight)
	{
		str_params << "_weighting";
	}
	if (par.edges)
	{
		str_params << "_edges";
	}
	return str_params.str();
}


std::string inline str_curtime()
{
	time_t rawtime;
	time (&rawtime);
	tm *timeinfo = localtime(&rawtime);
	char buffer[100];
	strftime(buffer, sizeof(buffer), "%I_%M_%S", timeinfo);
	return std::string(buffer);
}


#endif // EXAMPLES_UTIL_H
