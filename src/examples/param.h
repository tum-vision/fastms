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

#ifndef UTIL_PARAM_H
#define UTIL_PARAM_H

#include <cstring>  // for strcmp
#include <string>
#include <sstream>
#include <vector>



namespace
{

template<typename T> inline bool get_param__string_to_var(const std::string &s, T &var)
{
	std::stringstream ss;
	ss << s.c_str();
	ss >> var;
	return (bool)ss;
}
template<> inline bool get_param__string_to_var<std::string>(const std::string &s, std::string &var)
{
	if (s.length() > 0 && s[0] == '-')
	{
		return false;
	}
	else
	{
		var = s;
		return true;
	}
}
template<> inline bool get_param__string_to_var<bool>(const std::string &s, bool &var)
{
	if (s.length() == 0 || s[0] == 't' || s[0] == 'T' || s[0] == 'y' || s[0] == 'Y')
	{
		var = true;
		return true;
	}
	else if (s[0] == 'f' || s[0] == 'F' || s[0] == 'n' || s[0] == 'N')
	{
		var = false;
		return true;
	}
	else
	{
		std::stringstream ss;
		ss << s.c_str();
		ss >> var;
		return (bool)ss;
	}
}


template<typename T> inline void get_param__empty_to_var(T &var)
{
}
template<> inline void get_param__empty_to_var<bool>(bool &var)
{
	var = true;
}
template<> inline void get_param__empty_to_var<std::string>(std::string &var)
{
	var = "";
}

} // namespace


// get arguments for parameter
template<typename T> bool get_param(std::string param, std::vector<T> &vars, int argc, char **argv)
{
	bool param_found = false;
	vars.clear();
    for(int i = argc - 1; i >= 1; i--)
    {
        if (argv[i][0] != '-') continue;
        if (strcmp(argv[i] + 1, param.c_str()) == 0)
        {
        	param_found = true;
        	// read arguments
        	for (int k = i + 1; k < argc; k++)
        	{
                T var;
                if (!get_param__string_to_var(argv[k], var)) { break; }
                vars.push_back(var);
        	}
        	break;
        }
    }
    return param_found;
}


// get last argument for parameter
template<typename T> bool get_param(std::string param, T &var, int argc, char **argv)
{
	std::vector<T> vars;
	bool param_found = get_param(param, vars, argc, argv);
	if (param_found)
	{
		if (vars.size() > 0)
		{
			var = vars[vars.size() - 1];
		}
		else
		{
			get_param__empty_to_var(var);
		}
	}
	return param_found;
}



#endif // UTIL_PARAM_H
