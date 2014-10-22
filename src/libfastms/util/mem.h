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

#ifndef UTIL_MEM_H
#define UTIL_MEM_H



template<typename T>
class HeapArray
{
public:
	typedef T elem_type;

	explicit HeapArray(int num_elem) : data(NULL)
	{
		data = new T[num_elem];
	}
	~HeapArray()
	{
		if (data) delete[] data;
		data = NULL;
	}
	T& get(int i)
	{
		return data[i];
	}
	const T& get(int i) const
	{
		return data[i];
	}

private:
	HeapArray(const HeapArray<T> &other_array);
	HeapArray<T>& operator= (const HeapArray<T> &other_array);

	T *data;
};



#endif // UTIL_MEM_H
