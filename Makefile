#
# This file is part of fastms.
#
# 2014 Evgeny Strekalovskiy <evgeny dot strekalovskiy at in dot tum dot de> (Technical University of Munich)
#
# fastms is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# fastms is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with fastms. If not, see <http://www.gnu.org/licenses/>.
# 

all: targets

USE_CUDA:=1
USE_OPENMP:=1
USE_OPENCV:=1
USE_MEX:=1

TMP_DIR:=tmp



LIBS:=
DEFINES:=
INCLUDES:=
TARGETS:=

SOLVER_SOURCE_DIR:=src/libfastms
INCLUDES += -I$(SOLVER_SOURCE_DIR)


# check if mac or linux
MAC:=
UNAME:=$(shell uname)
ifeq ($(UNAME), Darwin)
    MAC:=1
else ifeq ($(UNAME), Linux)
else
    $(error Unexpected system: $(UNAME))
endif


# c++
ifeq ($(MAC), 1)
    GXX:=/usr/local/bin/g++-4.2
else
    GXX:=g++
endif
ARGS_GXX:=
ARGS_GXX += -Wall
ARGS_GXX += -O3
ARGS_GXX += -m64
ARGS_GXX += -fPIC
ifeq ($(USE_OPENMP), 1)
	ARGS_GXX += -fopenmp
endif
COMMAND_COMPILE_GXX=$(GXX) -c -o $@ $< $(ARGS_GXX) $(INCLUDES) $(DEFINES)
COMMAND_GET_DEPENDENCIES_GXX=@$(GXX) -M $< $(ARGS_GXX) $(INCLUDES) $(DEFINES) > $@.dep


# cuda
ifeq ($(USE_CUDA), 1)
    NVCC:=$(shell which nvcc)
    ifndef NVCC
        $(info WARNING: NVCC not in current path, disabling CUDA in compilation.)
        USE_CUDA:=
    endif
endif
ifeq ($(USE_CUDA), 1)
    CUDA_DIR:=$(shell dirname $(NVCC))/..
    ifeq ($(MAC), 1)
        CUDA_LIB_DIR:=$(CUDA_DIR)/lib
    else
        CUDA_LIB_DIR:=$(CUDA_DIR)/lib64
    endif
    INCLUDES += -I$(CUDA_DIR)/include
    LIBS += -L$(CUDA_LIB_DIR) -lcudart -lcublas
    ARGS_NVCC:=
    ARGS_NVCC += -ccbin $(GXX)
    ARGS_NVCC += --use_fast_math
    ARGS_NVCC += --compiler-options '$(ARGS_GXX)'
    #ARGS_NVCC += -gencode arch=compute_11,code=compute_11
    ARGS_NVCC += -gencode arch=compute_20,code=compute_20
    ARGS_NVCC += -gencode arch=compute_30,code=\"compute_30,sm_30\"
    #ARGS_NVCC += -gencode arch=compute_35,code=\"compute_35,sm_35\"
    #ARGS_NVCC += --ptxas-options=-v
    COMMAND_NVCC_COMPILE=$(NVCC) -c -o $@ $< $(ARGS_NVCC) $(INCLUDES) $(DEFINES) 
    COMMAND_GET_DEPENDENCIES_NVCC=@$(NVCC) -M $< $(ARGS_NVCC) $(INCLUDES) $(DEFINES) > $@.dep
else
    DEFINES += -DDISABLE_CUDA
endif


# openmp  
ifeq ($(USE_OPENMP), 1)
    LIBS += -lgomp
else
    DEFINES += -DDISABLE_OPENMP
endif


# opencv 
ifeq ($(USE_OPENCV), 1)
    OPENCV_EXISTS:=$(shell pkg-config --exists opencv; echo $$?)
    ifneq ($(OPENCV_EXISTS), 0)
        $(info WARNING: OpenCV not found, disabling OpenCV in compilation.)
        USE_OPENCV:=
    endif
endif
ifeq ($(USE_OPENCV), 1)
    LIBS += -lopencv_highgui -lopencv_core
else
    DEFINES += -DDISABLE_OPENCV
endif



# target: solver
SOLVER_SOURCES:=
SOLVER_SOURCES += $(shell find $(SOLVER_SOURCE_DIR) -name '*.cpp')
ifeq ($(USE_CUDA), 1)
    SOLVER_SOURCES += $(shell find $(SOLVER_SOURCE_DIR) -name '*.cu')
endif
SOLVER_OBJECTS:=$(foreach file, $(SOLVER_SOURCES), $(TMP_DIR)/$(file).o)
SOLVER_DEPENDENCIES:=$(foreach file, $(SOLVER_OBJECTS), $(file).dep)
-include $(SOLVER_DEPENDENCIES)
SOLVER_TARGET:=$(TMP_DIR)/$(SOLVER_SOURCE_DIR)/libfastms.o
TARGETS += $(SOLVER_TARGET)
COMMAND_LINK_SOLVER=ld -r -o $@ $^



# target: main
MAIN_SOURCES:=
MAIN_SOURCES += $(shell find src/examples -name '*.cpp')
MAIN_OBJECTS:=$(foreach file, $(MAIN_SOURCES), $(TMP_DIR)/$(file).o)
MAIN_OBJECTS += $(SOLVER_TARGET)
MAIN_DEPENDENCIES:=$(foreach file, $(MAIN_OBJECTS), $(file).dep)
-include $(MAIN_DEPENDENCIES)
MAIN_TARGET:=main
TARGETS += $(MAIN_TARGET)
COMMAND_LINK_MAIN=$(GXX) -o $@ $^ $(LIBS)



# mex
ifeq ($(USE_MEX), 1)
    ifeq ($(MAC), 1)
    	MATLAB_DIR:=/Applications/MATLAB_R2014a.app
    else
    	MATLAB_DIR:=/usr/local/lehrstuhl/DIR/matlab-R2013b
    endif
    ifeq ($(wildcard $(MATLAB_DIR)/bin/mex),)
        $(info WARNING: Did not find MATLAB in the specified directory $(MATLAB_DIR), disabling mex target compilation.)
        USE_MEX:=
    endif
endif
ifeq ($(USE_MEX), 1)
    ifeq ($(MAC), 1)
        MEX_SUFFIX:=mexmaci64
        SHARED_LIB_EXT:=dylib
    else
        MEX_SUFFIX:=mexa64
        SHARED_LIB_EXT:=so
    endif
    
    MEX_SOURCES:=$(shell find src/mex -name '*.cpp')
    MEX_OBJECTS:=$(foreach file, $(MEX_SOURCES), $(TMP_DIR)/$(file).o)
    MEX_OBJECTS += $(SOLVER_TARGET)
    MEX_DEPENDENCIES:=$(foreach file, $(MEX_OBJECTS), $(file).dep)
    -include $(MEX_DEPENDENCIES)

    MEX_DEFINES := $(DEFINES)

    MEX_INCLUDES :=$(INCLUDES) -I$(MATLAB_DIR)/extern/include
     
    MATLAB_LIB_DIR:=$(shell dirname `find $(MATLAB_DIR)/bin -name libmex.$(SHARED_LIB_EXT)`)
    MEX_LIBS:=$(LIBS) -L$(MATLAB_LIB_DIR) -lmex -lmx
    
    MEX_TARGET:=examples/matlab/fastms_mex.$(MEX_SUFFIX)
	TARGETS += $(MEX_TARGET)
	
    COMMAND_COMPILE_GXX_MEX=$(GXX) -c -o $@ $< $(ARGS_GXX) $(MEX_INCLUDES) $(MEX_DEFINES)
    COMMAND_GET_DEPENDENCIES_GXX_MEX=@$(GXX) -M $< $(ARGS_GXX) $(MEX_INCLUDES) $(MEX_DEFINES) > $@.dep
    COMMAND_LINK_MEX=$(GXX) -o $@ $^ -shared $(MEX_LIBS)
else
	DEFINES += -DDISABLE_MEX
endif 



# common commands
COMMAND_POSTPROCESS_DEPENDENCIES=@echo $@:`sed 's/.*://' $@.dep | tr "\n" " " | sed 's/\\\\/ /g'` > $@.dep; sed -e 's/^.*://' -e 's/  */::/g' $@.dep | tr ":" "\n" | sed -e 's/$$/:/' -e 's/^:$$//' >> $@.dep; echo >> $@.dep
COMMAND_MAKE_TARGET_DIR=@mkdir -p $(shell dirname $@)
COMMAND_CLEAN=@rm -rf $(TMP_DIR) $(TARGETS)



targets: $(TARGETS)



# solver
$(TMP_DIR)/$(SOLVER_SOURCE_DIR)/%.cpp.o: $(SOLVER_SOURCE_DIR)/%.cpp Makefile
	$(COMMAND_MAKE_TARGET_DIR)
	$(COMMAND_COMPILE_GXX)
	$(COMMAND_GET_DEPENDENCIES_GXX)
	$(COMMAND_POSTPROCESS_DEPENDENCIES)

$(TMP_DIR)/$(SOLVER_SOURCE_DIR)/%.cu.o: $(SOLVER_SOURCE_DIR)/%.cu Makefile
	$(COMMAND_MAKE_TARGET_DIR)
	$(COMMAND_NVCC_COMPILE)
	$(COMMAND_GET_DEPENDENCIES_NVCC)
	$(COMMAND_POSTPROCESS_DEPENDENCIES) 

$(SOLVER_TARGET): $(SOLVER_OBJECTS)
	$(COMMAND_MAKE_TARGET_DIR)
	$(COMMAND_LINK_SOLVER)



# main
$(TMP_DIR)/src/examples/%.cpp.o: src/examples/%.cpp Makefile
	$(COMMAND_MAKE_TARGET_DIR)
	$(COMMAND_COMPILE_GXX)
	$(COMMAND_GET_DEPENDENCIES_GXX)
	$(COMMAND_POSTPROCESS_DEPENDENCIES)
	 
$(MAIN_TARGET): $(MAIN_OBJECTS)
	$(COMMAND_MAKE_TARGET_DIR)
	$(COMMAND_LINK_MAIN)



# mex
$(TMP_DIR)/src/mex/%.cpp.o: src/mex/%.cpp Makefile
	$(COMMAND_MAKE_TARGET_DIR)
	$(COMMAND_COMPILE_GXX_MEX)
	$(COMMAND_GET_DEPENDENCIES_GXX_MEX)
	$(COMMAND_POSTPROCESS_DEPENDENCIES)
	
$(MEX_TARGET): $(MEX_OBJECTS)
	$(COMMAND_MAKE_TARGET_DIR)
	$(COMMAND_LINK_MEX)



clean:
	$(COMMAND_CLEAN)
