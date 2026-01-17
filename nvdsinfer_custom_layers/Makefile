################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

CUDA_VER = 12.2
DS_VER = 7.0
# set your cuda version and deepstream version

OBJ_DIR = ../obj
TARGET_DIR = ../target
CUSTOM_DIR = ../gst_custom_engine

CUDA_ROOT = /home/pogg/CUDA
DS_ROOT = /home/pogg/DeepStream
# set your cuda path and deepstream path

CC:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

CFLAGS:= -Wall -std=c++17 -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I$(DS_ROOT)/opt/nvidia/deepstream/deepstream-$(DS_VER)/sources/includes
CFLAGS+= -I/usr/local/cuda-$(CUDA_VER)/include
CFLAGS+= -I$(CUDA_ROOT)/TensorRT-8.6.1.6/include
CFLAGS+= -I/usr/local/cuda-$(CUDA_VER)/targets/x86_64-linux/include/
CFLAGS+= -I$(DS_ROOT)/opt/nvidia/deepstream/deepstream-$(DS_VER)/lib/glib-2.0/include
CFLAGS+= -I$(DS_ROOT)/deepstream-yolo/nvdsinfer_custom_layers
CFLAGS+= -I$(CUSTOM_DIR)
CFLAGS+= $(shell pkg-config --cflags gstreamer-1.0)

LIBS:=  -L/usr/local/cuda-$(CUDA_VER)/lib64
LIBS+= -L$(DS_ROOT)/opt/nvidia/deepstream/deepstream-$(DS_VER)/lib
LIBS+= -L$(CUDA_ROOT)/TensorRT-8.6.1.6/lib
LIBS+= -lnvinfer_plugin -lnvinfer -lnvonnxparser -lnvparsers -lcudart -lcublas -lstdc++fs
LIBS+= -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper

CFLAGS_BIN:= $(CFLAGS)
CFLAGS_SO:= $(CFLAGS) -shared
CUFLAGS:= -std=c++17 -shared 
CUFLAGS+= $(filter -I%,$(CFLAGS))
GST_LIBS = $(shell pkg-config --libs gstreamer-1.0)

LDFLAGS_COMMON:= -Wl,--start-group $(LIBS) $(GST_LIBS) -Wl,--end-group
LDFLAGS_SO:= -shared $(LDFLAGS_COMMON)
LDFLAGS_BIN:= $(LDFLAGS_COMMON)

INCS:= $(wildcard *.h)
SRCFILES := $(wildcard *.cpp)
SRCFILES += $(wildcard *.cu)

CUSTOM_SRC = $(CUSTOM_DIR)/deepstream-pose.cpp $(CUSTOM_DIR)/perf.cpp
CUSTOM_OBJS = $(addprefix $(OBJ_DIR)/,$(notdir $(CUSTOM_SRC:.cpp=.o)))

TARGET_LIB:= $(TARGET_DIR)/nvdsinfer_custom_layers.so
TARGET_BIN:= $(TARGET_DIR)/deepstream-pose

TARGET_OBJS:= $(addprefix $(OBJ_DIR)/,$(SRCFILES:.cpp=.o))
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

all: $(TARGET_LIB) $(TARGET_BIN)

$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(TARGET_DIR))

$(OBJ_DIR)/%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS_SO) $<

$(OBJ_DIR)/%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $(CUFLAGS) $<

$(OBJ_DIR)/%.o: $(CUSTOM_DIR)/%.cpp Makefile
	$(CC) -c -o $@ $(CFLAGS_BIN) $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@ $(TARGET_OBJS) $(LDFLAGS_SO)

$(TARGET_BIN) : $(CUSTOM_OBJS)
	$(CC) -o $@ $(CUSTOM_OBJS) $(LDFLAGS_BIN)

clean:
	rm -rf $(TARGET_DIR) $(OBJ_DIR)