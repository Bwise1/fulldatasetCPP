# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++11 -Wall -Wextra
CUFLAGS = -std=c++11
LDFLAGS = -lm -lcudart

# Directories
SRC_DIR = src
INCLUDE_DIR = includes
BUILD_DIR = build

# Source files
CXX_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)
SRCS = $(CXX_SRCS) $(CU_SRCS)

# Object files
CXX_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CXX_SRCS))
CU_OBJS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
OBJS = $(CXX_OBJS) $(CU_OBJS)

# Executable name and path
TARGET = main

# Default target
all: $(TARGET)

# Compile C++ source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Compile CUDA source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CUFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Link object files into the executable in the root directory
$(TARGET): $(OBJS)
	$(NVCC) $(CUFLAGS) -o $@ $^ $(LDFLAGS)

# Clean the build directory
clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean
