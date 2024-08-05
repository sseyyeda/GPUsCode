# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hossein/GPU_Programming/colorToGrayConversion

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hossein/GPU_Programming/colorToGrayConversion/build

# Include any dependencies generated for this target.
include CMakeFiles/GrayscaleConversion.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/GrayscaleConversion.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GrayscaleConversion.dir/flags.make

CMakeFiles/GrayscaleConversion.dir/GrayscaleConversion_generated_grayscale.cu.o: CMakeFiles/GrayscaleConversion.dir/GrayscaleConversion_generated_grayscale.cu.o.depend
CMakeFiles/GrayscaleConversion.dir/GrayscaleConversion_generated_grayscale.cu.o: CMakeFiles/GrayscaleConversion.dir/GrayscaleConversion_generated_grayscale.cu.o.cmake
CMakeFiles/GrayscaleConversion.dir/GrayscaleConversion_generated_grayscale.cu.o: ../grayscale.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/hossein/GPU_Programming/colorToGrayConversion/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/GrayscaleConversion.dir/GrayscaleConversion_generated_grayscale.cu.o"
	cd /home/hossein/GPU_Programming/colorToGrayConversion/build/CMakeFiles/GrayscaleConversion.dir && /usr/bin/cmake -E make_directory /home/hossein/GPU_Programming/colorToGrayConversion/build/CMakeFiles/GrayscaleConversion.dir//.
	cd /home/hossein/GPU_Programming/colorToGrayConversion/build/CMakeFiles/GrayscaleConversion.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/hossein/GPU_Programming/colorToGrayConversion/build/CMakeFiles/GrayscaleConversion.dir//./GrayscaleConversion_generated_grayscale.cu.o -D generated_cubin_file:STRING=/home/hossein/GPU_Programming/colorToGrayConversion/build/CMakeFiles/GrayscaleConversion.dir//./GrayscaleConversion_generated_grayscale.cu.o.cubin.txt -P /home/hossein/GPU_Programming/colorToGrayConversion/build/CMakeFiles/GrayscaleConversion.dir//GrayscaleConversion_generated_grayscale.cu.o.cmake

CMakeFiles/GrayscaleConversion.dir/main.cpp.o: CMakeFiles/GrayscaleConversion.dir/flags.make
CMakeFiles/GrayscaleConversion.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hossein/GPU_Programming/colorToGrayConversion/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/GrayscaleConversion.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GrayscaleConversion.dir/main.cpp.o -c /home/hossein/GPU_Programming/colorToGrayConversion/main.cpp

CMakeFiles/GrayscaleConversion.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GrayscaleConversion.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hossein/GPU_Programming/colorToGrayConversion/main.cpp > CMakeFiles/GrayscaleConversion.dir/main.cpp.i

CMakeFiles/GrayscaleConversion.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GrayscaleConversion.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hossein/GPU_Programming/colorToGrayConversion/main.cpp -o CMakeFiles/GrayscaleConversion.dir/main.cpp.s

CMakeFiles/GrayscaleConversion.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/GrayscaleConversion.dir/main.cpp.o.requires

CMakeFiles/GrayscaleConversion.dir/main.cpp.o.provides: CMakeFiles/GrayscaleConversion.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/GrayscaleConversion.dir/build.make CMakeFiles/GrayscaleConversion.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/GrayscaleConversion.dir/main.cpp.o.provides

CMakeFiles/GrayscaleConversion.dir/main.cpp.o.provides.build: CMakeFiles/GrayscaleConversion.dir/main.cpp.o


# Object files for target GrayscaleConversion
GrayscaleConversion_OBJECTS = \
"CMakeFiles/GrayscaleConversion.dir/main.cpp.o"

# External object files for target GrayscaleConversion
GrayscaleConversion_EXTERNAL_OBJECTS = \
"/home/hossein/GPU_Programming/colorToGrayConversion/build/CMakeFiles/GrayscaleConversion.dir/GrayscaleConversion_generated_grayscale.cu.o"

GrayscaleConversion: CMakeFiles/GrayscaleConversion.dir/main.cpp.o
GrayscaleConversion: CMakeFiles/GrayscaleConversion.dir/GrayscaleConversion_generated_grayscale.cu.o
GrayscaleConversion: CMakeFiles/GrayscaleConversion.dir/build.make
GrayscaleConversion: /usr/local/cuda-11.0/lib64/libcudart_static.a
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/librt.so
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
GrayscaleConversion: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
GrayscaleConversion: CMakeFiles/GrayscaleConversion.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hossein/GPU_Programming/colorToGrayConversion/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable GrayscaleConversion"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GrayscaleConversion.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GrayscaleConversion.dir/build: GrayscaleConversion

.PHONY : CMakeFiles/GrayscaleConversion.dir/build

CMakeFiles/GrayscaleConversion.dir/requires: CMakeFiles/GrayscaleConversion.dir/main.cpp.o.requires

.PHONY : CMakeFiles/GrayscaleConversion.dir/requires

CMakeFiles/GrayscaleConversion.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GrayscaleConversion.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GrayscaleConversion.dir/clean

CMakeFiles/GrayscaleConversion.dir/depend: CMakeFiles/GrayscaleConversion.dir/GrayscaleConversion_generated_grayscale.cu.o
	cd /home/hossein/GPU_Programming/colorToGrayConversion/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hossein/GPU_Programming/colorToGrayConversion /home/hossein/GPU_Programming/colorToGrayConversion /home/hossein/GPU_Programming/colorToGrayConversion/build /home/hossein/GPU_Programming/colorToGrayConversion/build /home/hossein/GPU_Programming/colorToGrayConversion/build/CMakeFiles/GrayscaleConversion.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/GrayscaleConversion.dir/depend
