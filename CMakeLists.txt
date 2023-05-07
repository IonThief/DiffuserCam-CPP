cmake_minimum_required(VERSION 3.21)

# Set project name and version
project(ADMM_acceleration VERSION 0.0.2)

# Project source files
AUX_SOURCE_DIRECTORY(src SOURCE_FILES)

# Executable
add_executable(reconstruct ${SOURCE_FILES})

# Prerequisite libraries
find_package(OpenCV REQUIRED)
find_package(pylon REQUIRED)

# Project include paths
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
  ${OpenCV_INCLUDE_DIRS}
	${pylon_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(reconstruct
  ${OpenCV_LIBS}
	${pylon_LIBRARIES}
)
