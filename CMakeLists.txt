cmake_minimum_required(VERSION 3.25)
project( ADMM )
find_package( OpenCV REQUIRED )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/inc ${OpenCV_INCLUDE_DIRS} )
add_executable( a main.cpp )
target_link_libraries( a ${OpenCV_LIBS} )
set_property(
	TARGET a
	PROPERTY ADDITIONAL_CLEAN_FILES time.csv
	)
