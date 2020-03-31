cmake_minimum_required(VERSION 2.8.12)
project(harris3d)
set (CMAKE_CXX_STANDARD 11)

find_package( PythonInterp 3.6 REQUIRED )
find_package( PythonLibs 3.6 REQUIRED )
find_package(pybind11 REQUIRED)
find_package( PCL 1.8 REQUIRED )

include_directories( ${PCL_INCLUDE_DIRS} )
link_directories( ${PCL_LIBRARY_DIRS} "D:\\PCL 1.8.1\\3rdParty\\Boost\\lib" "D:\\PCL 1.8.1\\3rdParty\\FLANN\\lib" )
add_definitions(${PCL_DEFINITIONS})

pybind11_add_module(harris3d main_.cpp)
target_link_libraries(harris3d PRIVATE ${PCL_LIBRARIES})
