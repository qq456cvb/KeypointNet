cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(iss3d)
find_package(PCL 1.2 REQUIRED)
find_package(ZLIB)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${ZLIB_INCLUDE_DIRS})
link_directories(${ZLIB_LIBRARIES})
add_definitions(${ZLIB_DEFINITIONS})

add_executable(iss3d main.cpp)
target_link_libraries(iss3d ${PCL_LIBRARIES} ${ZLIB_LIBRARIES})

SET(COMPILE_FLAGS "-std=c++11 -g")
add_definitions(${COMPILE_FLAGS})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -luuid -Wabi")
add_definitions(${CMAKE_CXX_FLAGS})
