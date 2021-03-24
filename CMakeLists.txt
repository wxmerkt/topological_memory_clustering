cmake_minimum_required(VERSION 3.0.2)
project(topological_memory_clustering)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

# C++17 required for elliptic integrals and std::reduce
# -std=c++1z may work with clang
# -O3 -ffast-math < the latter allows reordering of floating point operations
add_compile_options(-O2 -march=native)

find_package(catkin REQUIRED COMPONENTS
  pybind11_catkin
)

find_package(Eigen3 REQUIRED)

catkin_package(
  CATKIN_DEPENDS pybind11_catkin
)
    
# Python setup
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

set(SOURCES
  src/homology_clustering_py.cpp
)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Python wrapper
pybind_add_module(homology_clustering_py MODULE ${SOURCES})
target_compile_options(homology_clustering_py PRIVATE "-Wno-ignored-attributes")
install(TARGETS homology_clustering_py LIBRARY DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION})

# Install Python scripts
catkin_install_python(
  PROGRAMS 
    scripts/test_new_cpp_module
    scripts/visualise_quadcopter_dataset_in_rviz
    scripts/create_screenshot
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
