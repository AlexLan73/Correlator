cmake_minimum_required(VERSION 3.10)
project(OpenCL_Capability_Checker)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== НАЙТИ OpenCL =====
find_package(OpenCL REQUIRED)

# ===== СОЗДАТЬ EXECUTABLE =====
add_executable(capability_checker opencl_capability_checker.cpp)

# ===== LINK OpenCL =====
target_link_libraries(capability_checker PUBLIC OpenCL::OpenCL)

# ===== ВКЛЮЧИТЬ ПУТИ =====
target_include_directories(capability_checker PRIVATE ${OPENCL_INCLUDE_DIRS})

# ===== ОПЦИЯ: VERBOSE OUTPUT =====
option(ENABLE_VERBOSE "Enable verbose OpenCL output" ON)
if(ENABLE_VERBOSE)
    target_compile_definitions(capability_checker PRIVATE VERBOSE_MODE)
endif()

# ===== PRINT INFO =====
message(STATUS "OpenCL found!")
message(STATUS "OpenCL Include: ${OPENCL_INCLUDE_DIRS}")
message(STATUS "OpenCL Lib: ${OPENCL_LIBRARIES}")
