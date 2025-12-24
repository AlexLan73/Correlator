cmake_minimum_required(VERSION 3.16)
project(Correlator VERSION 1.0 LANGUAGES CXX)

# Установить стандарт C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Режим сборки по умолчанию
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Флаги компиляции
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")

# Отладочные символы для Release with Debug Info
# В RelWithDebInfo тоже отключаем debug вывод для производительности
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g -march=native -DNDEBUG")

# Поиск OpenCL
find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    message(STATUS "Found OpenCL: ${OpenCL_LIBRARIES}")
    message(STATUS "OpenCL include dir: ${OpenCL_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCL not found! Please install OpenCL development libraries.")
endif()

# Поиск clFFT (системная установка)
find_path(CLFFT_INCLUDE_DIR
    NAMES clFFT.h
    PATHS /usr/local/include /usr/include
    PATH_SUFFIXES clFFT
)

find_library(CLFFT_LIBRARY
    NAMES clFFT
    PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
)

if(NOT CLFFT_INCLUDE_DIR)
    message(FATAL_ERROR "clFFT headers not found! Please install clFFT library: sudo apt install libclfft-dev")
endif()

if(NOT CLFFT_LIBRARY)
    message(FATAL_ERROR "clFFT library not found! Please install clFFT library: sudo apt install libclfft-dev")
endif()

message(STATUS "Found clFFT: ${CLFFT_LIBRARY}")
message(STATUS "Found clFFT headers: ${CLFFT_INCLUDE_DIR}")

# Включить директории
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/correlator
    ${CLFFT_INCLUDE_DIR}
    ${OpenCL_INCLUDE_DIRS}
)

# Исходные файлы
set(SOURCES
    main.cpp
    src/cpu_converter.cpp
    src/fft_handler.cpp
    src/gpu_converter.cpp
)

# Создать исполняемый файл
add_executable(${PROJECT_NAME} ${SOURCES})

# Линковка библиотек
target_link_libraries(${PROJECT_NAME}
    ${CLFFT_LIBRARY}
    ${OpenCL_LIBRARIES}
    pthread
    rt
    m
)

# Создать директории для отчетов
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/Report)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/Report/JSON)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/Report/Validation)

# Установка (опционально)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY Report/ DESTINATION share/${PROJECT_NAME}/Report)

# Вывод информации о сборке
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "OpenCL: ${OpenCL_LIBRARIES}")
message(STATUS "clFFT: ${CLFFT_LIBRARY}")
message(STATUS "========================")
message(STATUS "")
