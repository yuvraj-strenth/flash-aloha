cmake_minimum_required(VERSION 3.16.3)

find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# System pybind11 2.11 looks up a NumPy 1 API slot that is null on NumPy 2, so
# assigning a NumPy array to an Eigen vector segfaults. Use the interpreter's
# pybind11 (2.12 or newer) for the Eigen casters.
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE ARX_PYBIND11_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE ARX_PYBIND11_RC
)
if(NOT ARX_PYBIND11_RC EQUAL 0 OR NOT IS_DIRECTORY "${ARX_PYBIND11_INCLUDE}")
    message(FATAL_ERROR
        "Install pybind11>=2.12 for ${Python3_EXECUTABLE}; NumPy 2 Eigen assignment requires it")
endif()
find_package(Eigen3 REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS container)
find_package(tinyxml2 REQUIRED)

if(TARGET kdl_parser::kdl_parser)
    set(KDL_PARSER_LIB kdl_parser::kdl_parser)
else()
    set(KDL_PARSER_LIB kdl_parser)
endif()

find_library(SOEM_LIBRARY
    NAMES soem
    HINTS
        $ENV{CONDA_PREFIX}/lib
    REQUIRED
)

pybind11_add_module(arx5_interface
    arx5_pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/app/joint_controller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/app/cartesian_controller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/app/cartesian_ik.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/app/controller_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/app/cartesian_pd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/app/friction_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include/libcan)
add_compile_options(-pthread)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/SelectSdkLibDir.cmake)
arx5_resolve_lib_dir(${CMAKE_CURRENT_SOURCE_DIR}/.. LIB_DIR)

target_include_directories(arx5_interface BEFORE PRIVATE "${ARX_PYBIND11_INCLUDE}")

target_link_libraries(arx5_interface PRIVATE
    ${LIB_DIR}/libhardware.so
    ${LIB_DIR}/libsolver.so
    spdlog::spdlog
    tinyxml2::tinyxml2
    Eigen3::Eigen
    Threads::Threads
    ${KDL_PARSER_LIB}
    orocos-kdl
    pthread
    ${SOEM_LIBRARY}
)
target_include_directories(arx5_interface PUBLIC ${EIGEN3_INCLUDE_DIRS})

# Hack for py310
target_include_directories(arx5_interface PUBLIC $ENV{CONDA_PREFIX}/include/kdl_parser $ENV{CONDA_PREFIX}/include/urdfdom_headers)

# # Optional: set the output directory for the built module
set_target_properties(arx5_interface PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(arx5_interface PROPERTIES PREFIX "")
set_target_properties(arx5_interface PROPERTIES OUTPUT_NAME "arx5_interface")
