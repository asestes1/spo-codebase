cmake_minimum_required(VERSION 3.16)
set (CMAKE_CXX_STANDARD 11)


project(gurobi-template)

option(CXX "enable C++ compilation" ON)
if(CXX)
    enable_language(CXX)
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Visual Studio compiler with static runtime libraries
if(MSVC AND MT)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
endif()

find_package(GUROBI REQUIRED)
find_package(Catch2 REQUIRED)
find_package(Boost 1.65 COMPONENTS program_options REQUIRED)
find_package(OpenCV REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${GUROBI_INCLUDE_DIRS} ./include ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_library(lrlib SHARED ./src/linreg.cpp ./include/oafitting/linreg.hpp)
add_library(oafitiolib SHARED ./src/oafitio.cpp ./include/oafitting/oafitio.hpp)
add_library(oafitlib SHARED ./src/oafit.cpp ./include/oafitting/oafit.hpp)
add_library(vendorlib SHARED ./src/vendor.cpp ./include/oafitting/vendor.hpp)
add_library(transportlib SHARED ./src/transport.cpp ./include/oafitting/transport.hpp)
add_library(stochprogramlib SHARED ./src/stochprogram.cpp ./include/oafitting/stochprogram.hpp)


add_executable(unittests ./test/oafit_tests.cpp)
add_executable(runoafit ./src/runoafit.cpp)
add_executable(lrrun ./src/runlr.cpp)
add_executable(rfrun ./src/runrf.cpp)
add_executable(runtrial ./src/runtrial.cpp)


if(CXX)
    set(CMAKE_CXX_STANDARD 11)
    target_link_libraries(oafitlib optimized ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})
    target_link_libraries(vendorlib optimized ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})
    target_link_libraries(transportlib optimized ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})
    target_link_libraries(unittests optimized ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})
    target_link_libraries(runoafit optimized ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})
endif()

target_link_libraries(lrlib Eigen3::Eigen oafitlib ${OpenCV_LIBS})
target_link_libraries(oafitlib Eigen3::Eigen ${GUROBI_LIBRARY})
target_link_libraries(oafitiolib Eigen3::Eigen oafitlib transportlib vendorlib)
target_link_libraries(stochprogramlib Eigen3::Eigen oafitlib lrlib)
target_link_libraries(vendorlib Eigen3::Eigen ${GUROBI_LIBRARY})
target_link_libraries(transportlib oafitlib stochprogramlib Eigen3::Eigen ${GUROBI_LIBRARY})


target_link_libraries(unittests ${GUROBI_LIBRARY} Eigen3::Eigen Catch2::Catch2WithMain oafitlib vendorlib transportlib)

target_link_libraries(runoafit Boost::program_options Eigen3::Eigen oafitiolib)
target_link_libraries(lrrun Boost::program_options Eigen3::Eigen oafitiolib lrlib)
target_link_libraries(rfrun Boost::program_options Eigen3::Eigen oafitiolib ${OpenCV_LIBS})
target_link_libraries(runtrial Boost::program_options Eigen3::Eigen oafitiolib ${OpenCV_LIBS} oafitiolib lrlib)


if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    include(FeatureSummary)
    feature_summary(WHAT ALL)
endif()
