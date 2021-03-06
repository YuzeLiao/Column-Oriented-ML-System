# Copyright (c) 2016 by contributors.
# Authour: Chao Ma (mctt90@gmail.com)
#
# This is the root CMakeLists.txt file of the F2M project,
# which build all subdirectories (packages) in the order of
# inter-package dependence.
#
# you can build the F2M using the following commands:
#
# $> cd F2M
# $> mkdir build
# $> cd build
# $> cmake ..
# $> make
#
# Thus you check out the F2M project and build it in a
# subdirectory ``build". If you want further to install the
# built project, you can type the command
#
# $> make install
#
project ("f2m")

cmake_minimum_required(VERSION 2.8)

#-------------------------------------------------------------------------------
# Take almost all warnings;
# Take warnings as errors;
# Do not generate debug symbols;
# Optimazation level 3;
# Using c++11
#-------------------------------------------------------------------------------
add_definitions(" -Wall -Wno-sign-compare -Werror -O3 -std=c++11 -mavx")

#-------------------------------------------------------------------------------
# Declare where our project will be installed.
#-------------------------------------------------------------------------------
set(CMAKE_INSTALL_PREFIX "f2m")

#-------------------------------------------------------------------------------
# Ensure executables are statically linked with libraries.
#-------------------------------------------------------------------------------
if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc")
endif()

#-------------------------------------------------------------------------------
# Set include paths and library paths.
#
# F2M depends on the following thirdparty libraries:
#
#  - gflags (command line args parser)
#
# You can install them using package management tools on your system,
# or build them from source code by youself. We suggest you to take
# the first way. While, in either way, you need to tell  your compiler
# where these libraries were installed. If you use the package management
# tools, for example, on Mac OS X, Homebrew installs header files to
# /usr/local/include and libraries to /usr/local/lib.
# In this way, you can just use the following commands:
#
#  set(THIRD_PARTY_DIR "/usr/local")
#  set(THIRD_PARTY_HEADER "${THIRD_PARTY_DIR}/include")
#  set(THIRD_PARTY_LIB "${THIRD_PARTY_DIR}/lib")
#
# include_directories(
#  "${PROJECT_SOURCE_DIR}"
#  "${PROJECT_SOURCE_DIR}/src/gtest/include"
#  "${THIRD_PARTY_HEADER}"
# )
#
# link_directories(
#  "${PROJECT_BINARY_DIR}"
#  "${THRID_PARTY_LIB}"
# )
#
# F2M uses googletest framework for unit testing. As it is not recommand
# to build googletest as a system-wide library, we must import googletest
# into our source code tree before building F2M.
#
#-------------------------------------------------------------------------------
#
# This is the configuration of my own Mac OS X using Homebrew:
#

set(THIRD_PARTY_DIR "/usr/local")
set(THIRD_PARTY_HEADER "${THIRD_PARTY_DIR}/include")
set(THIRD_PARTY_LIB "${THIRD_PARTY_DIR}/lib")

include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/gtest/include"
  "${THIRD_PARTY_HEADER}"
)

link_directories(
  "${PROJECT_BINARY_DIR}"
  "${THIRD_PARTY_LIB}"
)

#-------------------------------------------------------------------------------
# Declare packages in F2M project.
#-------------------------------------------------------------------------------
add_subdirectory(gtest)
add_subdirectory(demo)
add_subdirectory(src/base)
add_subdirectory(src/thread)
add_subdirectory(src/data)
add_subdirectory(src/reader)
add_subdirectory(src/loss)
add_subdirectory(src/update)
add_subdirectory(src/validate)
add_subdirectory(src/train)
