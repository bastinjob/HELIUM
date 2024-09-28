#minimum required version of cmake
cmake_minimum_required(VERSION 3.10)

#project name
project(HELIUM VERSION 1.0 LANGUAGES CXX)

#c++ version requirements
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#include directory
include_directories(include)

#add source files
set(SOURCES 
    src/main.cpp
    src/lexer.cpp 
    src/parser.cpp 
    src/ast.cpp 
    src/compiler.cpp 
)

#create and executable
add_executable(helium_compiler ${SOURCES})

add_executable(lexer_test tests/lexer_test.cpp src/lexer.cpp src/token.cpp)
