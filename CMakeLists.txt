cmake_minimum_required(VERSION 3.5)
project(svmlabel)

set(CMAKE_CXX_STANDARD 11)
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB ALL src/*.cpp)
file(GLOB HEAD include/*.h)
list(REMOVE_ITEM ALL "${CMAKE_CURRENT_LIST_DIR}/src/svm_struct_main.cpp")
list(REMOVE_ITEM ALL "${CMAKE_CURRENT_LIST_DIR}/src/svm_struct_classify.cpp")
add_library(svmlabel_lib SHARED ${ALL} ${HEAD})

add_executable(svm-detect-train src/svm_struct_main.cpp)
add_executable(svm-detect-predict src/svm_struct_classify.cpp)

target_link_libraries(svm-detect-train svmlabel_lib)
target_link_libraries(svm-detect-predict svmlabel_lib)
