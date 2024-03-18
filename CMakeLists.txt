cmake_minimum_required(VERSION 3.15)
project(gtsam-conan-project CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(GTSAM REQUIRED)

message(${GTSAM_INCLUDE_DIR})
include_directories(${GTSAM_INCLUDE_DIR})
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} gtsam::gtsam)
