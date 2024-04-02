cmake_minimum_required(VERSION 3.15)
project(gtsam-conan-project CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(GTSAM REQUIRED)

message(${GTSAM_INCLUDE_DIR})
include_directories(${GTSAM_INCLUDE_DIR})
set(SOURCE_FILES 
    src/SimulationData.cpp
    src/utils.cpp
    src/main.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} gtsam::gtsam)
