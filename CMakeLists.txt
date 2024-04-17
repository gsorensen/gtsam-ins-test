cmake_minimum_required(VERSION 3.15)
project(gtsam-conan-project CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(GTSAM REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)

message(${GTSAM_INCLUDE_DIR})
message(${Eigen3_INCLUDE_DIR})
include_directories(${GTSAM_INCLUDE_DIR})
include_directories(${Eigen3_INCLUDE_DIR})
set(SOURCE_FILES 
    src/SimulationData.cpp
    src/FactorGraphOptimisation.cpp
    src/utils.cpp
    src/main.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} gtsam::gtsam)
target_link_libraries(${PROJECT_NAME} fmt::fmt)
