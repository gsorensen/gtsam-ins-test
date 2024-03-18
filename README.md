# GTSAM Conan project

Basic template for a GTSAM project where the dependency on GTSAM is handled via
conan.

## Dependencies

- CMake
- Conan 

Dependencies can be installed via Homebrew on macOS and (I assume) your
favourite package manager on Linux distros.

## Building

The scripts are rather simple, just automates the commands needed. Should be run
in this order (all from root directory):
1. install_conan_packages.sh 
2. setup_cmake.sh 
3. build_project.sh 
4. run.sh

1. and 2. should only be run whenever dependencies change.

## Notes

Project structure is based off of the tutorial from Conan's documentation
[here](https://docs.conan.io/2.0/tutorial/consuming_packages/build_simple_cmake_project.html). The build scripts here are for Linux/macOS, and will not work on Windows. 
