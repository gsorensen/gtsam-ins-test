# Sets up the cmake project.
# To build, simply run build.sh
#
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1
cd ..
ln -s build/compile_commands.json
