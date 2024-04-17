# Installs packages defined in conanfile.txt
conan install . --output-folder=build --build=missing -s build_type=Debug
