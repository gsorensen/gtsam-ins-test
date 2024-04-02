#include "utils.hpp"
#include <iostream>

void print_vector(const Eigen::VectorXd &vec, const std::string &name)
{
    std::cout << name << " | X: " << vec.x() << " Y: " << vec.y() << " Z: " << vec.z() << "\n";
}

auto rad2deg(double rad) -> double
{
    return rad * 180 / M_PI;
}

auto deg2rad(double deg) -> double
{
    return deg / 180 * M_PI;
}
