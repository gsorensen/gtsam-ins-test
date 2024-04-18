#include "BleBeacon.hpp"

#include <gtsam/linear/NoiseModel.h>
#include <utility>

using gtsam::Point3;

BleBeacon::BleBeacon(gtsam::Point3 origin, const double &yaw, const double &sigma_rho, const double &sigma_alpha,
                     const double &sigma_Psi)
    : m_origin{std::move(origin)}, m_yaw{yaw}
{
    m_noise_model =
        gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << sigma_rho, sigma_alpha, sigma_Psi).finished());
}

auto BleBeacon::sigma_rho() const -> double
{
    return m_noise_model->sigma(0);
}

auto BleBeacon::sigma_alpha() const -> double
{
    return m_noise_model->sigma(1);
}

auto BleBeacon::sigma_Psi() const -> double
{
    return m_noise_model->sigma(2);
}

auto BleBeacon::sigmas() const -> gtsam::noiseModel::Diagonal::shared_ptr
{
    return m_noise_model;
}

auto BleBeacon::origin() const -> Point3
{
    return m_origin;
}

auto BleBeacon::yaw() const -> double
{
    return m_yaw;
}

auto BleBeacon::compute_range(const gtsam::Pose3 &pose) const -> double
{
    return 0;
}
auto BleBeacon::compute_azimuth_angle(const gtsam::Pose3 &pose) const -> double
{
    return 0;
}
auto BleBeacon::compute_elevation_angle(const gtsam::Pose3 &pose) const -> double
{
    return 0;
}
