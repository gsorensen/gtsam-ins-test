#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>

class BleBeacon
{
  public:
    explicit BleBeacon(gtsam::Point3 origin, const double &yaw, const double &sigma_rho, const double &sigma_alpha,
                       const double &sigma_Psi);
    [[nodiscard]] auto sigma_rho() const -> double;
    [[nodiscard]] auto sigma_alpha() const -> double;
    [[nodiscard]] auto sigma_Psi() const -> double;
    [[nodiscard]] auto sigmas() const -> gtsam::noiseModel::Diagonal::shared_ptr;
    [[nodiscard]] auto origin() const -> gtsam::Point3;
    [[nodiscard]] auto yaw() const -> double;
    [[nodiscard]] auto compute_range(const gtsam::Pose3 &pose) const -> double;
    [[nodiscard]] auto compute_azimuth_angle(const gtsam::Pose3 &pose) const -> double;
    [[nodiscard]] auto compute_elevation_angle(const gtsam::Pose3 &pose) const -> double;

  private:
    // Origin and yaw angle relative to {n}
    gtsam::Point3 m_origin;
    double m_yaw;

    // Noise model
    gtsam::noiseModel::Diagonal::shared_ptr m_noise_model;
};
