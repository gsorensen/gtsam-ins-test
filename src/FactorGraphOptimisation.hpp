#include "SimulationData.hpp"
#include "utils.hpp"

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include <fmt/core.h>

#include <variant>

enum class Optimiser
{
    iSam2,
    fixLag,
    LM
};

using Matrix15d = Eigen::Matrix<double, 15, 15>;

using PreintegratedMeasurement =
    std::variant<gtsam::PreintegratedImuMeasurements, gtsam::PreintegratedCombinedMeasurements>;

class FactorGraphOptimisation
{
  public:
    explicit FactorGraphOptimisation(const SimulationData &data, const Optimiser &optimisation_scheme,
                                     const double &fixed_lag, bool should_print_marginals);
    auto dt() const -> double;
    auto N() const -> size_t;
    auto predict_state(const int &idx) -> void;
    auto propagate_state_without_optimising(const int &idx) -> void;
    auto increment_correction_count() -> void;
    auto add_imu_factor_to_graph(const int &idx) -> void;
    auto insert_predicted_state_into_values(const int &idx) -> void;
    auto add_gnss_factor_to_graph(const int &idx) -> void;
    auto optimise(const int &idx, const Optimiser &optimisation_scheme) -> void;
    auto compute_and_print_errors(const int &idx) -> void;
    auto print_current_preintegration_measurement(const int &idx) const -> void;
    auto export_data_to_csv(const std::string &filename) const -> void;

  private:
    size_t m_N;
    bool m_print_marginals;
    gtsam::Values m_initial_values;
    gtsam::Values m_result{};
    std::uint64_t m_correction_count = 0;
    Eigen::Vector3d m_current_position_error{};
    Eigen::Vector3d m_current_velocity_error{};
    Eigen::Vector3d m_current_orientation_error{};
    double m_current_position_error_norm{};
    double m_current_velocity_error_norm{};
    double m_current_orientation_error_norm{};
    double m_output_time = 0.0;
    double m_dt = 0.01;
    std::vector<Matrix15d> m_P = {};
    Matrix15d m_P_k = Eigen::MatrixXd::Zero(15, 15);
    Matrix15d m_P_corr = Eigen::MatrixXd::Zero(15, 15);
    gtsam::NavState m_prev_state;
    gtsam::NavState m_prop_state;
    gtsam::imuBias::ConstantBias m_prev_bias;
    PreintegratedMeasurement m_imu_preintegrated;
    gtsam::NonlinearFactorGraph m_graph;
    gtsam::ISAM2 m_isam2;
    gtsam::IncrementalFixedLagSmoother m_fixed_lag_smoother;
    gtsam::FixedLagSmoother::KeyTimestampMap m_smoother_timestamps_maps;
    Eigen::MatrixXd m_P_X;
    Eigen::MatrixXd m_P_V;
    Eigen::MatrixXd m_P_B;
    gtsam::Vector3 m_acc_bias_true;
    gtsam::Vector3 m_gyro_bias_true;
    gtsam::imuBias::ConstantBias m_imu_bias_true;
    SimulationData m_data;
    std::vector<Eigen::Vector3d> m_position_error = {};
    std::vector<Eigen::Vector3d> m_velocity_error = {};
    std::vector<Eigen::Vector3d> m_orientation_error = {};
    std::vector<Eigen::Vector3d> m_acc_bias_error = {};
    std::vector<Eigen::Vector3d> m_gyro_bias_error = {};
    /// TODO: Store prev state etc
};

auto initialise_prior_noise_models()
    -> std::tuple<gtsam::noiseModel::Diagonal::shared_ptr, gtsam::noiseModel::Diagonal::shared_ptr,
                  gtsam::noiseModel::Diagonal::shared_ptr>;

auto get_ISAM2_params(const Optimiser &optimisation_scheme) -> gtsam::ISAM2Params;

void integrate_measurement(PreintegratedMeasurement &measurement, const Eigen::Vector3d &f, const Eigen::Vector3d &w,
                           double dt);

auto get_preintegrated_params(const PreintegratedMeasurement &measurement)
    -> boost::shared_ptr<gtsam::PreintegrationParams>;

auto get_preintegrated_meas_cov(const PreintegratedMeasurement &measurement) -> Eigen::MatrixXd;

void reset_preintegration_bias(PreintegratedMeasurement &measurement, const gtsam::imuBias::ConstantBias &prev_bias);

void print_preintegration(const PreintegratedMeasurement &measurement);

auto get_preintegrated_IMU_measurement_ptr(const gtsam::imuBias::ConstantBias &prior_imu_bias,
                                           bool use_combined_measurement) -> PreintegratedMeasurement;
