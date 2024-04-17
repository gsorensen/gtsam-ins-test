#include "SimulationData.hpp"
#include "utils.hpp"

#include <Eigen/src/Core/Matrix.h>
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

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include <fmt/core.h>
#include <tuple>
#include <variant>

using namespace gtsam;

using symbol_shorthand::B; // bias (ax, ay, az, gx, gy, gz)
using symbol_shorthand::V; // velocity (xdot, ydot, zdot)
using symbol_shorthand::X; // pose (x, y, z, r, p, y)

using Matrix15d = Eigen::Matrix<double, 15, 15>;

using PreintegratedMeasurement = std::variant<PreintegratedImuMeasurements, PreintegratedCombinedMeasurements>;

enum class Optimiser
{
    iSam2,
    fixLag,
    LM
};

void integrate_measurement(PreintegratedMeasurement &measurement, const Eigen::Vector3d &f, const Eigen::Vector3d &w,
                           double dt)
{
    std::visit([f, w, dt](auto &&x) { x.integrateMeasurement(f, w, dt); }, measurement);
}

auto get_preintegrated_params(const PreintegratedMeasurement &measurement)
    -> boost::shared_ptr<gtsam::PreintegrationParams>
{
    return std::visit([](auto &&x) -> boost::shared_ptr<gtsam::PreintegrationParams> { return x.params(); },
                      measurement);
}
auto get_preintegrated_meas_cov(const PreintegratedMeasurement &measurement) -> Eigen::MatrixXd
{
    return std::visit([](auto &&x) -> Eigen::MatrixXd { return x.preintMeasCov(); }, measurement);
}

void reset_preintegration_bias(PreintegratedMeasurement &measurement, const imuBias::ConstantBias &prev_bias)
{
    std::visit([prev_bias](auto &&x) { x.resetIntegrationAndSetBias(prev_bias); }, measurement);
}

void print_preintegration(const PreintegratedMeasurement &measurement)
{
    std::visit([](auto &&x) { x.print(); }, measurement);
}

auto get_ISAM2_params(const Optimiser &optimisation_scheme) -> ISAM2Params
{
    ISAM2Params params;
    switch (optimisation_scheme)
    {
    case Optimiser::iSam2:
        fmt::print("Using ISAM2\n");
        params.relinearizeThreshold = 0.001;
        params.relinearizeSkip = 1;
        params.findUnusedFactorSlots = true;
        // params.setFactorization("QR");
        break;
    case Optimiser::fixLag:
        fmt::print("Using ISAM2 Fixed lag smoother\n");
        params.relinearizeThreshold = 0.001;
        params.relinearizeSkip = 1;
        params.findUnusedFactorSlots = true;
        // params.setFactorization("QR");
        break;
    case Optimiser::LM:
        fmt::print("Using LM\n");
        break;
    default:
        break;
    }

    return params;
}

auto get_preintegrated_IMU_measurement_ptr(const imuBias::ConstantBias &prior_imu_bias, bool use_combined_measurement)
    -> PreintegratedMeasurement
{
    /// NOTE: These mirror the noise sigma parameters in the MATLAB script
    double accel_noise_sigma = 0.0012;
    double gyro_noise_sigma = 4.3633e-5;
    double accel_bias_rw_sigma = 5.2193e-4;
    double gyro_bias_rw_sigma = 3.6697e-5;
    Matrix33 measured_acc_cov = Matrix33::Identity(3, 3) * pow(accel_noise_sigma, 2);
    Matrix33 measured_omega_cov = Matrix33::Identity(3, 3) * pow(gyro_noise_sigma, 2);
    Matrix33 bias_acc_cov = Matrix33::Identity(3, 3) * pow(accel_bias_rw_sigma, 2);
    Matrix33 bias_omega_cov = Matrix33::Identity(3, 3) * pow(gyro_bias_rw_sigma, 2);

    /// TODO: What is this quantity related to in the "normal" INS case, is
    /// this a seperate tuning parameter not present there?
    Matrix33 integration_error_cov =
        Matrix33::Identity(3, 3) * 1e-10; // error committed in integrating position from velocities
    Matrix66 bias_acc_omega_int = Matrix::Identity(6, 6);
    bias_acc_omega_int.block<3, 3>(0, 0) = bias_acc_cov;
    bias_acc_omega_int.block<3, 3>(3, 3) = bias_omega_cov;

    boost::shared_ptr<PreintegratedCombinedMeasurements::Params> p =
        PreintegratedCombinedMeasurements::Params::MakeSharedD(9.81);

    p->accelerometerCovariance =
        measured_acc_cov; ///< continuous-time "Covariance" describing accelerometer bias random walk
    p->integrationCovariance = integration_error_cov; ///<- error committed in integrating position from velocities
    p->gyroscopeCovariance = measured_omega_cov; ///< continuous-time "Covariance" describing gyroscope bias random walk
    p->biasAccCovariance = bias_acc_cov;     ///< continuous-time "Covariance" describing accelerometer bias random walk
    p->biasOmegaCovariance = bias_omega_cov; ///< continuous-time "Covariance" describing gyroscope bias random walk
    p->biasAccOmegaInt = bias_acc_omega_int; ///< covariance of bias used as initial estimate.

    PreintegratedMeasurement measurement;
    if (use_combined_measurement)
    {
        measurement = PreintegratedCombinedMeasurements(p, prior_imu_bias);
    }
    else
    {
        measurement = PreintegratedImuMeasurements(p, prior_imu_bias);
    }

    return measurement;
}

auto initialise_prior_noise_models()
    -> std::tuple<noiseModel::Diagonal::shared_ptr, noiseModel::Diagonal::shared_ptr, noiseModel::Diagonal::shared_ptr>
{
    // Assemble prior noise model
    noiseModel::Diagonal::shared_ptr prior_pose_noise_model = noiseModel::Diagonal::Sigmas(
        (Vector(6) << deg2rad(0.01), deg2rad(0.01), deg2rad(0.01), 0.1, 0.1, 0.1).finished());
    noiseModel::Diagonal::shared_ptr prior_velocity_noise_model = noiseModel::Isotropic::Sigma(3, 0.01);
    noiseModel::Diagonal::shared_ptr prior_bias_noise_model =
        noiseModel::Diagonal::Sigmas((Vector(6) << 0.5, 0.5, 0.5, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1)).finished());

    return {prior_pose_noise_model, prior_velocity_noise_model, prior_bias_noise_model};
}

class FactorGraphOptimisation
{
  public:
    explicit FactorGraphOptimisation(const SimulationData &data, const Optimiser &optimisation_scheme,
                                     const double &fixed_lag, bool should_print_marginals);

    auto dt() const -> double
    {
        return m_dt;
    }
    auto N() const -> size_t
    {
        return m_N;
    }
    auto predict_state(const int &idx) -> void;
    auto propagate_state_without_optimising(const int &idx) -> void;
    auto increment_correction_count() -> void
    {
        m_correction_count++;
    }
    auto add_imu_factor_to_graph(const int &idx) -> void;
    auto insert_predicted_state_into_values(const int &idx) -> void;
    auto add_gnss_factor_to_graph(const int &idx) -> void;
    auto optimise(const int &idx, const Optimiser &optimisation_scheme) -> void;
    auto print_errors(const int &idx) -> void;
    auto print_current_preintegration_measurement(const int &idx) const -> void;

  private:
    size_t m_N;
    bool m_print_marginals;
    Values m_initial_values;
    Values m_result{};
    std::uint64_t m_correction_count = 0;
    double m_current_position_error = 0.0;
    double m_current_velocity_error = 0.0;
    double m_current_orientation_error = 0.0;
    double m_output_time = 0.0;
    double m_dt = 0.01;
    std::list<Matrix15d> m_P = {};
    Matrix15d m_P_k = Eigen::MatrixXd::Zero(15, 15);
    Matrix15d m_P_corr = Eigen::MatrixXd::Zero(15, 15);
    NavState m_prev_state;
    NavState m_prop_state;
    imuBias::ConstantBias m_prev_bias;
    PreintegratedMeasurement m_imu_preintegrated;
    NonlinearFactorGraph m_graph;
    ISAM2 m_isam2;
    IncrementalFixedLagSmoother m_fixed_lag_smoother;
    FixedLagSmoother::KeyTimestampMap m_smoother_timestamps_maps;
    Eigen::MatrixXd m_P_X;
    Eigen::MatrixXd m_P_V;
    Eigen::MatrixXd m_P_B;
    Vector3 m_acc_bias_true;
    Vector3 m_gyro_bias_true;
    imuBias::ConstantBias m_imu_bias_true;
    SimulationData m_data;
    /// TODO: Store prev state etc
};

FactorGraphOptimisation::FactorGraphOptimisation(const SimulationData &data, const Optimiser &optimisation_scheme,
                                                 const double &fixed_lag, bool should_print_marginals)
    : m_data{data}, m_print_marginals(should_print_marginals)
{
    // Set the prior based on data
    Point3 prior_point{data.p_nb_n.col(0)};
    Rot3 prior_rotation =
        Rot3::Quaternion(data.q_nb.col(0)[0], data.q_nb.col(0)[1], data.q_nb.col(0)[2], data.q_nb.col(0)[3]);
    Pose3 prior_pose{prior_rotation, prior_point};
    Vector3 prior_velocity{data.v_ib_i.col(0)};
    imuBias::ConstantBias prior_imu_bias;

    // Constant state
    Vector3 acc_bias_true(-0.276839, -0.244186, 0.337360);
    Vector3 gyro_bias_true(-0.0028, 0.0021, -0.0032);
    imuBias::ConstantBias imu_bias_true(acc_bias_true, gyro_bias_true);

    Values initial_values;
    std::uint64_t correction_count = 0;

    initial_values.insert(X(correction_count), prior_pose);
    initial_values.insert(V(correction_count), prior_velocity);
    initial_values.insert(B(correction_count), prior_imu_bias);

    ISAM2Params isam2_params = get_ISAM2_params(optimisation_scheme);
    ISAM2 isam2 = ISAM2(isam2_params);
    IncrementalFixedLagSmoother fixed_lag_smoother = IncrementalFixedLagSmoother(fixed_lag, isam2_params);

    // FixedLagSmoother smoother;
    // smoother = new IncrementalFixedLagSmoother(lag, isam2_params);
    FixedLagSmoother::KeyTimestampMap smoother_timestamps_maps;
    smoother_timestamps_maps[X(correction_count)] = 0.0;
    smoother_timestamps_maps[V(correction_count)] = 0.0;
    smoother_timestamps_maps[B(correction_count)] = 0.0;

    auto graph = NonlinearFactorGraph();
    const auto &[pose_noise, velocity_noise, bias_noise] = initialise_prior_noise_models();
    graph.add(PriorFactor<Pose3>(X(correction_count), prior_pose, pose_noise));
    graph.add(PriorFactor<Vector3>(V(correction_count), prior_velocity, velocity_noise));
    graph.add(PriorFactor<imuBias::ConstantBias>(B(correction_count), prior_imu_bias, bias_noise));

    PreintegratedMeasurement imu_preintegrated_ = get_preintegrated_IMU_measurement_ptr(prior_imu_bias, true);
    NavState prev_state{prior_pose, prior_velocity};

    // Covariance matrices
    Eigen::MatrixXd P_X;
    Eigen::MatrixXd P_V;
    Eigen::MatrixXd P_B;

    Marginals marginals_init{graph, initial_values};
    P_X = marginals_init.marginalCovariance(X(correction_count));
    P_V = marginals_init.marginalCovariance(V(correction_count));
    P_B = marginals_init.marginalCovariance(B(correction_count));

    fmt::print("Initial pose covariance\n");
    std::cout << P_X << "\n";
    fmt::print("Initial velocity covariance\n");
    std::cout << P_V << "\n";
    fmt::print("Initial bias covariance\n");
    std::cout << P_B << "\n";

    static const size_t N = data.p_nb_n.cols();
    Matrix15d P0 = Eigen::MatrixXd::Zero(15, 15);
    P0.block<6, 6>(0, 0) = P_X;
    P0.block<3, 3>(6, 6) = P_V;
    P0.block<6, 6>(9, 9) = P_B;
    std::list<Matrix15d> P = {};
    P.push_back(P0);
    Matrix15d P_k = Eigen::MatrixXd::Zero(15, 15);
    Matrix15d P_corr = Eigen::MatrixXd::Zero(15, 15);

    m_initial_values = initial_values;
    m_P = P;
    m_P_corr = P_corr;
    m_P_k = P_k;
    m_prev_state = prev_state;
    m_prop_state = prev_state;
    m_imu_preintegrated = imu_preintegrated_;
    m_graph = graph;
    m_fixed_lag_smoother = fixed_lag_smoother;
    m_isam2 = isam2;
    m_smoother_timestamps_maps = smoother_timestamps_maps;
    m_prev_bias = prior_imu_bias;
    m_N = N;
    m_P_X = P_X;
    m_P_V = P_V;
    m_P_B = P_B;
    m_acc_bias_true = acc_bias_true;
    m_gyro_bias_true = gyro_bias_true;
    m_imu_bias_true = imu_bias_true;
}
auto FactorGraphOptimisation::predict_state(const int &idx) -> void
{
    m_output_time += m_dt;
    fmt::print("({}) Starting iteration...\n", idx);

    fmt::print("({}) Integrating measurement...\n", idx);

    integrate_measurement(m_imu_preintegrated, m_data.f_meas.col(idx), m_data.w_meas.col(idx), m_dt);

    fmt::print("({}) Prediction...\n", idx);
    Vector3 pos_prev = m_prev_state.pose().translation();
    Vector3 vel_prev = m_prev_state.velocity();
    Rot3 rot_prev = m_prev_state.pose().rotation();

    Vector3 inc_ang = (m_data.w_meas.col(idx) - m_prev_bias.gyroscope()) * m_dt;
    Rot3 delta_rot = Rot3::Expmap(inc_ang);
    Rot3 rot_new = rot_prev * delta_rot;
    rot_new.normalized();

    Vector3 acc_new = rot_new * (m_data.f_meas.col(idx) - m_prev_bias.accelerometer()) +
                      get_preintegrated_params(m_imu_preintegrated)->getGravity();
    Vector3 vel_new = vel_prev + acc_new * m_dt;
    Vector3 pos_new = pos_prev + (vel_new + vel_prev) * m_dt / 2;

    m_prop_state = NavState(rot_new, pos_new, vel_new);
}

auto FactorGraphOptimisation::add_imu_factor_to_graph(const int &idx) -> void
{
    fmt::print("({}) Creating combined IMU factor...\n", idx);
    CombinedImuFactor imu_factor = {X(m_correction_count - 1),
                                    V(m_correction_count - 1),
                                    X(m_correction_count),
                                    V(m_correction_count),
                                    B(m_correction_count - 1),
                                    B(m_correction_count),
                                    std::get<PreintegratedCombinedMeasurements>(m_imu_preintegrated)};

    fmt::print("({}) Adding combined IMU factor to graph...\n", idx);
    m_graph.add(imu_factor);
}

auto FactorGraphOptimisation::insert_predicted_state_into_values(const int &idx) -> void
{
    fmt::print("({}) Insert prediction into values...\n", idx);
    m_initial_values.insert(X(m_correction_count), m_prop_state.pose());
    m_initial_values.insert(V(m_correction_count), m_prop_state.v());
    m_initial_values.insert(B(m_correction_count), m_prev_bias);
}

auto FactorGraphOptimisation::add_gnss_factor_to_graph(const int &idx) -> void
{
    fmt::print("({}) Add GNSS factor for aiding measurement...\n", idx);
    noiseModel::Diagonal::shared_ptr correction_noise =
        noiseModel::Diagonal::Sigmas((Vector(3) << 1.5, 1.5, 3).finished());

    GPSFactor gps_factor{X(m_correction_count), m_data.z_GNSS.col(idx), correction_noise};
    m_graph.add(gps_factor);
}

auto FactorGraphOptimisation::propagate_state_without_optimising(const int &idx) -> void
{
    std::ignore = idx;
    m_P_k = m_P_corr + get_preintegrated_meas_cov(m_imu_preintegrated);
    m_P.push_back(m_P_k);
    m_prev_state = m_prop_state;
}

auto FactorGraphOptimisation::optimise(const int &idx, const Optimiser &optimisation_scheme) -> void
{
    fmt::print("({}) Optimising...\n", idx);
    auto start_optimisation = std::chrono::system_clock::now();
    switch (optimisation_scheme)
    {
    case Optimiser::iSam2: {
        m_isam2.update(m_graph, m_initial_values);
        m_result = m_isam2.calculateEstimate();

        m_P_X = m_isam2.marginalCovariance(X(m_correction_count));
        m_P_V = m_isam2.marginalCovariance(V(m_correction_count));
        m_P_B = m_isam2.marginalCovariance(B(m_correction_count));

        /// TODO: Fix this
        if (m_print_marginals)
        {
            gtsam::print(m_P_X, fmt::format("({}) Pose Covariance:", idx));
            gtsam::print(m_P_V, fmt::format("({}) Velocity Covariance:", idx));
            gtsam::print(m_P_B, fmt::format("({}) Bias Covariance:", idx));
        }

        m_graph.resize(0);
        m_initial_values.clear();
        break;
    }
    case Optimiser::fixLag: {
        m_smoother_timestamps_maps[X(m_correction_count)] = m_output_time;
        m_smoother_timestamps_maps[V(m_correction_count)] = m_output_time;
        m_smoother_timestamps_maps[B(m_correction_count)] = m_output_time;

        m_fixed_lag_smoother.update(m_graph, m_initial_values, m_smoother_timestamps_maps);
        m_result = m_fixed_lag_smoother.calculateEstimate();

        m_P_X = m_fixed_lag_smoother.marginalCovariance(X(m_correction_count));
        m_P_V = m_fixed_lag_smoother.marginalCovariance(V(m_correction_count));
        m_P_B = m_fixed_lag_smoother.marginalCovariance(B(m_correction_count));
        if (m_print_marginals)
        {
            gtsam::print(m_P_X, fmt::format("({}) Pose Covariance:", idx));
            gtsam::print(m_P_V, fmt::format("({}) Velocity Covariance:", idx));
            gtsam::print(m_P_B, fmt::format("({}) Bias Covariance:", idx));
        }

        // reset the graph
        m_graph.resize(0);
        m_initial_values.clear();
        m_smoother_timestamps_maps.clear();
        break;
    }
    case Optimiser::LM: {
        LevenbergMarquardtOptimizer optimizer(m_graph, m_initial_values);
        m_result = optimizer.optimize();

        Marginals marginals{m_graph, m_result};
        GaussianFactor::shared_ptr results = marginals.marginalFactor(X(m_correction_count));
        m_P_X = marginals.marginalCovariance(X(m_correction_count));
        m_P_V = marginals.marginalCovariance(V(m_correction_count));
        m_P_B = marginals.marginalCovariance(B(m_correction_count));
        if (m_print_marginals)
        {
            results->print();
            gtsam::print(m_P_X, fmt::format("({}) Pose Covariance:", idx));
            gtsam::print(m_P_V, fmt::format("({}) Velocity Covariance:", idx));
            gtsam::print(m_P_B, fmt::format("({}) Bias Covariance:", idx));
        }
        break;
    }
    default:
        fmt::print("If this appears, either a new unhandled scheme has been added or something is very wrong\n");
    }

    m_P_k = Eigen::MatrixXd::Zero(15, 15);
    m_P_k.block<6, 6>(0, 0) = m_P_X;
    m_P_k.block<3, 3>(6, 6) = m_P_V;
    m_P_k.block<6, 6>(9, 9) = m_P_B;
    m_P_corr = m_P_k;
    m_P.push_back(m_P_k);
    fmt::print("({}) Overriding preintegration and resetting prev_state...\n", idx);
    // Override the beginning of the preintegration for the

    // prev_state  = NavState(result.at<Pose3>(X(correction_count)),
    // result.at<Vector3>(V(correction_count)));
    auto pose_corrected = m_result.at<Pose3>(X(m_correction_count));
    Rot3 rot_corrected = pose_corrected.rotation().normalized();
    const Point3 &pos_corrected = pose_corrected.translation();
    auto vel_corrected = m_result.at<Vector3>(V(m_correction_count));
    m_prev_state = NavState(rot_corrected, pos_corrected, vel_corrected);
    m_prev_bias = m_result.at<imuBias::ConstantBias>(B(m_correction_count));

    // cout << "(" << i << ") Preintegration before reset \n";
    // imu_preintegrated_->print();

    // Reset the preintegration object
    reset_preintegration_bias(m_imu_preintegrated, m_prev_bias);

    // cout << "(" << i << ") Preintegration after reset \n";
    // imu_preintegrated_->print();
    auto end_optimisation = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_opt = end_optimisation - start_optimisation;
    fmt::print("({}) Optmisation time elapsed: {} [s]\n", idx, elapsed_opt.count());
}

auto FactorGraphOptimisation::print_errors(const int &idx) -> void
{
    // Print out the position, orientation and velocity error for comparison + bias values
    Vector3 gtsam_position = m_prev_state.pose().translation();
    Vector3 true_position = m_data.p_nb_n.col(idx);
    Vector3 position_error = gtsam_position - true_position;
    m_current_position_error = position_error.norm();

    Quaternion gtsam_quat = m_prev_state.pose().rotation().toQuaternion();
    Quaternion true_quat = Rot3::Quaternion(m_data.q_nb.col(idx)[0], m_data.q_nb.col(idx)[1], m_data.q_nb.col(idx)[2],
                                            m_data.q_nb.col(idx)[3])
                               .toQuaternion();
    // Quaternion quat_error           = gtsam_quat * true_quat.inverse();
    Quaternion quat_error = gtsam_quat.inverse() * true_quat;
    quat_error.normalize();
    Vector3 euler_angle_error{quat_error.x() * 2, quat_error.y() * 2, quat_error.z() * 2};
    m_current_orientation_error = euler_angle_error.norm();

    Vector3 true_velocity = m_data.v_ib_i.col(idx);
    Vector3 gtsam_velocity = m_prev_state.velocity();
    Vector3 velocity_error = gtsam_velocity - true_velocity;
    m_current_velocity_error = velocity_error.norm();

    fmt::print("({}) Pos err [m]: {} - Att err [deg]: {} - Vel err [m/s]: {}\n", idx, m_current_position_error,
               m_current_orientation_error * rad2deg(1), m_current_velocity_error);
    m_prev_bias.print(fmt::format("({})      Bias values: ", idx));
    m_imu_bias_true.print(fmt::format("({}) True bias values: ", idx));

    Eigen::Vector3d acc_error = (m_imu_bias_true.accelerometer() - m_prev_bias.accelerometer()).transpose();
    Eigen::Vector3d gyro_error = (m_imu_bias_true.gyroscope() - m_prev_bias.gyroscope()).transpose();
    fmt::print("({}) Acc bias errors [m/s/s]: {} {} {}\n", idx, acc_error.x(), acc_error.y(), acc_error.z());
    fmt::print("({}) Gyro bias errors [deg/s]: {} {} {}\n", idx, gyro_error.x() * rad2deg(1),
               gyro_error.y() * rad2deg(1), gyro_error.z() * rad2deg(1));
}

auto FactorGraphOptimisation::print_current_preintegration_measurement(const int &idx) const -> void
{
    fmt::print("({}) ", idx);
    std::visit([](auto &&x) { x.print(); }, m_imu_preintegrated);
}

/// TODO: Consider whether necessary
// auto FactorGraphOptimisation::display_marginals(const int& idx,) const -> void
//{
//     fmt::print("Printing marginals\n");
//     /// TODO: Move to initialsiation
//     if (optimise)
//     {
//
//         Marginals marginals{graph, result};
//         GaussianFactor::shared_ptr results = marginals.marginalFactor(X(correction_count));
//         results->print();
//
//         auto P_Xe = marginals.marginalCovariance(X(correction_count));
//
//         std::cou << "Pose Covariance:\n" << P_Xe << std::endl;
//         auto P_V = marginals.marginalCovariance(V(correction_count));
//         std::cout << "Velocity Covariance:\n" << P_V << std::endl;
//         auto P_B = marginals.marginalCovariance(B(correction_count));
//         std::cout << "Bias Covariance:\n" << P_B << std::endl;
//     }
// }

auto main(int argc, char *argv[]) -> int
{
    if (argc != 2)
    {
        std::cerr << fmt::format("Usage: {} <file_path>\n", argv[0]);
        return 1;
    }

    SimulationData data{};
    auto parse_result = parse_data(argv[1]);
    if (parse_result)
    {
        data = parse_result.value();
    }
    else
    {
        return -1;
    }

    /* CONSTANTS*/
    const Optimiser optimisation_scheme = Optimiser::fixLag;
    const bool optimise = true;
    const bool print_marginals = false;
    const double fixed_lag = 5.0; // fixed smoother lag

    /// Try except used because GTSAM under the hood may throw an exception
    try
    {
        FactorGraphOptimisation fgo{data, optimisation_scheme, fixed_lag, print_marginals};

        const auto filtering_start_time = std::chrono::system_clock::now();
        const auto N = static_cast<int>(fgo.N());

        for (int idx = 1; idx < N; idx++)
        {
            fgo.predict_state(idx);

            bool update_step = optimise && (idx + 1) % 10 == 0;
            if (update_step)
            {
                fgo.increment_correction_count();
                fgo.add_imu_factor_to_graph(idx);
                fgo.insert_predicted_state_into_values(idx);
                fgo.add_gnss_factor_to_graph(idx);
                fgo.optimise(idx, optimisation_scheme);
            }
            else
            {
                fgo.propagate_state_without_optimising(idx);
            }

            fgo.print_errors(idx);
        }

        fgo.print_current_preintegration_measurement(N - 1);

        auto filtering_end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = filtering_end_time - filtering_start_time;
        fmt::print("Elapsed time: {} [s] - Time horizon data: {} [s]\n", elapsed_seconds.count(),
                   (static_cast<double>(N) + 1) * fgo.dt());
    }
    catch (std::invalid_argument &e)
    {
        std::cout << e.what() << '\n';
    }
    catch (std::runtime_error &e)
    {
        std::cout << e.what() << '\n';
    }
    catch (std::bad_variant_access &e)
    {
        std::cout << e.what() << '\n';
    }

    return 0;
}
