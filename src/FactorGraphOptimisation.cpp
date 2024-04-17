#include "FactorGraphOptimisation.hpp"

using gtsam::symbol_shorthand::B; // bias (ax, ay, az, gx, gy, gz)
using gtsam::symbol_shorthand::V; // velocity (xdot, ydot, zdot)
using gtsam::symbol_shorthand::X; // pose (x, y, z, r, p, y)

auto initialise_prior_noise_models()
    -> std::tuple<gtsam::noiseModel::Diagonal::shared_ptr, gtsam::noiseModel::Diagonal::shared_ptr,
                  gtsam::noiseModel::Diagonal::shared_ptr>
{
    // Assemble prior noise model
    gtsam::noiseModel::Diagonal::shared_ptr prior_pose_noise_model = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << deg2rad(0.01), deg2rad(0.01), deg2rad(0.01), 1.5, 1.5, 5).finished());
    gtsam::noiseModel::Diagonal::shared_ptr prior_velocity_noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 0.01);
    gtsam::noiseModel::Diagonal::shared_ptr prior_bias_noise_model = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.5, 0.5, 0.5, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1)).finished());

    return {prior_pose_noise_model, prior_velocity_noise_model, prior_bias_noise_model};
}
/// TODO Move
auto get_ISAM2_params(const Optimiser &optimisation_scheme) -> gtsam::ISAM2Params
{
    gtsam::ISAM2Params params;
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

void reset_preintegration_bias(PreintegratedMeasurement &measurement, const gtsam::imuBias::ConstantBias &prev_bias)
{
    std::visit([prev_bias](auto &&x) { x.resetIntegrationAndSetBias(prev_bias); }, measurement);
}

void print_preintegration(const PreintegratedMeasurement &measurement)
{
    std::visit([](auto &&x) { x.print(); }, measurement);
}

auto get_preintegrated_IMU_measurement_ptr(const gtsam::imuBias::ConstantBias &prior_imu_bias,
                                           bool use_combined_measurement) -> PreintegratedMeasurement
{
    /// NOTE: These mirror the noise sigma parameters in the MATLAB script
    double accel_noise_sigma = 0.0012;
    double gyro_noise_sigma = 4.3633e-5;
    double accel_bias_rw_sigma = 5.2193e-4;
    double gyro_bias_rw_sigma = 3.6697e-5;
    gtsam::Matrix33 measured_acc_cov = gtsam::Matrix33::Identity(3, 3) * pow(accel_noise_sigma, 2);
    gtsam::Matrix33 measured_omega_cov = gtsam::Matrix33::Identity(3, 3) * pow(gyro_noise_sigma, 2);
    gtsam::Matrix33 bias_acc_cov = gtsam::Matrix33::Identity(3, 3) * pow(accel_bias_rw_sigma, 2);
    gtsam::Matrix33 bias_omega_cov = gtsam::Matrix33::Identity(3, 3) * pow(gyro_bias_rw_sigma, 2);

    /// TODO: What is this quantity related to in the "normal" INS case, is
    /// this a seperate tuning parameter not present there?
    gtsam::Matrix33 integration_error_cov =
        gtsam::Matrix33::Identity(3, 3) * 1e-10; // error committed in integrating position from velocities
    gtsam::Matrix66 bias_acc_omega_int = gtsam::Matrix::Identity(6, 6);
    bias_acc_omega_int.block<3, 3>(0, 0) = bias_acc_cov;
    bias_acc_omega_int.block<3, 3>(3, 3) = bias_omega_cov;

    boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> p =
        gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedD(9.81);

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
        measurement = gtsam::PreintegratedCombinedMeasurements(p, prior_imu_bias);
    }
    else
    {
        measurement = gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);
    }

    return measurement;
}

FactorGraphOptimisation::FactorGraphOptimisation(const SimulationData &data, const Optimiser &optimisation_scheme,
                                                 const double &fixed_lag, bool should_print_marginals)
    : m_data{data}, m_print_marginals(should_print_marginals)
{
    // Set the prior based on data
    gtsam::Point3 prior_point{data.p_nb_n.col(0)};
    gtsam::Rot3 prior_rotation =
        gtsam::Rot3::Quaternion(data.q_nb.col(0)[0], data.q_nb.col(0)[1], data.q_nb.col(0)[2], data.q_nb.col(0)[3]);
    gtsam::Pose3 prior_pose{prior_rotation, prior_point};
    gtsam::Vector3 prior_velocity{data.v_ib_i.col(0)};
    gtsam::imuBias::ConstantBias prior_imu_bias;

    // Constant state
    gtsam::Vector3 acc_bias_true(-0.276839, -0.244186, 0.337360);
    gtsam::Vector3 gyro_bias_true(-0.0028, 0.0021, -0.0032);
    gtsam::imuBias::ConstantBias imu_bias_true(acc_bias_true, gyro_bias_true);

    gtsam::Values initial_values;
    std::uint64_t correction_count = 0;

    initial_values.insert(X(correction_count), prior_pose);
    initial_values.insert(V(correction_count), prior_velocity);
    initial_values.insert(B(correction_count), prior_imu_bias);

    gtsam::ISAM2Params isam2_params = get_ISAM2_params(optimisation_scheme);
    gtsam::ISAM2 isam2 = gtsam::ISAM2(isam2_params);
    gtsam::IncrementalFixedLagSmoother fixed_lag_smoother = gtsam::IncrementalFixedLagSmoother(fixed_lag, isam2_params);

    // FixedLagSmoother smoother;
    // smoother = new IncrementalFixedLagSmoother(lag, isam2_params);
    gtsam::FixedLagSmoother::KeyTimestampMap smoother_timestamps_maps;
    smoother_timestamps_maps[X(correction_count)] = 0.0;
    smoother_timestamps_maps[V(correction_count)] = 0.0;
    smoother_timestamps_maps[B(correction_count)] = 0.0;

    auto graph = gtsam::NonlinearFactorGraph();
    const auto &[pose_noise, velocity_noise, bias_noise] = initialise_prior_noise_models();
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(correction_count), prior_pose, pose_noise));
    graph.add(gtsam::PriorFactor<gtsam::Vector3>(V(correction_count), prior_velocity, velocity_noise));
    graph.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(correction_count), prior_imu_bias, bias_noise));

    PreintegratedMeasurement imu_preintegrated_ = get_preintegrated_IMU_measurement_ptr(prior_imu_bias, true);
    gtsam::NavState prev_state{prior_pose, prior_velocity};

    // Covariance matrices
    Eigen::MatrixXd P_X;
    Eigen::MatrixXd P_V;
    Eigen::MatrixXd P_B;

    gtsam::Marginals marginals_init{graph, initial_values};
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
    std::vector<Matrix15d> P = {};
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

auto FactorGraphOptimisation::dt() const -> double
{
    return m_dt;
}

auto FactorGraphOptimisation::N() const -> size_t
{
    return m_N;
}

auto FactorGraphOptimisation::increment_correction_count() -> void
{
    m_correction_count++;
}

auto FactorGraphOptimisation::predict_state(const int &idx) -> void
{
    m_output_time += m_dt;
    fmt::print("({}) Starting iteration...\n", idx);

    fmt::print("({}) Integrating measurement...\n", idx);

    integrate_measurement(m_imu_preintegrated, m_data.f_meas.col(idx), m_data.w_meas.col(idx), m_dt);

    fmt::print("({}) Prediction...\n", idx);
    gtsam::Vector3 pos_prev = m_prev_state.pose().translation();
    gtsam::Vector3 vel_prev = m_prev_state.velocity();
    gtsam::Rot3 rot_prev = m_prev_state.pose().rotation();

    gtsam::Vector3 inc_ang = (m_data.w_meas.col(idx) - m_prev_bias.gyroscope()) * m_dt;
    gtsam::Rot3 delta_rot = gtsam::Rot3::Expmap(inc_ang);
    gtsam::Rot3 rot_new = rot_prev * delta_rot;
    rot_new.normalized();

    gtsam::Vector3 acc_new = rot_new * (m_data.f_meas.col(idx) - m_prev_bias.accelerometer()) +
                             get_preintegrated_params(m_imu_preintegrated)->getGravity();
    gtsam::Vector3 vel_new = vel_prev + acc_new * m_dt;
    gtsam::Vector3 pos_new = pos_prev + (vel_new + vel_prev) * m_dt / 2;

    m_prop_state = gtsam::NavState(rot_new, pos_new, vel_new);
}

auto FactorGraphOptimisation::add_imu_factor_to_graph(const int &idx) -> void
{
    fmt::print("({}) Creating combined IMU factor...\n", idx);
    gtsam::CombinedImuFactor imu_factor = {X(m_correction_count - 1),
                                           V(m_correction_count - 1),
                                           X(m_correction_count),
                                           V(m_correction_count),
                                           B(m_correction_count - 1),
                                           B(m_correction_count),
                                           std::get<gtsam::PreintegratedCombinedMeasurements>(m_imu_preintegrated)};

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
    gtsam::noiseModel::Diagonal::shared_ptr correction_noise =
        gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 1.5, 1.5, 3).finished());

    gtsam::GPSFactor gps_factor{X(m_correction_count), m_data.z_GNSS.col(idx), correction_noise};
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
        gtsam::LevenbergMarquardtOptimizer optimizer(m_graph, m_initial_values);
        m_result = optimizer.optimize();

        gtsam::Marginals marginals{m_graph, m_result};
        gtsam::GaussianFactor::shared_ptr results = marginals.marginalFactor(X(m_correction_count));
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
    auto pose_corrected = m_result.at<gtsam::Pose3>(X(m_correction_count));
    gtsam::Rot3 rot_corrected = pose_corrected.rotation().normalized();
    const gtsam::Point3 &pos_corrected = pose_corrected.translation();
    auto vel_corrected = m_result.at<gtsam::Vector3>(V(m_correction_count));
    m_prev_state = gtsam::NavState(rot_corrected, pos_corrected, vel_corrected);
    m_prev_bias = m_result.at<gtsam::imuBias::ConstantBias>(B(m_correction_count));

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

auto FactorGraphOptimisation::compute_and_print_errors(const int &idx) -> void
{
    // Print out the position, orientation and velocity error for comparison + bias values
    gtsam::Vector3 gtsam_position = m_prev_state.pose().translation();
    gtsam::Vector3 true_position = m_data.p_nb_n.col(idx);
    gtsam::Vector3 position_error = gtsam_position - true_position;
    m_position_error.push_back(m_current_position_error);
    m_current_position_error = position_error;
    m_current_position_error_norm = position_error.norm();

    gtsam::Quaternion gtsam_quat = m_prev_state.pose().rotation().toQuaternion();
    gtsam::Quaternion true_quat = gtsam::Rot3::Quaternion(m_data.q_nb.col(idx)[0], m_data.q_nb.col(idx)[1],
                                                          m_data.q_nb.col(idx)[2], m_data.q_nb.col(idx)[3])
                                      .toQuaternion();
    // Quaternion quat_error           = gtsam_quat * true_quat.inverse();
    gtsam::Quaternion quat_error = gtsam_quat.inverse() * true_quat;
    quat_error.normalize();
    gtsam::Vector3 euler_angle_error{quat_error.x() * 2, quat_error.y() * 2, quat_error.z() * 2};
    m_orientation_error.push_back(m_current_orientation_error);
    m_current_orientation_error = euler_angle_error;
    m_current_orientation_error_norm = euler_angle_error.norm();

    gtsam::Vector3 true_velocity = m_data.v_ib_i.col(idx);
    gtsam::Vector3 gtsam_velocity = m_prev_state.velocity();
    gtsam::Vector3 velocity_error = gtsam_velocity - true_velocity;
    m_velocity_error.push_back(m_current_velocity_error);
    m_current_velocity_error = velocity_error;
    m_current_velocity_error_norm = velocity_error.norm();

    Eigen::Vector3d acc_error = (m_imu_bias_true.accelerometer() - m_prev_bias.accelerometer()).transpose();
    Eigen::Vector3d gyro_error = (m_imu_bias_true.gyroscope() - m_prev_bias.gyroscope()).transpose();

    /// NOTE: Done a bit differently to the other states
    m_acc_bias_error.push_back(acc_error);
    m_gyro_bias_error.push_back(gyro_error);

    /// TODO: COnsider moving to separate print functions
    fmt::print("({}) Pos err [m]: {} - Att err [deg]: {} - Vel err [m/s]: {}\n", idx, m_current_position_error_norm,
               m_current_orientation_error_norm * rad2deg(1), m_current_velocity_error_norm);
    m_prev_bias.print(fmt::format("({})      Bias values: ", idx));
    m_imu_bias_true.print(fmt::format("({}) True bias values: ", idx));

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

auto FactorGraphOptimisation::export_data_to_csv(const std::string &filename) const -> void
{
    std::ofstream output_file(filename);
    if (!output_file.is_open())
    {
        std::cerr << "Error opening file: " << filename << '\n';
        return;
    }

    // Write any necessary headers or initial data
    output_file << "Column1, Column2, Column3" << '\n';

    double t = 0.0;
    for (int i = 0; i < m_N; i++)
    {
        t += m_dt; // NOTE Placement before/after
        Matrix15d P = m_P[i];
        Eigen::Vector3d pos_err = m_position_error[i];
        Eigen::Vector3d vel_err = m_velocity_error[i];
        Eigen::Vector3d att_err = m_orientation_error[i];
        Eigen::Vector3d acc_err = m_acc_bias_error[i];
        Eigen::Vector3d gyro_err = m_gyro_bias_error[i];
        output_file << fmt::format("{},", t);
        output_file << fmt::format("{},{},{},{},{},{},", att_err.x(), att_err.y(), att_err.z(), P(0, 0), P(1, 1),
                                   P(2, 2));
        output_file << fmt::format("{},{},{},{},{},{},", pos_err.x(), pos_err.y(), pos_err.z(), P(3, 3), P(4, 4),
                                   P(5, 5));
        output_file << fmt::format("{},{},{},{},{},{},", vel_err.x(), vel_err.y(), vel_err.z(), P(6, 6), P(7, 7),
                                   P(8, 8));
        output_file << fmt::format("{},{},{},{},{},{},", acc_err.x(), acc_err.y(), acc_err.z(), P(9, 9), P(10, 10),
                                   P(11, 11));
        output_file << fmt::format("{},{},{},{},{},{},", gyro_err.x(), gyro_err.y(), gyro_err.z(), P(12, 12), P(13, 13),
                                   P(14, 14));
        output_file << "\n";
    }

    // Close the file when done
    output_file.close();
}
