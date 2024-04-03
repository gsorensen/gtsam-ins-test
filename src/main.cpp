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

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include <fmt/core.h>
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

/* CONSTANTS*/
// const Opt opt = iSam2;
const Optimiser opt = Optimiser::fixLag;
// const Opt opt = LM;

const bool optimise = true;
// const bool optimise = false;

// const bool print_marginals = true;
const bool print_marginals = false;

const double fixed_lag = 5.0; // fixed smoother lag

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

auto get_ISAM2_params(const Optimiser &opt) -> ISAM2Params
{
    ISAM2Params params;
    switch (opt)
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

auto main(int argc, char *argv[]) -> int
{
    if (argc != 2)
    {
        std::cerr << fmt::format("Usage: {} <file_path>\n", argv[0]);
        return 1;
    }

    /// NOTE: Try except used because GTSAM under the hood may throw an
    /// exception if it receives an invalid argument.
    try
    {
        SimulationData data = parse_data(argv[1]).value();

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
        Values result;
        std::uint64_t correction_count = 0;

        initial_values.insert(X(correction_count), prior_pose);
        initial_values.insert(V(correction_count), prior_velocity);
        initial_values.insert(B(correction_count), prior_imu_bias);

        ISAM2Params isam2_params = get_ISAM2_params(opt);
        ISAM2 isam2 = ISAM2(isam2_params);
        IncrementalFixedLagSmoother fixed_lag_smoother = IncrementalFixedLagSmoother(fixed_lag, isam2_params);

        // FixedLagSmoother smoother;
        // smoother = new IncrementalFixedLagSmoother(lag, isam2_params);
        FixedLagSmoother::KeyTimestampMap smoother_timestamps_maps;
        smoother_timestamps_maps[X(correction_count)] = 0.0;
        smoother_timestamps_maps[V(correction_count)] = 0.0;
        smoother_timestamps_maps[B(correction_count)] = 0.0;

        // Assemble prior noise model and add it to the NonlinearFactorGraph
        noiseModel::Diagonal::shared_ptr prior_noise_model = noiseModel::Diagonal::Sigmas(
            (Vector(6) << deg2rad(0.01), deg2rad(0.01), deg2rad(0.01), 0.1, 0.1, 0.1).finished());
        noiseModel::Diagonal::shared_ptr velocity_noise_model = noiseModel::Isotropic::Sigma(3, 0.01);
        noiseModel::Diagonal::shared_ptr bias_noise_model = noiseModel::Diagonal::Sigmas(
            (Vector(6) << 0.5, 0.5, 0.5, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1)).finished());

        auto *graph = new NonlinearFactorGraph();

        graph->add(PriorFactor<Pose3>(X(correction_count), prior_pose, prior_noise_model));
        graph->add(PriorFactor<Vector3>(V(correction_count), prior_velocity, velocity_noise_model));
        graph->add(PriorFactor<imuBias::ConstantBias>(B(correction_count), prior_imu_bias, bias_noise_model));

        PreintegratedMeasurement imu_preintegrated_ = get_preintegrated_IMU_measurement_ptr(prior_imu_bias, true);
        NavState prev_state{prior_pose, prior_velocity};
        NavState prop_state = prev_state;

        imuBias::ConstantBias prev_bias = prior_imu_bias;

        // Covariance matrices
        Eigen::MatrixXd P_X;
        Eigen::MatrixXd P_V;
        Eigen::MatrixXd P_B;

        Marginals marginals_init{*graph, initial_values};
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
        fmt::print("Number of samples: {}\n", N);
        Matrix15d P0 = Eigen::MatrixXd::Zero(15, 15);
        P0.block<6, 6>(0, 0) = P_X;
        P0.block<3, 3>(6, 6) = P_V;
        P0.block<6, 6>(9, 9) = P_B;
        std::list<Matrix15d> P = {};
        P.push_back(P0);

        // Keep track of the total error over the entire run for a simple performance metric.
        double current_position_error = 0.0;
        double current_orientation_error = 0.0;

        double output_time = 0.0;
        double dt = 0.01;

        Eigen::MatrixXd meas_cov = get_preintegrated_meas_cov(imu_preintegrated_);
        std::cout << "Meas cov\n";
        std::cout << meas_cov << "\n";

        auto start_filtering = std::chrono::system_clock::now();
        for (int i = 1; i < N; i++)
        {
            output_time += dt;
            fmt::print("({}) Starting iteration...\n", i);

            fmt::print("({}) Integrating measurement...\n", i);

            integrate_measurement(imu_preintegrated_, data.f_meas.col(i), data.w_meas.col(i), dt);

            fmt::print("({}) Prediction...\n", i);
            // prop_state = imu_preintegrated_->predict(prev_state, prev_bias);
            //  "Manual predicition"
            Vector3 pos_prev = prev_state.pose().translation();
            Vector3 vel_prev = prev_state.velocity();
            Rot3 rot_prev = prev_state.pose().rotation();

            Vector3 inc_ang = (data.w_meas.col(i) - prev_bias.gyroscope()) * dt;
            Rot3 delta_rot = Rot3::Expmap(inc_ang);
            /*
            double inc_ang_mag = inc_ang.norm();
            Rot3 delta_rot;
            if ( inc_ang_mag > 1e-8)
            {
                delta_rot = Rot3::Rodrigues( inc_ang(0), inc_ang(1), inc_ang(2) );
            }
            else
            {
                Matrix3 S_delta = skewSymmetric( inc_ang(0), inc_ang(1), inc_ang(2) );
                //delta_rot = Rot3::Identity() + S_delta;
            }
            */
            Rot3 rot_new = rot_prev * delta_rot;
            rot_new.normalized();
            // rot_new = Rot3::Quaternion(data.q_nb.col(i)[0], data.q_nb.col(i)[1], data.q_nb.col(i)[2],
            // data.q_nb.col(i)[3]);

            Vector3 acc_new = rot_new * (data.f_meas.col(i) - prev_bias.accelerometer()) +
                              get_preintegrated_params(imu_preintegrated_)->getGravity();
            Vector3 vel_new = vel_prev + acc_new * dt;
            Vector3 pos_new = pos_prev + (vel_new + vel_prev) * dt / 2;

            prop_state = NavState(rot_new, pos_new, vel_new);

            const bool is_update_step = (i + 1) % 10 == 0;
            if (optimise && is_update_step)
            {
                correction_count++;

                // auto *preint_imu_combined = dynamic_cast<PreintegratedImuMeasurements *>(imu_preintegrated_);
                //                auto *preint_imu_combined = dynamic_cast<PreintegratedCombinedMeasurements
                //                *>(imu_preintegrated_);

                // ImuFactor imu_factor = {X(correction_count - 1), V(correction_count - 1), X(correction_count),
                //                         V(correction_count),     B(correction_count - 1), *preint_imu_combined};

                fmt::print("({}) Creating combined IMU factor...\n", i);
                CombinedImuFactor imu_factor = {X(correction_count - 1),
                                                V(correction_count - 1),
                                                X(correction_count),
                                                V(correction_count),
                                                B(correction_count - 1),
                                                B(correction_count),
                                                std::get<PreintegratedCombinedMeasurements>(imu_preintegrated_)};

                fmt::print("({}) Adding combined IMU factor to graph...\n", i);
                graph->add(imu_factor);
                // imuBias::ConstantBias zero_bias{Vector3{0.0, 0.0, 0.0}, Vector3{0.0, 0.0, 0.0}};

                //// NOTE: This should NOT be added when using combined measurements
                // graph->add(BetweenFactor<imuBias::ConstantBias>(B(correction_count - 1), B(correction_count),
                // zero_bias,
                //                                                bias_noise_model));

                fmt::print("({}) Insert prediction into values...\n", i);
                initial_values.insert(X(correction_count), prop_state.pose());
                initial_values.insert(V(correction_count), prop_state.v());
                initial_values.insert(B(correction_count), prev_bias);

                fmt::print("({}) Add GNSS factor for aiding measurement...\n", i);
                noiseModel::Diagonal::shared_ptr correction_noise =
                    noiseModel::Diagonal::Sigmas((Vector(3) << 1.5, 1.5, 3).finished());

                /*
                Eigen::VectorXd pos = data.p_nb_n.col(i);
                Eigen::VectorXd meas = data.z_GNSS.col(i);
                double pos_x = pos.x();
                double pos_y = pos.y();
                double pos_z = pos.z();
                double meas_x = meas.x();
                double meas_y = meas.y();
                double meas_z = meas.z();
                */
                // GPSFactor gps_factor{X(correction_count), data.z_GNSS.col(i), correction_noise};
                GPSFactor gps_factor{X(correction_count), data.p_nb_n.col(i), correction_noise};
                graph->add(gps_factor);

                fmt::print("({}) Optimising...\n", i);
                auto start_optimisation = std::chrono::system_clock::now();
                switch (opt)
                {
                case Optimiser::iSam2: {
                    isam2.update(*graph, initial_values);
                    result = isam2.calculateEstimate();

                    P_X = isam2.marginalCovariance(X(correction_count));
                    P_V = isam2.marginalCovariance(V(correction_count));
                    P_B = isam2.marginalCovariance(B(correction_count));
                    if (print_marginals)
                    {
                        gtsam::print(P_X, fmt::format("{} Pose Covariance:", i));
                        gtsam::print(P_V, fmt::format("{} Velocity Covariance:", i));
                        gtsam::print(P_B, fmt::format("{} Bias Covariance:", i));
                    }

                    graph->resize(0);
                    initial_values.clear();
                    break;
                }
                case Optimiser::fixLag: {
                    smoother_timestamps_maps[X(correction_count)] = output_time;
                    smoother_timestamps_maps[V(correction_count)] = output_time;
                    smoother_timestamps_maps[B(correction_count)] = output_time;

                    fixed_lag_smoother.update(*graph, initial_values, smoother_timestamps_maps);
                    result = fixed_lag_smoother.calculateEstimate();

                    P_X = fixed_lag_smoother.marginalCovariance(X(correction_count));
                    P_V = fixed_lag_smoother.marginalCovariance(V(correction_count));
                    P_B = fixed_lag_smoother.marginalCovariance(B(correction_count));
                    if (print_marginals)
                    {
                        gtsam::print(P_X, fmt::format("{} Pose Covariance:", i));
                        gtsam::print(P_V, fmt::format("{} Velocity Covariance:", i));
                        gtsam::print(P_B, fmt::format("{} Bias Covariance:", i));
                    }

                    // reset the graph
                    graph->resize(0);
                    initial_values.clear();
                    smoother_timestamps_maps.clear();
                    break;
                }
                case Optimiser::LM: {
                    LevenbergMarquardtOptimizer optimizer(*graph, initial_values);
                    result = optimizer.optimize();

                    Marginals marginals{*graph, result};
                    GaussianFactor::shared_ptr results = marginals.marginalFactor(X(correction_count));
                    P_X = marginals.marginalCovariance(X(correction_count));
                    P_V = marginals.marginalCovariance(V(correction_count));
                    P_B = marginals.marginalCovariance(B(correction_count));
                    if (print_marginals)
                    {
                        results->print();
                        gtsam::print(P_X, fmt::format("{} Pose Covariance:", i));
                        gtsam::print(P_V, fmt::format("{} Velocity Covariance:", i));
                        gtsam::print(P_B, fmt::format("{} Bias Covariance:", i));
                    }
                    break;
                }
                default:
                    return -1;
                }
                Matrix15d Pk = Eigen::MatrixXd::Zero(15, 15);
                Pk.block<6, 6>(0, 0) = P_X;
                Pk.block<3, 3>(6, 6) = P_V;
                Pk.block<6, 6>(9, 9) = P_B;
                P.push_back(Pk);
                fmt::print("({}) Overriding preintegration and resetting prev_state...\n", i);
                // Override the beginning of the preintegration for the

                // prev_state  = NavState(result.at<Pose3>(X(correction_count)),
                // result.at<Vector3>(V(correction_count)));
                Pose3 pose_corrected = result.at<Pose3>(X(correction_count));
                Rot3 rot_corrected = pose_corrected.rotation().normalized();
                Point3 pos_corrected = pose_corrected.translation();
                Vector3 vel_corrected = result.at<Vector3>(V(correction_count));
                prev_state = NavState(rot_corrected, pos_corrected, vel_corrected);
                prev_bias = result.at<imuBias::ConstantBias>(B(correction_count));

                // cout << "(" << i << ") Preintegration before reset \n";
                // imu_preintegrated_->print();

                // Reset the preintegration object
                reset_preintegration_bias(imu_preintegrated_, prev_bias);

                // cout << "(" << i << ") Preintegration after reset \n";
                // imu_preintegrated_->print();
                auto end_optimisation = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_opt = end_optimisation - start_optimisation;
                fmt::print("({}) Optmisation time elapsed: {} [s]\n", i, elapsed_opt.count());
            }
            else
            {
                Matrix15d Pk = Eigen::MatrixXd::Zero(15, 15);
                // Pk = imu_preintegrated_->preintMeasCov(); // does not compile.
                //  no member named 'preintMeasCov' in 'gtsam::ManifoldPreintegration'
                P.push_back(Pk);
                prev_state = prop_state;
            }

            // Print out the position, orientation and velocity error for comparison + bias values
            Vector3 gtsam_position = prev_state.pose().translation();
            Vector3 true_position = data.p_nb_n.col(i);
            Vector3 position_error = gtsam_position - true_position;
            current_position_error = position_error.norm();

            Quaternion gtsam_quat = prev_state.pose().rotation().toQuaternion();
            Quaternion true_quat =
                Rot3::Quaternion(data.q_nb.col(i)[0], data.q_nb.col(i)[1], data.q_nb.col(i)[2], data.q_nb.col(i)[3])
                    .toQuaternion();
            // Quaternion quat_error           = gtsam_quat * true_quat.inverse();
            Quaternion quat_error = gtsam_quat.inverse() * true_quat;
            quat_error.normalize();
            Vector3 euler_angle_error{quat_error.x() * 2, quat_error.y() * 2, quat_error.z() * 2};
            current_orientation_error = euler_angle_error.norm();

            Vector3 true_velocity = data.v_ib_i.col(i);
            Vector3 gtsam_velocity = prev_state.velocity();
            Vector3 velocity_error = gtsam_velocity - true_velocity;
            double current_velocity_error = velocity_error.norm();

            fmt::print("({}) Pos err [m]: {} - Att err [deg]: {} - Vel err [m/s]: {}\n", i, current_position_error,
                       current_orientation_error * rad2deg(1), current_velocity_error);
            prev_bias.print(fmt::format("({})      Bias values: ", i));
            imu_bias_true.print(fmt::format("({}) True bias values: ", i));

            Eigen::Vector3d acc_error = (imu_bias_true.accelerometer() - prev_bias.accelerometer()).transpose();
            Eigen::Vector3d gyro_error = (imu_bias_true.gyroscope() - prev_bias.gyroscope()).transpose();
            fmt::print("({}) Acc bias errors [m/s/s]: {} {} {}\n", i, acc_error.x(), acc_error.y(), acc_error.z());
            fmt::print("({}) Gyro bias errors [deg/s]: {} {} {}\n", i, gyro_error.x() * rad2deg(1),
                       gyro_error.y() * rad2deg(1), gyro_error.z() * rad2deg(1));
        }

        fmt::print("Printing marginals\n");
        if (optimise)
        {
            /*
            Marginals marginals{*graph, result};
            GaussianFactor::shared_ptr results = marginals.marginalFactor(X(correction_count));
            results->print();

            auto P_Xe = marginals.marginalCovariance(X(correction_count));
            std::cou << "Pose Covariance:\n" << P_Xe << std::endl;
            auto P_V = marginals.marginalCovariance(V(correction_count));
            std::cout << "Velocity Covariance:\n" << P_V << std::endl;
            auto P_B = marginals.marginalCovariance(B(correction_count));
            std::cout << "Bias Covariance:\n" << P_B << std::endl;
            */
        }
        print_preintegration(imu_preintegrated_);

        auto end_filtering = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_filtering - start_filtering;
        fmt::print("Elapsed time: {} [s] - Time horizon data: {} [s]\n", elapsed_seconds.count(),
                   (data.p_nb_n.cols() + 1) * dt);
    }
    catch (std::invalid_argument &e)
    {
        std::cout << e.what() << '\n';
    }
    catch (std::runtime_error &e)
    {
        std::cout << e.what() << '\n';
    }
    catch (std::bad_optional_access &e)
    {
        std::cout << e.what() << '\n';
    }
    catch (std::bad_variant_access &e)
    {
        std::cout << e.what() << '\n';
    }

    return 0;
}
