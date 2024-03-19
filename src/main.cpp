#include "gtsam/geometry/Quaternion.h"
#include <Eigen/src/Core/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

using namespace gtsam;

using symbol_shorthand::B; // bias (ax, ay, az, gx, gy, gz)
using symbol_shorthand::V; // velocity (xdot, ydot, zdot)
using symbol_shorthand::X; // pose (x, y, z, r, p, y)

PreintegrationType *imu_preintegrated_;

// Define a struct to hold the submatrices
struct Data
{
    Eigen::VectorXd types;
    Eigen::MatrixXd p_nb_n;
    Eigen::MatrixXd v_ib_i;
    Eigen::MatrixXd q_nb;
    Eigen::MatrixXd f_meas;
    Eigen::MatrixXd w_meas;
    Eigen::MatrixXd b_acc;
    Eigen::MatrixXd b_ars;
    Eigen::VectorXd t;
    Eigen::MatrixXd z_GNSS;
};

Eigen::MatrixXd read_CSV(const std::string &filename)
{
    std::ifstream data(filename);
    std::string line;
    std::vector<double> values;
    int rows = 0;
    while (std::getline(data, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ','))
        {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        values.data(), rows, values.size() / rows);
}

void print_vector(const Eigen::VectorXd &v, const std::string &name)
{
    std::cout << name << " | X: " << v.x() << " Y: " << v.y() << " Z: " << v.z() << "\n";
}

Data extract_data(const Eigen::MatrixXd &m, const std::optional<std::uint64_t> &num_rows = std::nullopt)
{
    Data data;
    std::uint64_t rows = num_rows.has_value() ? num_rows.value() : 500;
    data.types = m.col(0).transpose();
    data.p_nb_n = m.block(0, 1, rows, 3).transpose();
    data.v_ib_i = m.block(0, 4, rows, 3).transpose();
    data.q_nb = m.block(0, 7, rows, 4).transpose();
    data.f_meas = m.block(0, 11, rows, 3).transpose();
    data.w_meas = m.block(0, 14, rows, 3).transpose();
    data.b_acc = m.block(0, 17, rows, 3).transpose();
    data.b_ars = m.block(0, 20, rows, 3).transpose();
    data.t = m.col(23).transpose();
    data.z_GNSS = m.block(0, 24, rows, 3).transpose();
    return data;
}

const bool optimise = true;

/// TODO: FOllow this example using iSAM https://github.com/borglab/gtsam/blob/develop/examples/ImuFactorsExample2.cpp
/// and look at speed

int main()
{
    /// NOTE: Try except used because GTSAM under the hood may throw an
    /// exception if it receives an invalid argument.
    try
    {
        /// NOTE: This is where my data file is, extract_data shows format, path
        /// needs to be updated
        Eigen::MatrixXd m = read_CSV("/Users/ghms/ws/ntnu/ins_simulator/src/test_data.csv");
        Data data = extract_data(m);

        // Set the prior based on data
        Point3 prior_point{data.p_nb_n.col(0)};
        Rot3 prior_rotation =
            Rot3::Quaternion(data.q_nb.col(0)[0], data.q_nb.col(0)[1], data.q_nb.col(0)[2], data.q_nb.col(0)[3]);
        Pose3 prior_pose{prior_rotation, prior_point};
        Vector3 prior_velocity{data.v_ib_i.col(0)};
        imuBias::ConstantBias prior_imu_bias;

        Values initial_values;
        std::uint64_t correction_count = 0;

        initial_values.insert(X(correction_count), prior_pose);
        initial_values.insert(V(correction_count), prior_velocity);
        initial_values.insert(B(correction_count), prior_imu_bias);

        // Assemble prior noise model and add it to the NonlinearFactorGraph
        /// NOTE: Based on definition of pose, assumed that attitude
        /// uncertainty comes before position uncertainity
        noiseModel::Diagonal::shared_ptr prior_noise_model =
            noiseModel::Diagonal::Sigmas((Vector(6) << 0.5, 0.5, 0.5, 1.5, 1.5, 5).finished());
        // noiseModel::Diagonal::Sigmas((Vector(6) << 1.5, 1.5, 5, 0.5, 0.5, 0.5).finished());

        noiseModel::Diagonal::shared_ptr velocity_noise_model = noiseModel::Isotropic::Sigma(3, 0.2);

        /// NOTE: Assumption is that acceleormeter noise is before gyroscope
        noiseModel::Diagonal::shared_ptr bias_noise_model =
            noiseModel::Diagonal::Sigmas((Vector(6) << 0.5, 0.5, 0.5, 0.05, 0.05, 0.05).finished());

        auto *graph = new NonlinearFactorGraph();

        graph->add(PriorFactor<Pose3>(X(correction_count), prior_pose, prior_noise_model));
        graph->add(PriorFactor<Vector3>(V(correction_count), prior_velocity, velocity_noise_model));
        graph->add(PriorFactor<imuBias::ConstantBias>(B(correction_count), prior_imu_bias, bias_noise_model));

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
            Matrix33::Identity(3, 3) * 1e-8; // error committed in integrating position from velocities
        Matrix66 bias_acc_omega_int = Matrix::Identity(6, 6) * 1e-5;

        boost::shared_ptr<PreintegratedCombinedMeasurements::Params> p =
            PreintegratedCombinedMeasurements::Params::MakeSharedD(9.81);

        p->accelerometerCovariance = measured_acc_cov;
        p->integrationCovariance = integration_error_cov;
        p->gyroscopeCovariance = measured_omega_cov;
        p->biasAccCovariance = bias_acc_cov;
        p->biasOmegaCovariance = bias_omega_cov;
        p->biasAccOmegaInt = bias_acc_omega_int;

        // imu_preintegrated_ = new PreintegratedImuMeasurements(p, prior_imu_bias);
        imu_preintegrated_ = new PreintegratedCombinedMeasurements(p, prior_imu_bias);

        NavState prev_state{prior_pose, prior_velocity};
        NavState prop_state = prev_state;

        /// NOTE: Bias in script modelled as Gauss-Markov, this is probably not
        /// correct
        imuBias::ConstantBias prev_bias = prior_imu_bias;

        // Keep track of the total error over the entire run for a simple performance metric.
        double current_position_error = 0.0;
        double current_orientation_error = 0.0;

        double output_time = 0.0;
        double dt = 0.01;
        for (int i = 1; i < data.p_nb_n.cols(); i++)
        {
            correction_count++;

            imu_preintegrated_->integrateMeasurement(data.f_meas.col(i), data.w_meas.col(i), dt);

            // auto *preint_imu_combined = dynamic_cast<PreintegratedImuMeasurements *>(imu_preintegrated_);
            auto *preint_imu_combined = dynamic_cast<PreintegratedCombinedMeasurements *>(imu_preintegrated_);

            // ImuFactor imu_factor = {X(correction_count - 1), V(correction_count - 1), X(correction_count),
            //                         V(correction_count),     B(correction_count - 1), *preint_imu_combined};

            CombinedImuFactor imu_factor = {X(correction_count - 1), V(correction_count - 1), X(correction_count),
                                            V(correction_count),     B(correction_count - 1), B(correction_count),
                                            *preint_imu_combined};
            graph->add(imu_factor);

            // imuBias::ConstantBias zero_bias{Vector3{0.0, 0.0, 0.0}, Vector3{0.0, 0.0, 0.0}};

            //// NOTE: This should NOT be added when using combined measurements
            // graph->add(BetweenFactor<imuBias::ConstantBias>(B(correction_count - 1), B(correction_count), zero_bias,
            //                                                bias_noise_model));

            prop_state = imu_preintegrated_->predict(prev_state, prev_bias);
            initial_values.insert(X(correction_count), prop_state.pose());
            initial_values.insert(V(correction_count), prop_state.v());
            initial_values.insert(B(correction_count), prev_bias);

            if (optimise)
            {
                if ((i + 1) % 10 == 0)
                {
                    std::cout << "Adding aiding measurement...\n";
                    noiseModel::Diagonal::shared_ptr correction_noise =
                        noiseModel::Diagonal::Sigmas((Vector(3) << 1.5, 1.5, 3).finished());
                    GPSFactor gps_factor{X(correction_count), data.z_GNSS.col(i), correction_noise};
                    // GPSFactor gps_factor{X(correction_count), data.z_GNSS.col(i), correction_noise};
                    graph->add(gps_factor);
                }

                LevenbergMarquardtOptimizer optimizer(*graph, initial_values);
                Values result = optimizer.optimize();

                // Override the beginning of the preintegration for the
                prev_state = NavState(result.at<Pose3>(X(correction_count)), result.at<Vector3>(V(correction_count)));
                prev_bias = result.at<imuBias::ConstantBias>(B(correction_count));
            }
            else
            {
                prev_state = prop_state;
            }

            //            prev_state = prop_state;

            // Reset the preintegration object
            imu_preintegrated_->resetIntegrationAndSetBias(prev_bias);

            // Print out the position and orientation error for
            // comparison
            Vector3 gtsam_position = prev_state.pose().translation();
            Vector3 true_position = data.p_nb_n.col(i);
            Vector3 position_error = gtsam_position - true_position;
            current_position_error = position_error.norm();

            Quaternion gtsam_quat = prev_state.pose().rotation().toQuaternion();
            Quaternion true_quat =
                Rot3::Quaternion(data.q_nb.col(i)[0], data.q_nb.col(i)[1], data.q_nb.col(i)[2], data.q_nb.col(i)[3])
                    .toQuaternion();
            Quaternion quat_error = gtsam_quat * true_quat.inverse();
            quat_error.normalize();
            Vector3 euler_angle_error{quat_error.x() * 2, quat_error.y() * 2, quat_error.z() * 2};
            current_orientation_error = euler_angle_error.norm();

            // print_vector(gtsam_position, "Predicted position");
            // print_vector(true_position, "True position");

            std::cout << "Position error:" << current_position_error
                      << " - Attitude error: " << current_orientation_error << "\n";
        }

        //        graph->print();
    }
    catch (std::invalid_argument &e)
    {

        std::cout << e.what() << '\n';
    }

    return 0;
}
