#include "FactorGraphOptimisation.hpp"
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
#include <stdexcept>
#include <string>

#include <fmt/core.h>
#include <variant>

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

            fgo.compute_and_print_errors(idx);
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
