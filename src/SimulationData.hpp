#include <Eigen/Core>
#include <cstdint>
#include <optional>

struct SimulationData
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
    Eigen::MatrixXd p_hat;
    Eigen::MatrixXd v_hat;
    Eigen::MatrixXd q_hat;
    Eigen::MatrixXd b_acc_hat;
    Eigen::MatrixXd b_ars_hat;
    // NOTE: This is a bit wasteful, but the data is there every time step
    Eigen::VectorXd num_locators;
    Eigen::MatrixXd z_PARS;
};

// Helper function. Takes an absolute path to a CSV file of the correect format, and returns an
// Eigen Matrix containing the parsed data if the file is found
[[nodiscard]] auto read_from_CSV(const std::string &filename) -> std::optional<Eigen::MatrixXd>;

// Takes an absolute path to a CSV file and returns a SimulationData object if
// the file exists.
// Will currently crash if the format of the file is not the expected CSV
[[nodiscard]] auto parse_data(const std::string &filename, const std::optional<std::uint64_t> &num_rows = std::nullopt)
    -> std::optional<SimulationData>;
