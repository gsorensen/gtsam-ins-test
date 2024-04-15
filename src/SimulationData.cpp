#include "SimulationData.hpp"
#include <fstream>
#include <iostream>

[[nodiscard]] auto parse_data(const std::string &filename, const std::optional<std::uint64_t> &num_rows)
    -> std::optional<SimulationData>
{
    const auto data_matrix_result = read_from_CSV(filename);
    if (!data_matrix_result)
    {
        return std::nullopt;
    }
    const Eigen::MatrixXd &data_matrix = data_matrix_result.value();

    SimulationData data;
    std::uint64_t rows = num_rows.has_value() ? num_rows.value() : data_matrix.rows();
    data.types = data_matrix.col(0).transpose();
    data.p_nb_n = data_matrix.block(0, 1, rows, 3).transpose();
    data.v_ib_i = data_matrix.block(0, 4, rows, 3).transpose();
    data.q_nb = data_matrix.block(0, 7, rows, 4).transpose();
    data.f_meas = data_matrix.block(0, 11, rows, 3).transpose();
    data.w_meas = data_matrix.block(0, 14, rows, 3).transpose();
    data.b_acc = data_matrix.block(0, 17, rows, 3).transpose();
    data.b_ars = data_matrix.block(0, 20, rows, 3).transpose();
    data.t = data_matrix.col(23).transpose();
    data.z_GNSS = data_matrix.block(0, 24, rows, 3).transpose();

    if (data_matrix.cols() > 43)
    {
        data.num_locators = data_matrix.col(43).transpose();
        int measurement_size = data.num_locators[0];
        data.z_PARS = data_matrix.block(0, 44, rows, measurement_size);
    }
    // NOTE: Temp check to use generated data before change, simply set the
    // estimates to true states if the data file doesn't contain them
    if (data_matrix.cols() >= 43)
    {
        data.p_hat = data_matrix.block(0, 27, rows, 3).transpose();
        data.v_hat = data_matrix.block(0, 30, rows, 3).transpose();
        data.q_hat = data_matrix.block(0, 33, rows, 4).transpose();
        data.b_acc_hat = data_matrix.block(0, 37, rows, 3).transpose();
        data.b_ars_hat = data_matrix.block(0, 40, rows, 3).transpose();
    }
    else
    {
        data.p_hat = data_matrix.block(0, 1, rows, 3).transpose();
        data.v_hat = data_matrix.block(0, 4, rows, 3).transpose();
        data.q_hat = data_matrix.block(0, 7, rows, 4).transpose();
        data.b_acc_hat = data_matrix.block(0, 17, rows, 3).transpose();
        data.b_ars_hat = data_matrix.block(0, 20, rows, 3).transpose();
    }

    return data;
}

[[nodiscard]] auto read_from_CSV(const std::string &filename) -> std::optional<Eigen::MatrixXd>
{
    std::ifstream data(filename);

    if (!data)
    {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return std::nullopt;
    }
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
