#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <oafitting/stochprogram.hpp>
#include <oafitting/linreg.hpp>
#include <math.h>
#include <vector>

namespace pt = boost::property_tree;

Eigen::MatrixXd sp::minus_to_zero(const Eigen::MatrixXd &matrix)
{

    int num_rows = matrix.rows();
    int num_cols = matrix.cols();
    Eigen::MatrixXd newmatrix(num_rows, num_cols);
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            if (matrix(i, j) > 0.0)
            {
                newmatrix(i, j) = matrix(i, j);
            }
            else
            {
                newmatrix(i, j) = 0.0;
            }
        }
    }
    return newmatrix;
}

sp::KnnFitParams sp::read_knn_params(std::string filename)
{
    pt::ptree root;
    pt::read_json(filename, root);
    std::string fitmethod = root.get<std::string>("tuningmethod");
    if (fitmethod == "fixed")
    {
        std::vector<int> possible_values = {root.get<int>("k")};
        return KnnFitParams(0.0, possible_values);
    }
    else if (fitmethod == "validate")
    {
        std::vector<int> possible_values;
        for (pt::ptree::value_type &kvaluenode : root.get_child("possiblevals"))
        {
            possible_values.emplace_back(kvaluenode.second.get_value<int>());
        }
        double validateprop = root.get<double>("validationprop");
        return KnnFitParams(validateprop, possible_values);
    }
    else
    {
        throw std::invalid_argument("Knn Parameter file is not properly formatted");
    }
}

double sp::GRBLossCalculator::get_avgloss(const Eigen::MatrixXd &solutions, const Eigen::MatrixXd &parameters)
{
    int ntest = solutions.rows();
    double total = 0.0;
    for (int i = 0; i < ntest; i++)
    {
        double best_obj = fsolver.optimal_value(parameters.row(i));
        total += oaf::get_loss(solutions.row(i), parameters.row(i), fsolver, rsolver, best_obj);
    }
    return total / (double)ntest;
}

double sp::EuclideanMetric::evaluate(const Eigen::VectorXd &vector1, const Eigen::VectorXd &vector2) const
{
    Eigen::VectorXd difference = vector1 - vector2;
    return sqrt(difference.dot(difference));
}

oaf::FittingResult sp::LeastSqLrFitter::fit(const Eigen::MatrixXd &train_side_info,
                                            const Eigen::MatrixXd &train_param) const
{
    return lr::fitmodel(train_side_info, train_param);
}