#include "gurobi_c++.h"
#include <oafitting/oafit.hpp>
#include <oafitting/oafitio.hpp>
#include <Eigen/Dense>
#include <boost/program_options.hpp>
#include <oafitting/linreg.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/ml.hpp>

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    po::options_description opt_desc("Allowed options");
    double fittol;
    opt_desc.add_options()("sideinfo-file", po::value<std::string>(), "Training side information.")("parameter-file", po::value<std::string>(), "Training parameter.")("sidetest-file", po::value<std::string>(), "Side information test set")("out-file", po::value<std::string>(), "Output file. WILL BE OVERWRITTEN IF IT EXISTS.");

    po::positional_options_description popt_desc;
    popt_desc.add("sideinfo-file", 1).add("parameter-file", 1).add("sidetest-file", 1).add("out-file", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(opt_desc).positional(popt_desc).run(), vm);
    po::notify(vm);

    std::string inputfile;
    if (vm.count("sideinfo-file") && vm.count("parameter-file") && vm.count("sidetest-file") && vm.count("out-file"))
    {
        std::string filename = vm["sideinfo-file"].as<std::string>();
        Eigen::MatrixXd side_matrix = oafio::read_eigen_matrix(filename);

        filename = vm["parameter-file"].as<std::string>();
        Eigen::MatrixXd param_matrix = oafio::read_eigen_matrix(filename);

        filename = vm["sidetest-file"].as<std::string>();
        Eigen::MatrixXd test_matrix = oafio::read_eigen_matrix(filename);

        filename = vm["out-file"].as<std::string>();

    
        oafio::write_eigen_matrix(filename, lr::rf_fitnpredict(side_matrix, param_matrix, test_matrix));
    }
    else
    {
        throw std::invalid_argument("This executable needs to be run with four command line arguments: side information, parameter file, and model file.");
    }

    return 0;
}