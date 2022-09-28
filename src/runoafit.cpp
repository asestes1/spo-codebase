#include "gurobi_c++.h"
#include <oafitting/oafit.hpp>
#include <oafitting/transport.hpp>
#include <oafitting/oafitio.hpp>
#include <Eigen/Dense>
#include <boost/program_options.hpp>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    po::options_description opt_desc("Allowed options");
    double fittol;
    opt_desc.add_options()
    ("sideinfo-file", po::value<std::string>(), "Training side information.")
    ("parameter-file", po::value<std::string>(), "Training parameter.")
    ("optmodel-file", po::value<std::string>(), "File describing optimization problem.")
    ("guess", po::value<std::string>(), "File describing guess for coefficients.")
    ("coeff-out", po::value<std::string>(), "File to write coefficient file. WILL OVERWRITE IF ALREADY EXISTS")
    ("fittol", po::value<double>(&fittol)->default_value(0.05), "Tolerance in fitting. ");

    po::positional_options_description popt_desc;
    popt_desc.add("sideinfo-file", 1).add("parameter-file", 1).add("optmodel-file", 1).add("guess", 1).add("coeff-out", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(opt_desc).positional(popt_desc).run(), vm);
    po::notify(vm);

    std::string inputfile;
    if (vm.count("sideinfo-file") && vm.count("optmodel-file") && vm.count("parameter-file") && vm.count("coeff-out") && vm.count("guess"))
    {
        std::string filename = vm["sideinfo-file"].as<std::string>();
        Eigen::MatrixXd side_matrix = oafio::read_eigen_matrix(filename);

        filename = vm["optmodel-file"].as<std::string>();
        oaf::ModelInitializer *mymodel = oafio::read_model_initializer(filename);

        filename = vm["parameter-file"].as<std::string>();
        Eigen::MatrixXd param_matrix = oafio::read_eigen_matrix(filename);

        filename = vm["guess"].as<std::string>();
        oaf::FittingResult guess = oafio::read_linear_model(filename);

        filename = vm["coeff-out"].as<std::string>();

        GRBEnv myenv{false};
        Eigen::VectorXd initial_pred = mymodel->default_pred();

        oaf::GRBFirstSolver mysolver{myenv, *mymodel, initial_pred};

        oaf::FirstStageResult fs_result = mysolver.get_lexmin(initial_pred);

        oaf::GRBRecSolver rec_solver{myenv, *mymodel, fs_result.lexmin_sol, initial_pred};

        oaf::FittingResult coeff = oaf::oafit(myenv, side_matrix, param_matrix, guess.coeff, guess.intercept,
                                              mysolver, rec_solver, fittol);

        oafio::write_linear_model(filename, coeff);

        //delete mymodel;
    }
    else
    {
        throw std::invalid_argument("This executable needs to be run with four command line arguments: side information, parameter file, and model file.");
    }

    return 0;
}