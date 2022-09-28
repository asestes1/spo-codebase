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

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    po::options_description opt_desc("Allowed options");
    double fittol;
    opt_desc.add_options()
      ("sideinfo-file", po::value<std::string>(), "Training side information.")
      ("parameter-file", po::value<std::string>(), "Training parameter.")
      ("coeff-out", po::value<std::string>(), "File to write coefficient file. WILL OVERWRITE IF ALREADY EXISTS");

    po::positional_options_description popt_desc;
    popt_desc.add("sideinfo-file", 1).add("parameter-file", 1).add("coeff-out",1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(opt_desc).positional(popt_desc).run(), vm);
    po::notify(vm);

    std::string inputfile;
    if (vm.count("sideinfo-file") && vm.count("parameter-file") && vm.count("coeff-out"))
    {
        std::string filename = vm["sideinfo-file"].as<std::string>();
        Eigen::MatrixXd side_matrix = oafio::read_eigen_matrix(filename);

        filename = vm["parameter-file"].as<std::string>();
        Eigen::MatrixXd param_matrix = oafio::read_eigen_matrix(filename);

        filename = vm["coeff-out"].as<std::string>();

        std::ofstream myfile(filename, std::ios::out);
        if (!myfile.is_open())
        {
            std::string myerror = "Unable to open ouput file to store coefficients. Filename: ";
            throw std::runtime_error(myerror.append(filename));
        }
        else
        {

            oaf::FittingResult coeff = lr::fitmodel(side_matrix,param_matrix);
            oafio::write_linear_model(filename, coeff);
        }
        myfile.close();
        //delete mymodel;
    }
    else
    {
        throw std::invalid_argument("This executable needs to be run with four command line arguments: side information, parameter file, and model file.");
    }

    return 0;
}