#include "gurobi_c++.h"
#include <oafitting/stochprogram.hpp>
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
/*
 * This runs a trial of several methods and stores results. Eight required command line arguments:
 * - the filename describing the optimization problem (see example_transport.json)
 * - the filename for the training set for the side information (see example_sideinfo.txt)
 * - the filename for the training set for the parameter (see example_param.txt)
 * - the filename for the test set for the side information (see example_sideinfo.txt)
 * - the filename for the test set for the parameter (see example_param.txt)
 * - the filename containing parameters for the knn method (see example_knn_param.json or example_knn_param2.json)
 * - the filename to write information regarding how the methods performed.
 * - the filename to write information regarding how long the methods took.
 */
int main(int argc, char *argv[])
{
    po::options_description opt_desc("Allowed options");
    double fittol;
    opt_desc.add_options()
    ("optmodel-file", po::value<std::string>(), "File describing optimization problem.")
    ("train-sideinfo-file", po::value<std::string>(), "Training side information.")
    ("train-parameter-file", po::value<std::string>(), "Training parameter.")
    ("test-sideinfo-file", po::value<std::string>(), "Test side information.")
    ("test-parameter-file", po::value<std::string>(), "Test parameter.")
    ("knn-parameter-file", po::value<std::string>(), "KNN parameter file.")
    ("value-out-file", po::value<std::string>(), "File to write losses. WILL OVERWRITE IF ALREADY EXISTS")
    ("timing-out-file", po::value<std::string>(), "File to write timing. WILL OVERWRITE IF ALREADY EXISTS")
    ("fittol", po::value<double>(&fittol)->default_value(0.001), "Tolerance in fitting. ");

    po::positional_options_description popt_desc;
    popt_desc.add("optmodel-file", 1).add("train-sideinfo-file", 1).add("train-parameter-file", 1)
    .add("test-sideinfo-file", 1).add("test-parameter-file", 1).add("knn-parameter-file", 1)
    .add("value-out-file", 1).add("timing-out-file", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(opt_desc).positional(popt_desc).run(), vm);
    po::notify(vm);

    std::string inputfile;
    if (vm.count("optmodel-file") && vm.count("train-sideinfo-file") && vm.count("train-parameter-file") &&
        vm.count("test-sideinfo-file") && vm.count("test-parameter-file") && vm.count("knn-parameter-file")
        && vm.count("value-out-file") && vm.count("timing-out-file"))
    {
        GRBEnv myenv{false};

        std::string filename = vm["train-sideinfo-file"].as<std::string>();
        Eigen::MatrixXd train_side = oafio::read_eigen_matrix(filename);

        filename = vm["train-parameter-file"].as<std::string>();
        Eigen::MatrixXd train_param = oafio::read_eigen_matrix(filename);

        filename = vm["test-sideinfo-file"].as<std::string>();
        Eigen::MatrixXd test_side = oafio::read_eigen_matrix(filename);

        filename = vm["test-parameter-file"].as<std::string>();
        Eigen::MatrixXd test_param = oafio::read_eigen_matrix(filename);


        std::string value_filename = vm["value-out-file"].as<std::string>();
        std::string timing_filename = vm["timing-out-file"].as<std::string>();

        double modelbuildtime = 0.0;
        std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        filename = vm["optmodel-file"].as<std::string>();
        oaf::ModelInitializer *mymodel = oafio::read_model_initializer(filename);
        std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
        modelbuildtime = std::chrono::duration_cast<std::chrono::milliseconds>(end- start).count();
        

        std::cout << "Running OA" << std::endl;
        double oaf_setup_time;
        start = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd initial_pred = mymodel->default_pred();
        oaf::GRBFirstSolver mysolver{myenv, *mymodel, initial_pred};
        oaf::FirstStageResult fs_result = mysolver.get_lexmin(initial_pred);
        oaf::GRBRecSolver rec_solver{myenv, *mymodel, fs_result.lexmin_sol, initial_pred};
        end = std::chrono::high_resolution_clock::now();
        oaf::FittingResult guess = lr::fitmodel(train_side, train_param);
        sp::OaLrFitter<oaf::GRBFirstSolver, oaf::GRBRecSolver> oaffitter{mysolver, rec_solver, guess, fittol};

        oaf_setup_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        double oaffittime;
        double oafpredtime;
        double oafsoltime;
        Eigen::MatrixXd oafsols = sp::lr_ftp(train_side, train_param, test_side, oaffitter,
                                             mysolver, oaffittime, oafpredtime, oafsoltime);

        std::cout << "Running LR" << std::endl;
        double lrfittime;
        double lrpredtime;
        double lrsoltime;
        sp::LeastSqLrFitter myfitter{};
        Eigen::MatrixXd lrsols = sp::lr_ftp(train_side, train_param, test_side, myfitter,
                                            mysolver, lrfittime, lrpredtime, lrsoltime);

        std::cout << "Running RF" << std::endl;
        double rffittime;
        double rfpredtime;
        double rfsoltime;
        Eigen::MatrixXd rfsols = sp::rf_ftp(train_side, train_param, test_side,
                                            mysolver, rffittime, rfpredtime, rfsoltime);


    
        std::string knn_filename = vm["knn-parameter-file"].as<std::string>();
        sp::KnnFitParams myparams = sp::read_knn_params(knn_filename);
        double knnfittime_oa;
        sp::GRBLossCalculator mylosscalc{mysolver, rec_solver};
        std::cout << "Tuning Knn OA loss" << std::endl;
        int bestk_oa = find_best_k_oaloss(train_side, train_param, sp::EuclideanMetric(), *mymodel,
                    mylosscalc, myparams.validateprop, myparams.possiblevals, knnfittime_oa);

        double knnsoltime_oa;

        std::cout << "Running Knn OA loss" << std::endl;
        Eigen::MatrixXd knnsols_oa = sp::knn_sp_solution(train_side, train_param, test_side, bestk_oa, *mymodel,
                                                      sp::EuclideanMetric(), knnsoltime_oa);

        
        double knnfittime_mse;
        std::cout << "Tuning MSE Knn" << std::endl;
        int bestk_mse = find_best_k_mse(train_side, train_param, sp::EuclideanMetric(),
                                     myparams.validateprop, myparams.possiblevals, knnfittime_mse);

        double knnsoltime_mse;

        std::cout << "Running MSE Knn" << std::endl;
        Eigen::MatrixXd knnsols_mse = sp::knn_sp_solution(train_side, train_param, test_side, bestk_mse, *mymodel,
                                                      sp::EuclideanMetric(), knnsoltime_mse);

        std::cout <<"Writing results."<<std::endl;
        std::ofstream valuefile(value_filename, std::ios::out);
        valuefile << "OAF_LOSS,"
                  << "LR_LOSS,"
                  << "RF_LOSS,"
                  << "BK_OA_LOSS,"
                  << "BK_MSE_LOSS,"
                  << "BEST_OBJ\n";
        std::cout << "TESTA" << std::endl;
        int ntest = test_side.rows();
        for (int i = 0; i < ntest; i++)
        {
            double best_obj = mysolver.optimal_value(test_param.row(i));
            valuefile << oaf::get_loss(oafsols.row(i), test_param.row(i), mysolver, rec_solver, best_obj) << ",";
            valuefile << oaf::get_loss(lrsols.row(i), test_param.row(i), mysolver, rec_solver, best_obj) << ",";
            valuefile << oaf::get_loss(rfsols.row(i), test_param.row(i), mysolver, rec_solver, best_obj) << ",";
            valuefile << oaf::get_loss(knnsols_oa.row(i), test_param.row(i), mysolver, rec_solver, best_obj) << ",";
            valuefile << oaf::get_loss(knnsols_mse.row(i), test_param.row(i), mysolver, rec_solver, best_obj) << ",";
            valuefile << best_obj << "\n";
        }
        valuefile.close();

        std::cout << "Writing Timing." << std::endl;

        std::ofstream timingfile(timing_filename, std::ios::out);
        timingfile << "OAF_FIT,"
                  << "OAF_PREDICT,"
                  << "OAF_SOL,"
                  << "LR_FIT,"
                  << "LR_PREDICT,"
                  << "LR_SOL,";
        timingfile << "RF_FIT,"
                  << "RF_PREDICT,"
                  << "RF_SOL,"
                  << "KNN_OA_SOL,"
                  << "KNN_OA_FIT,"
                  << "KNN_MSE_SOL,"
                  << "KNN_MSE_FIT,"
                  << "MODEL_BUILD\n";
        timingfile << oaffittime +oaf_setup_time<< "," << oafpredtime << "," << oafsoltime << ",";
        timingfile << lrfittime << "," << lrpredtime << "," << lrsoltime << ",";
        timingfile << rffittime << "," << rfpredtime << "," << rfsoltime << ",";
        timingfile << knnsoltime_oa  << "," << knnfittime_oa << ","
        << knnsoltime_mse  << "," << knnfittime_mse << ","<< modelbuildtime <<"\n";
    }
    else
    {
        throw std::invalid_argument("This executable needs to be run with eight command line arguments: optimization model, side info training set, parameter training set, side info test set, parameter test set, and output file");
    }

    return 0;
}