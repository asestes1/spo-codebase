#ifndef LINREG_H
#define LINREG_H
#include <Eigen/Dense>
#include <oafitting/oafit.hpp>
#include <opencv2/core.hpp>

namespace lr
{
    cv::Mat eigen_to_cv_matrix(const Eigen::MatrixXd &);
    cv::Mat eigen_to_cv_vector(const Eigen::VectorXd &);
    oaf::FittingResult fitmodel(const Eigen::MatrixXd &side_train, const Eigen::MatrixXd &param_train);
    Eigen::MatrixXd rf_fitnpredict(const Eigen::MatrixXd &side_train, const Eigen::MatrixXd &param_train,
                                   const Eigen::MatrixXd & side_test);
    Eigen::MatrixXd rf_fitnpredict(const Eigen::MatrixXd &side_train, const Eigen::MatrixXd &param_train,
                                   const Eigen::MatrixXd & side_test, double & fittingtime, double & predicttime);
}

#endif