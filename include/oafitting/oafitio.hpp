#ifndef OAFITIO_H
#define OAFITIO_H
#include <Eigen/Dense>
#include <oafitting/oafit.hpp>
#include <oafitting/transport.hpp>
#include <boost/property_tree/ptree.hpp>
#include <oafitting/vendor.hpp>

namespace pt = boost::property_tree;

namespace oafio{
    Eigen::VectorXd read_vector(const pt::ptree & root);
    Eigen::MatrixXd read_eigen_matrix(const pt::ptree & root, int rows, int cols);

    Eigen::MatrixXd read_eigen_matrix(const std::string &, const std::string &);
    Eigen::MatrixXd read_eigen_matrix(const std::string &);

    VendorInitializer* read_vendor_initializer(const pt::ptree & root);
    TransportationInitializer* read_transport_initializer(const pt::ptree & root);
    oaf::ModelInitializer* read_model_initializer(const std::string & filename);
    oaf::FittingResult read_linear_model(const std::string & filename);
    void write_linear_model(const std::string & filename, const oaf::FittingResult &);
    void write_eigen_matrix(const std::string & filename, const Eigen::MatrixXd &);
}

#endif