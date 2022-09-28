#include <oafitting/oafitio.hpp>
#include <oafitting/transport.hpp>
#include <Eigen/Dense>
#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include<boost/property_tree/ptree.hpp>
#include<boost/property_tree/json_parser.hpp>


Eigen::MatrixXd oafio::read_eigen_matrix(const std::string & filename){
    return read_eigen_matrix(filename, ",");
}

/**
 * This takes a file where the first row is a single integer describing the number of rows in the matrix, 
 * the second row is the number of columns in the matrix,
 * and each row is a delimited description of the matrix.
 * 
 */
Eigen::MatrixXd oafio::read_eigen_matrix(const std::string & filename, const std::string & delim)
{

    std::ifstream myfile(filename, std::ios::in);
    std::string line;
    if (!myfile.is_open())
    {
        std::string myerror = "Unable to open input file specifying matrix. Filename: ";
        throw std::runtime_error(myerror.append(filename));
    }

    int rows = -1;
    int cols = -1;
    if (std::getline(myfile, line))
    {
        rows = std::stoi(line);
    }
    if (std::getline(myfile, line))
    {
        cols = std::stoi(line);
    }
    if (rows == -1 || cols == -1)
        throw std::invalid_argument("Error reading rows and columns of matrix file.");

    Eigen::MatrixXd mymatrix{rows, cols};
    
    int rows_read = 0;
    while (std::getline(myfile, line))
    {
        int cols_read = 0;    
        std::size_t next_break = line.find(delim);
        while (next_break != std::string::npos)
        {
            std::string next_str = line.substr(0, next_break);
            mymatrix(rows_read, cols_read) = std::stod(next_str);
            line.erase(0, next_break + 1);
            next_break = line.find(delim);
            cols_read++;
        }
        mymatrix(rows_read, cols_read) = std::stod(line);
        cols_read++;
        if (cols_read != cols)
        {
            std::string error_msg = "Some row in matrix file doesn't match specified number of cols. Row number: ";
            error_msg.append(std::to_string(rows_read)).append(" Cols read: ").append(std::to_string(cols_read));
            throw std::invalid_argument(error_msg);
        }
        rows_read++;
    }
    if (rows_read != rows)
    {
        throw std::invalid_argument("Columns in matrix file don't match specified number of columns.");
    }
    myfile.close();

    return mymatrix;
}

Eigen::MatrixXd oafio::read_eigen_matrix(const pt::ptree & root, int rows, int cols){
    Eigen::MatrixXd mymatrix{rows,cols};
    int i =0;
    for(const pt::ptree::value_type & rowdata: root){
        int j=0;
        for(const pt::ptree::value_type & coldata: rowdata.second){
            mymatrix(i,j) = coldata.second.get_value<double>();
            j++;
        }
        i++;
    }
    return mymatrix;
}

Eigen::VectorXd oafio::read_vector(const pt::ptree & root){
    std::vector<double> myvector;
    for(const pt::ptree::value_type & element: root){
        myvector.push_back(element.second.get_value<double>());
    }
    Eigen::VectorXd mynewvector(myvector.size());
    int i=0;
    for(const double & myelement: myvector ){
        mynewvector(i) = myelement;
        i++;
    }
    return mynewvector;
}

VendorInitializer* oafio::read_vendor_initializer(const pt::ptree & root){
    Eigen::VectorXd buy_costs = read_vector(root.get_child("buy_costs"));
    Eigen::VectorXd sell_costs = read_vector(root.get_child("sell_costs"));
    Eigen::VectorXd salvage_values = read_vector(root.get_child("salvage_values"));
    double budget = root.get<double>("budget");
    double volume = root.get<double>("volume");
    return new  VendorInitializer(buy_costs, sell_costs, salvage_values, budget, volume);
}

TransportationInitializer* oafio::read_transport_initializer(const pt::ptree & root){
    Eigen::VectorXd prod_costs = read_vector(root.get_child("prod_costs"));
    Eigen::VectorXd unmet_costs = read_vector(root.get_child("unmet_costs"));
    Eigen::VectorXd scrap_costs = read_vector(root.get_child("scrap_costs"));
    Eigen::MatrixXd transport_costs = read_eigen_matrix(root.get_child("transport_costs"), prod_costs.size(),
                unmet_costs.size());
    return new TransportationInitializer(prod_costs, scrap_costs, unmet_costs, transport_costs);
}

oaf::ModelInitializer* oafio::read_model_initializer(const std::string & filename){
    pt::ptree root;
    pt::read_json(filename, root);
    std::string modeltype = root.get<std::string>("modeltype");
    if(modeltype == "transportation"){
        return read_transport_initializer(root);
    }else if(modeltype == "vendor"){
        return read_vendor_initializer(root);
    }else{
        throw std::invalid_argument("Error in reading model file: improper model type.");
    }
}

oaf::FittingResult oafio::read_linear_model(const std::string & filename){
    pt::ptree root;
    pt::read_json(filename, root);
    int rows = root.get<int>("sidedim",0);
    int cols = root.get<int>("paramdim",0);
    Eigen::VectorXd intercept = read_vector(root.get_child("intercept"));
    Eigen::MatrixXd coeff = read_eigen_matrix(root.get_child("coeff"), cols, rows);
    return oaf::FittingResult(coeff, intercept);
}

void oafio::write_linear_model(const std::string & filename, const oaf::FittingResult & result){
    pt::ptree root;
    int rows = result.coeff.rows();
    int cols = result.coeff.cols();
    root.put("sidedim", cols);
    root.put("paramdim", rows);
    
    pt::ptree coeff_node;
    for(int i =0; i < rows; i++){
        pt::ptree row_node;
        for(int j =0; j < cols; j++){
            pt::ptree cell;
            cell.put_value(result.coeff(i,j));
            row_node.push_back(std::make_pair("", cell));
        }
        coeff_node.push_back(std::make_pair("", row_node));
    }

    root.add_child("coeff", coeff_node);

    pt::ptree intercept_node;
    for(int i=0; i < rows; i++){
        pt::ptree cell;
        cell.put_value(result.intercept(i));
        intercept_node.push_back(std::make_pair("", cell));
    }
    root.add_child("intercept", intercept_node);
    pt::write_json(filename, root);
    return;
}

void oafio::write_eigen_matrix(const std::string & filename, const Eigen::MatrixXd & mymatrix){
    std::ofstream myfile(filename, std::ios::in);
    if (!myfile.is_open())
    {
        std::string myerror = "Unable to open input file specifying matrix. Filename: ";
        throw std::runtime_error(myerror.append(filename));
    }
    int nrows = mymatrix.rows();
    int ncols = mymatrix.cols();
    myfile << nrows << "\n";
    myfile << ncols << "\n";
    for(int i=0; i < nrows; i++){
        for(int j=0; j < ncols; j++){
            myfile << mymatrix(i,j);
            if(j < ncols-1){
                myfile << ",";
            }

        }
        myfile<<"\n";
    }
    myfile.close();
}
