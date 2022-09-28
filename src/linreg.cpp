#include <oafitting/oafit.hpp>
#include <oafitting/linreg.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/ml.hpp>
#include <chrono>

oaf::FittingResult lr::fitmodel(const Eigen::MatrixXd &side_train, const Eigen::MatrixXd &param_train)
{
    int num_samples = side_train.rows();
    int num_features = side_train.cols();
    int num_params = param_train.cols();
    Eigen::MatrixXd design_matrix(num_samples, num_features + 1);
    design_matrix.col(0) = Eigen::VectorXd::Ones(num_samples);
    design_matrix.rightCols(num_features) = side_train;


    Eigen::VectorXd intercept(num_params);
    Eigen::MatrixXd coeff(num_params, num_features);

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> mydecomp(design_matrix.transpose() * design_matrix);
    for (int i = 0; i < num_params; i++)
    {
        Eigen::VectorXd result = mydecomp.solve(design_matrix.transpose() * param_train.col(i));
        intercept(i) = result(0);
        coeff.row(i) = result.bottomRows(num_features).transpose();
    }
    return oaf::FittingResult(std::move(coeff), std::move(intercept));
}

cv::Mat lr::eigen_to_cv_matrix(const Eigen::MatrixXd &mymatrix)
{
    int nrows = mymatrix.rows();
    int ncols = mymatrix.cols();
    cv::Mat outmatrix(nrows, ncols, CV_32F);
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            outmatrix.at<float>(i, j) = mymatrix(i, j);
        }
    }
    return outmatrix;
}

cv::Mat lr::eigen_to_cv_vector(const Eigen::VectorXd &myvector)
{
    int nelem = myvector.size();
    cv::Mat outmatrix(nelem, 1, CV_32F);
    for (int i = 0; i < nelem; i++)
    {
        outmatrix.at<float>(i, 0) = myvector(i);
    }
    return outmatrix;
}

Eigen::MatrixXd lr::rf_fitnpredict(const Eigen::MatrixXd &side_train, const Eigen::MatrixXd &param_train,
                                   const Eigen::MatrixXd &side_test)
{
    double dummy;
    return rf_fitnpredict(side_train, param_train, side_test, dummy, dummy);
}

Eigen::MatrixXd lr::rf_fitnpredict(const Eigen::MatrixXd &side_train, const Eigen::MatrixXd &param_train,
                                   const Eigen::MatrixXd &side_test, double & fittingtime, double & predicttime)
{
    int nparam = param_train.cols();
    int ntest = side_test.rows();
    Eigen::MatrixXd mypreds(side_test.rows(), param_train.cols());
    cv::Mat train_in = eigen_to_cv_matrix(side_train);
    fittingtime = 0;
    predicttime = 0;

    for (int i = 0; i < nparam; i++)
    {
        cv::Mat param_in = eigen_to_cv_vector(param_train.col(i));

        
        std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        cv::Ptr<cv::ml::TrainData> mytrain = cv::ml::TrainData::create(train_in,
                                                                       cv::ml::ROW_SAMPLE,
                                                                       param_in);
        
        cv::Ptr<cv::ml::RTrees> mytrees = cv::ml::RTrees::create();
        mytrees->train(mytrain);
        std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
        fittingtime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        cv::Mat outmatrix;
        
        start = std::chrono::high_resolution_clock::now();
        mytrees->predict(eigen_to_cv_matrix(side_test), outmatrix);
        end = std::chrono::high_resolution_clock::now();
        predicttime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        for (int j = 0; j < ntest; j++)
        {
            mypreds(j, i) = outmatrix.at<float>(j);
        }
    }
    return mypreds;
}
