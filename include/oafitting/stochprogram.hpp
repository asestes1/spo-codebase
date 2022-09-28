#ifndef STOCHPROGRAM_H
#define STOCHPROGRAM_H

#include <Eigen/Dense>

#include <oafitting/oafit.hpp>
#include <oafitting/linreg.hpp>
#include <vector>
#include <queue>
#include <chrono>
#include <math.h>

namespace sp
{
    Eigen::MatrixXd minus_to_zero(const Eigen::MatrixXd &matrix);
    class EuclideanMetric
    {
    public:
        double evaluate(const Eigen::VectorXd &vector1, const Eigen::VectorXd &vector2) const;
    };

    struct KnnKey
    {
        int index;
        double distance;
        KnnKey(int index, double distance) : index(index), distance(distance){};
    };

    struct KnnFitParams
    {
        double validateprop;
        std::vector<int> possiblevals;
        KnnFitParams(double validateprop, std::vector<int> possiblevals) : validateprop(validateprop), possiblevals(possiblevals){};
    };

    KnnFitParams read_knn_params(std::string filename);

    class KnnKeyComparator
    {
    public:
        int operator()(const KnnKey &key1, const KnnKey &key2)
        {
            return key1.distance < key2.distance;
        }
    };

    class GRBLossCalculator
    {
    private:
        oaf::GRBFirstSolver &fsolver;
        oaf::GRBRecSolver &rsolver;

    public:
        double get_avgloss(const Eigen::MatrixXd &solutions, const Eigen::MatrixXd &parameters);
        GRBLossCalculator(oaf::GRBFirstSolver &fsolver, oaf::GRBRecSolver &rsolver) : fsolver(fsolver), rsolver(rsolver){};
        GRBLossCalculator(GRBLossCalculator &&) = delete;
        GRBLossCalculator &operator=(const GRBLossCalculator &) = delete;
        GRBLossCalculator &operator=(GRBLossCalculator &&) = delete;
        GRBLossCalculator(const GRBLossCalculator &) = delete;
    };

    template <class Metric>
    std::vector<int> find_knn(const Eigen::MatrixXd &train_side_info, int k,
                              const Eigen::VectorXd &test_side_info, const Metric &mymetric)
    {

        int num_train = train_side_info.rows();
        std::vector<int> knn_indices{};
        if (k > num_train)
        {
            for (int i = 0; i < num_train; i++)
            {
                knn_indices.push_back(i);
            }
            return knn_indices;
        }

        std::priority_queue<KnnKey, std::vector<KnnKey>, KnnKeyComparator> myheap{};
        for (int i = 0; i < k; i++)
        {
            myheap.emplace(i, mymetric.evaluate(train_side_info.row(i), test_side_info));
        }
        for (int i = k; i < num_train; i++)
        {
            double next_distance = mymetric.evaluate(train_side_info.row(i), test_side_info);
            KnnKey top = myheap.top();
            if (next_distance < top.distance)
            {
                myheap.pop();
                myheap.emplace(i, mymetric.evaluate(train_side_info.row(i), test_side_info));
            }
        }
        for (int i = 0; i < k; i++)
        {
            knn_indices.push_back(myheap.top().index);
            myheap.pop();
        }
        return knn_indices;
    }

    template <class SpSolver, class Metric>
    Eigen::MatrixXd knn_sp_solution(const Eigen::MatrixXd &train_side_info,
                                    const Eigen::MatrixXd &train_param,
                                    const Eigen::MatrixXd &test_side_info,
                                    int k,
                                    const SpSolver &mysolver,
                                    const Metric &mymetric,
                                    double &solutiontime)
    {

        std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        int numinstances = test_side_info.rows();
        std::vector<std::vector<oaf::Scenario>> instances;
        for (int i = 0; i < numinstances; i++)
        {
            std::vector<oaf::Scenario> next_instance;
            std::vector<int> knn_indices = find_knn(train_side_info, k, test_side_info.row(i), mymetric);
            
            int n_indices = knn_indices.size();
            double probability = 1.0 / n_indices;
            for (int j : knn_indices)
            {
                next_instance.emplace_back(probability, train_param.row(j));
            }
            instances.push_back(next_instance);
        }
        Eigen::MatrixXd solutions = mysolver.solve_instances(instances);
        std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
        solutiontime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return solutions;
    }

    template <class Metric>
    Eigen::MatrixXd knn_pred(const Eigen::MatrixXd &train_side_info,
                                    const Eigen::MatrixXd &train_param,
                                    const Eigen::MatrixXd &test_side_info,
                                    int k,
                                    const Metric &mymetric)
    {
        int numinstances = test_side_info.rows();
        int param_dim = train_param.cols();
        Eigen::MatrixXd preds = Eigen::MatrixXd::Zero(numinstances, param_dim);
        for (int i = 0; i < numinstances; i++)
        {
            std::vector<int> knn_indices = find_knn(train_side_info, k, test_side_info.row(i), mymetric);
            int n_indices = knn_indices.size();
            Eigen::VectorXd myvector = Eigen::VectorXd::Zero(param_dim);
            for(int knn_index: knn_indices){
                myvector += train_param.row(knn_index);
            }
            preds.row(i) = myvector/knn_indices.size();
        }
        return preds;
    }

    template <class Metric>
    int find_best_k_mse(const Eigen::MatrixXd &train_side_info, const Eigen::MatrixXd &train_param, const Metric &mymetric,
                    double validateprop, const std::vector<int> &possiblevals,
                    double &fittime)
    {
        fittime = 0.0;
        std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        if (possiblevals.size() == 1)
        {
            std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
            fittime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();;
            return possiblevals[0];
        }
        int total_train = train_side_info.rows();
        int num_train = (int)total_train * validateprop;
        int num_validate = total_train - num_train;
        bool firstround = true;
        int bestk = -1;
        double best_loss = -1;
        for (int possiblek : possiblevals)
        {
            
            Eigen::MatrixXd knnsols = sp::knn_pred(train_side_info.topRows(num_train),
                                                   train_param.topRows(num_train),
                                                   train_side_info.bottomRows(num_validate),
                                                   possiblek,
                                                   mymetric);
            double avgloss = 0.0;
            Eigen::MatrixXd pred_diff = train_param.bottomRows(num_validate) - knnsols;

            for(int i =0; i< num_validate; i++){
                avgloss += pred_diff.row(i).dot(pred_diff.row(i));
            }

            avgloss = avgloss/num_validate;
            if (avgloss < best_loss || firstround)
            {
                best_loss=avgloss;
                bestk = possiblek;
                firstround=false;
            }
        }
        std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
        fittime=std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return bestk;
    }

    template <class SpSolver, class Metric, class LossCalculator>
    int find_best_k_oaloss(const Eigen::MatrixXd &train_side_info, const Eigen::MatrixXd &train_param, const Metric &mymetric,
                    const SpSolver &mysolver,
                    LossCalculator &mylosscalc, double validateprop, const std::vector<int> &possiblevals,
                    double &fittime)
    {
        fittime = 0.0;
        std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        if (possiblevals.size() == 1)
        {
            std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
            fittime= std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            return possiblevals[0];
        }
        int total_train = train_side_info.rows();
        int num_train = (int)total_train * validateprop;
        int num_validate = total_train - num_train;
        bool firstround = true;
        int bestk = -1;
        double best_loss = -1;
        for (int possiblek : possiblevals)
        {
            double dummy;
            Eigen::MatrixXd knnsols = sp::knn_sp_solution(train_side_info.topRows(num_train),
                                                          train_param.topRows(num_train), train_side_info.bottomRows(num_validate), possiblek,
                                                          mysolver, mymetric, dummy);
            double avgloss = mylosscalc.get_avgloss(knnsols, train_param.bottomRows(num_validate));
            if (avgloss < best_loss || firstround)
            {
                firstround = false;
                best_loss = avgloss;
                bestk = possiblek;
            }
        }
        std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
        fittime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return bestk;
    }
    template <class LRFitter, class DetSolver>
    Eigen::MatrixXd lr_ftp(const Eigen::MatrixXd &train_side_info,
                           const Eigen::MatrixXd &train_param,
                           const Eigen::MatrixXd &test_side_info,
                           const LRFitter &fitter,
                           const DetSolver &solver)
    {
        double dummydouble;
        return lr_ftp(train_side_info, train_param, test_side_info, fitter, solver, dummydouble, dummydouble, dummydouble);
    }

    class LeastSqLrFitter
    {
    public:
        oaf::FittingResult fit(const Eigen::MatrixXd &train_side_info,
                               const Eigen::MatrixXd &train_param) const;
    };

    template <typename F, typename P>
    struct OaLrFitter
    {
        F &first_solver;
        P &dual_solver;
        oaf::FittingResult guess;
        double tol;

        OaLrFitter(F &first_solver, P &dual_solver, oaf::FittingResult guess,
                   double tol) : first_solver(first_solver), dual_solver(dual_solver), guess(guess), tol(tol){};
        OaLrFitter(const OaLrFitter &other) = delete;
        OaLrFitter(OaLrFitter &&other) = delete;
        OaLrFitter &operator=(const OaLrFitter &other) = delete;
        OaLrFitter &operator=(OaLrFitter &&other) = delete;

        oaf::FittingResult fit(const Eigen::MatrixXd &train_side_info,
                               const Eigen::MatrixXd &train_param)
        {
            GRBEnv myenv{};
            return oafit(myenv, train_side_info, train_param, guess.coeff, guess.intercept, first_solver, dual_solver, tol);
        }
    };

    template <class LRFitter, class DetSolver>
    Eigen::MatrixXd lr_ftp(const Eigen::MatrixXd &train_side_info,
                           const Eigen::MatrixXd &train_param,
                           const Eigen::MatrixXd &test_side_info,
                           LRFitter &fitter,
                           DetSolver &solver,
                           double &fitting_time,
                           double &prediction_time,
                           double &solution_time)
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        oaf::FittingResult myresult = fitter.fit(train_side_info, train_param);
        std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
        fitting_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd predictions = minus_to_zero(oaf::predict(myresult, test_side_info));
        end = std::chrono::high_resolution_clock::now();
        prediction_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        int num_instances = predictions.rows();
        start = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd solutions = solver.solve_all(predictions);
        end = std::chrono::high_resolution_clock::now();
        solution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return solutions;
    }

    template <class DetSolver>
    Eigen::MatrixXd rf_ftp(const Eigen::MatrixXd &train_side_info,
                           const Eigen::MatrixXd &train_param,
                           const Eigen::MatrixXd &test_side_info,
                           DetSolver &solver)
    {
        double dummy;
        return rf_ftp(train_side_info, train_param, test_side_info, solver, dummy, dummy, dummy);
    }

    template <class DetSolver>
    Eigen::MatrixXd rf_ftp(const Eigen::MatrixXd &train_side_info,
                           const Eigen::MatrixXd &train_param,
                           const Eigen::MatrixXd &test_side_info,
                           DetSolver &solver,
                           double &fitting_time,
                           double &prediction_time,
                           double &solution_time)
    {

        Eigen::MatrixXd predictions = minus_to_zero(lr::rf_fitnpredict(train_side_info, train_param, test_side_info, fitting_time, prediction_time));
        int num_instances = predictions.rows();
        std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd solutions = solver.solve_all(predictions);
        std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
        solution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        return solutions;
    }
}

#endif