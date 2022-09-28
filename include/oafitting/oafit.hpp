#ifndef OAFIT_H
#define OAFIT_H
#include <Eigen/Dense>
#include "gurobi_c++.h"
#include <map>
#include <set>
#include <unistd.h>
namespace oaf
{

    struct FirstStageResult
    {
        bool feasible;
        mutable bool is_transpose; // True if decomposition is for the transposition of the basis rather than for the basis.
        double first_stage_cost;   // Cost incurred by solution in the first stage.
        Eigen::VectorXd ray_known;
        Eigen::VectorXd ray_predict;

        Eigen::MatrixXd matrix;
        Eigen::VectorXd lexmin_sol;
        mutable Eigen::ColPivHouseholderQR<Eigen::MatrixXd> decomp; //Decomposition for basis or transpose of basis.
        std::vector<int> first_included;                            //First-stage variable indices included in basis
        std::vector<int> rec_included;                              //Recourse variable indices included in basis.

        FirstStageResult(const Eigen::VectorXd &ray_known,
                         const Eigen::VectorXd &ray_predict) : feasible(false),
                                                               is_transpose(false),
                                                               first_stage_cost(),
                                                               ray_known(ray_known),
                                                               ray_predict(ray_predict),
                                                               matrix(),
                                                               lexmin_sol(), decomp(),
                                                               first_included(), rec_included(){};
        FirstStageResult(double first_stage_cost,
                         const Eigen::VectorXd &lexmin_sol, const Eigen::MatrixXd &matrix,
                         const std::vector<int> &first_included,
                         const std::vector<int> &rec_included,
                         bool is_transpose);

        void untranspose() const;
        void transpose() const;
        FirstStageResult(FirstStageResult &&);
        FirstStageResult &operator=(const FirstStageResult &) = delete;
        FirstStageResult &operator=(FirstStageResult &&);
        FirstStageResult(const FirstStageResult &) = delete;
    };

    struct RecResult
    {
        double second_stage_cost;  //Cost incurred by solution in the second stage.
        Eigen::VectorXd adj_costs; //
        RecResult(double second_stage_cost, Eigen::VectorXd adj_costs) : second_stage_cost(second_stage_cost),
                                                                         adj_costs(adj_costs){};
        //TODO: add more constructors.
    };

    struct GRBFirstStruct
    {
        Eigen::VectorXd first_costs;
        Eigen::VectorXd rec_costs;
        std::vector<GRBVar> first_vars;
        std::vector<GRBVar> rec_vars;
        Eigen::VectorXd known_rhs;
        std::vector<GRBConstr> known_constr;
        std::vector<GRBConstr> predict_constr;
        int dim_param;
        //TODO: there may be a memory leak; check copy constructor and copy assignment operator.
        //TODO: update implementations to set InfUnbdInfro to 1.

        GRBModel *model;
        GRBFirstStruct(GRBModel *model, const Eigen::VectorXd &first_costs, const Eigen::VectorXd &rec_costs,
                       const std::vector<GRBVar> &first_vars, const std::vector<GRBConstr> &known_constr,
                       const Eigen::VectorXd &known_rhs, const std::vector<GRBVar> &rec_vars,
                       const std::vector<GRBConstr> &predict_constr) : model(model), first_costs(first_costs),
                                                                       rec_costs(rec_costs),
                                                                       first_vars(first_vars), rec_vars(rec_vars),
                                                                       known_rhs(known_rhs), known_constr(known_constr),
                                                                       predict_constr(predict_constr),
                                                                       dim_param(predict_constr.size()){};
        ~GRBFirstStruct();
        GRBFirstStruct(const GRBFirstStruct &) = delete;
        GRBFirstStruct(GRBFirstStruct &&);
        GRBFirstStruct &operator=(const GRBFirstStruct &) = delete;
        GRBFirstStruct &operator=(GRBFirstStruct &&);
    };

    struct GRBRecDualStruct
    {
        GRBModel *model;
        std::vector<GRBVar> known_dual_vars;
        std::vector<GRBVar> predict_dual_vars;
        //TODO: I think that there is a memory leak here.
        //TODO: update implementations to set InfUnbdInfro to 1.

        GRBRecDualStruct(GRBModel *model, const std::vector<GRBVar> &known_dual_vars,
                         const std::vector<GRBVar> &predict_dual_vars) : model(model),
                                                                         known_dual_vars(known_dual_vars), predict_dual_vars(predict_dual_vars){};
        ~GRBRecDualStruct();
        GRBRecDualStruct(const GRBRecDualStruct &) = delete;
        GRBRecDualStruct(GRBRecDualStruct &&);
        GRBRecDualStruct &operator=(const GRBRecDualStruct &) = delete;
        GRBRecDualStruct &operator=(GRBRecDualStruct &&);
    };

    struct FittingResult
    {
        Eigen::MatrixXd coeff;
        Eigen::VectorXd intercept;
        //TODO: add move constructor, remove copy constructor.
        FittingResult(const Eigen::MatrixXd &coeff, const Eigen::VectorXd &intercept) : coeff(coeff), intercept(intercept){};
    };


    Eigen::MatrixXd predict(const FittingResult &, const Eigen::MatrixXd & test_data);

    struct GRBMasterStruct
    {
        std::vector<GRBVar> loss_vars;
        std::vector<GRBVar> intercept_vars;
        std::vector<std::vector<GRBVar>> coeff_vars;
        Eigen::VectorXd known_rhs;
        GRBModel model;
        int n_train;
        int side_dim;
        int param_dim;
        GRBMasterStruct(const GRBEnv &, const Eigen::VectorXd &, int, int, int);
        FittingResult readresult();
    };

    void addfeascut(GRBMasterStruct &mymastermodel, const Eigen::VectorXd &ray_known, const Eigen::VectorXd &ray_predict,
                    const Eigen::VectorXd &side_info);
    void addoptcut(GRBMasterStruct &mymastermodel, const Eigen::VectorXd &prediction,
                   const Eigen::VectorXd &side_info, Eigen::VectorXd &subgrad,
                   double loss, int loss_index);

    template <class F, class P>
    FittingResult oafit(const GRBEnv &myenv, const Eigen::MatrixXd &train_side,
                        const Eigen::MatrixXd &train_parameter, const Eigen::MatrixXd &coeff, const Eigen::VectorXd &intercept,
                        F &first_solver, P &dual_solver, double tol)
    {
        int n_train = train_side.rows();
        int param_dim = train_parameter.cols();
        int side_dim = train_side.cols();
        GRBMasterStruct mymastermodel{myenv, first_solver.get_known_rhs(), n_train, side_dim, param_dim};

        FittingResult myfit{coeff, intercept};
        std::vector<double> best_obj;
        for (int i = 0; i < n_train; i++)
        {
            double opt_val = first_solver.optimal_value(train_parameter.row(i));
            best_obj.push_back(opt_val);
        }

        std::vector<double> est_loss;
        for (int i = 0; i < n_train; i++)
        {
            est_loss.push_back(0.0);
        }

        bool done = false;
        bool immediately_done = true;
        while (!done)
        {
            Eigen::MatrixXd predictions = predict(myfit, train_side);
            
            done = !addcuts(mymastermodel, train_side, train_parameter, predictions, est_loss, best_obj,
                            first_solver, dual_solver, n_train, tol);
            if (!done)
            {
                immediately_done = false;
            }
            if (!immediately_done)
            { //This avoids an edge case where the guess is correct
                myfit = mymastermodel.readresult();
                for (int i = 0; i < n_train; i++)
                {
                    est_loss[i] = mymastermodel.loss_vars[i].get(GRB_DoubleAttr_X);
                }
            }
        }
        return FittingResult{myfit.coeff, myfit.intercept};
    }

    struct OptCut
    {
        double loss;
        Eigen::VectorXd prediction;
        Eigen::VectorXd subgrad;
        OptCut(const Eigen::VectorXd &prediction, const Eigen::VectorXd &subgrad, double loss) : prediction(prediction), subgrad(subgrad), loss(loss){};
    };

    Eigen::VectorXd get_subgradient(const FirstStageResult &, const RecResult &, int num_basis, int num_predict);

    template <class F, class R>
    double get_loss(const Eigen::VectorXd &fs_sol, const Eigen::VectorXd & actual_rhs,
     F & fs_solver, R &rec_solver, double best_obj)
    {
        RecResult rec_result = rec_solver.solve(fs_sol, actual_rhs);
        return rec_result.second_stage_cost + fs_solver.get_fs_costs(fs_sol) - best_obj;
    }

    double get_loss(const FirstStageResult &fs_result, const RecResult &rec_result, double best_obj)
    {
        return rec_result.second_stage_cost + fs_result.first_stage_cost - best_obj;
    }

    template <class F, class D>
    OptCut find_optcut(F &first_solver, D &rec_solver,
                       const Eigen::VectorXd &init_predict, const Eigen::VectorXd &actual, double best_obj,
                       const FirstStageResult &init_fs_result, const RecResult &init_rec_result, int num_basis, int num_predict,
                        double tol)
    {
        Eigen::VectorXd init_subgrad = get_subgradient(init_fs_result, init_rec_result, num_basis, num_predict);
        double init_loss = get_loss(init_fs_result, init_rec_result, best_obj);
        if (init_subgrad.dot(actual - init_predict) + init_loss <= 0.0)
        {
            return OptCut{init_predict, init_subgrad, init_loss};
        }

        double lb = 0;
        double ub = 1;
        Eigen::VectorXd lower_pt = actual;
        Eigen::VectorXd upper_pt = init_predict;
        while (true)
        {
            double ratio = (lb + ub) / 2.0;
            Eigen::VectorXd test_pt = (lower_pt + upper_pt) / 2.0;


            
            FirstStageResult fs_result = first_solver.get_lexmin(test_pt);
            RecResult rec_result = rec_solver.solve(fs_result.lexmin_sol, actual);
            Eigen::VectorXd subgrad = get_subgradient(fs_result, rec_result, num_basis, num_predict);



            double loss = get_loss(fs_result, rec_result, best_obj);

            if ((subgrad.dot(actual - test_pt) + loss) <= tol 
                && (subgrad.dot(init_predict - test_pt) + loss) >= init_loss-tol)
            {
                return OptCut{test_pt, subgrad, loss};
            }

            if (loss >= ratio * init_loss)
            {
                ub = ratio;
                upper_pt = test_pt;
            }
            else
            {
                lb = ratio;
                lower_pt = test_pt;
            }
            // sleep(1);

        }
    }

    template <class F, class D>
    bool addcuts(GRBMasterStruct &mymastermodel,
                 const Eigen::MatrixXd &train_side,
                 const Eigen::MatrixXd &train_parameter,
                 const Eigen::MatrixXd &predictions,
                 const std::vector<double> &est_loss,
                 const std::vector<double> &best_obj,
                 F &first_solver,
                 D &rec_solver, int n_train, double tol)
    {

        bool cuts_added = false;
        int n_basis = first_solver.get_basis_size();
        int n_predict = first_solver.get_n_predict();
        double max_gap = 0.0;
        int opt_cuts_added = 0;
        int feas_cuts_added = 0;
        for (int i = 0; i < n_train; i++)
        {
            Eigen::VectorXd side_info = train_side.row(i);
            Eigen::VectorXd actual = train_parameter.row(i);
            Eigen::VectorXd prediction = predictions.row(i);
            FirstStageResult myresult = first_solver.get_lexmin(prediction);

            if (myresult.feasible)
            {

                RecResult recresult = rec_solver.solve(myresult.lexmin_sol, actual);


                double loss = get_loss(myresult, recresult, best_obj[i]);
                if (loss - tol > est_loss[i])
                {
                    if (loss - est_loss[i] > max_gap)
                    {
                        max_gap = loss - est_loss[i];
                    }

                    OptCut optcut = find_optcut(first_solver, rec_solver, prediction, actual, best_obj[i],
                                                myresult, recresult, n_basis, n_predict, tol);


                    addoptcut(mymastermodel, optcut.prediction, side_info, optcut.subgrad, optcut.loss, i);
                    cuts_added = true;
                    opt_cuts_added++;
                }
                else
                {
                    // std::cout << "NO_CUT_REQUIRED" << std::endl;
                }
            }
            else
            {
                addfeascut(mymastermodel, myresult.ray_known, myresult.ray_predict, side_info);
                cuts_added = true;
                feas_cuts_added++;
            }
        }
        std::cout << "Added " << opt_cuts_added << " optimality cuts and " << feas_cuts_added << " feasibility cuts." << std::endl;
        std::cout << "Max gap: " << max_gap << std::endl;
        mymastermodel.model.update();
        return cuts_added;
    }

    struct Scenario
    {
        double probability;
        Eigen::VectorXd parameters;
        Scenario(double prob, Eigen::VectorXd params) : probability(prob), parameters(params){};
    };


    class ModelInitializer
    {
    public:
        virtual ~ModelInitializer(){};
        virtual GRBFirstStruct init_model(GRBEnv &, const Eigen::VectorXd &) const = 0;
        virtual Eigen::MatrixXd make_first_matrix() const = 0;
        virtual Eigen::MatrixXd make_rec_matrix() const = 0;
        virtual Eigen::VectorXd fixed_rhs() const = 0;
        virtual Eigen::VectorXd default_pred() const = 0;
        virtual GRBRecDualStruct init_rec_model(GRBEnv &, const Eigen::VectorXd &sol, const Eigen::VectorXd &rec_rhs) const = 0;
        virtual void update_rec_model(GRBRecDualStruct &, const Eigen::VectorXd &sol, const Eigen::VectorXd &rec_rhs) const = 0;
        virtual Eigen::VectorXd get_adj_costs(const Eigen::VectorXd &, const Eigen::VectorXd &) const = 0;
        virtual Eigen::MatrixXd solve_instances(const std::vector<std::vector<Scenario>> &) const =0;
    };

    class GRBFirstSolver
    {
    private:
        GRBFirstStruct mymodel;

        Eigen::VectorXd known_rhs;
        int basis_size;
        int n_predict;
        Eigen::MatrixXd first_matrix; 
        Eigen::MatrixXd rec_matrix;
        double basis_tol;

        FirstStageResult read_infeasible();
        FirstStageResult solve(bool transpose);
        Eigen::VectorXd get_unit_basis_costs(int, const FirstStageResult &);
        Eigen::VectorXd get_basis_costs(const FirstStageResult &);
        bool is_already_solved(int, const std::map<int, Eigen::VectorXd> &, const std::map<int, Eigen::VectorXd> &,
                               const std::set<int> &, const std::set<int> &, const FirstStageResult &,
                               const std::set<int> &first_basis, const std::set<int> &rec_basis);
        void reset_model();
        void zero_obj();

    public:
        GRBFirstSolver(GRBEnv &grb_env, const ModelInitializer &initializer,
                       const Eigen::VectorXd &initial_pred,
                       double basis_tol = 0.000001) : mymodel(std::move(initializer.init_model(grb_env, initial_pred))),
                                                      first_matrix(std::move(initializer.make_first_matrix())),
                                                      rec_matrix(std::move(initializer.make_rec_matrix())),
                                                      known_rhs(initializer.fixed_rhs()), basis_tol(basis_tol)
        {
            n_predict = initial_pred.size();
            basis_size = first_matrix.rows();
            mymodel.model->set(GRB_IntParam_InfUnbdInfo, 1);
        }

        FirstStageResult solve(const Eigen::VectorXd &prediction, bool);

        Eigen::MatrixXd solve_all(const Eigen::MatrixXd & predictions);
        double get_fs_costs(const Eigen::VectorXd & solution);
        double optimal_value(const Eigen::VectorXd &prediction);
        void update_rhs(const Eigen::VectorXd &new_rec);
        Eigen::VectorXd get_known_rhs();
        int get_basis_size();
        int get_n_predict();
        FirstStageResult get_lexmin(const Eigen::VectorXd &);
    };

    class GRBRecSolver
    {
    private:
        ModelInitializer &modelbuilder; // NOTE: this will free the object when it goes out of scope.
                                        //The point
        GRBRecDualStruct mymodel;

        GRBRecSolver(const GRBRecSolver &) = delete;
        GRBRecSolver(GRBRecDualStruct &&) = delete;
        GRBRecSolver &operator=(const GRBRecSolver &) = delete;
        GRBRecSolver &operator=(GRBRecSolver &&) = delete;

    public:
        GRBRecSolver(GRBEnv &grb_env, ModelInitializer &modelbuilder, const Eigen::VectorXd &initial_sol,
                     const Eigen::VectorXd &initial_rhs) : mymodel(modelbuilder.init_rec_model(grb_env, initial_sol, initial_rhs)),
                                                           modelbuilder(modelbuilder){};

        RecResult solve(const Eigen::VectorXd first_stage_sol, const Eigen::VectorXd &actual_rhs)
        {
            modelbuilder.update_rec_model(mymodel, first_stage_sol, actual_rhs);
            mymodel.model->optimize();
            Eigen::VectorXd known_dual_sol{mymodel.known_dual_vars.size()};
            //TODO: be more careful about feasibility
            int i = 0;
            for (GRBVar myvar : mymodel.known_dual_vars)
            {
                known_dual_sol(i) = myvar.get(GRB_DoubleAttr_X);
                i++;
            }
            Eigen::VectorXd predict_dual_sol{mymodel.predict_dual_vars.size()};
            i = 0;
            for (GRBVar myvar : mymodel.predict_dual_vars)
            {
                predict_dual_sol(i) = myvar.get(GRB_DoubleAttr_X);
                i++;
            }
            return RecResult{mymodel.model->get(GRB_DoubleAttr_ObjVal),
                             modelbuilder.get_adj_costs(known_dual_sol, predict_dual_sol)};
        }
    };
}

#endif