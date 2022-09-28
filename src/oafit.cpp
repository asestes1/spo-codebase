#include <oafitting/oafit.hpp>
#include <string>
#include <set>
#include <map>

oaf::FirstStageResult::FirstStageResult(double first_stage_cost,
                                        const Eigen::VectorXd &lexmin_sol,
                                        const Eigen::MatrixXd &matrix,
                                        const std::vector<int> &first_included,
                                        const std::vector<int> &rec_included,
                                        bool is_transpose) : first_stage_cost(first_stage_cost), feasible(true),
                                                             ray_known(), ray_predict(), lexmin_sol(lexmin_sol),
                                                             matrix(matrix), decomp(), first_included(first_included),
                                                             rec_included(rec_included),
                                                             is_transpose(is_transpose)
{
    if (is_transpose)
    {
        decomp.compute(matrix.transpose());
    }
    else
    {
        decomp.compute(matrix);
    }
}

void oaf::FirstStageResult::transpose() const
{
    if (!is_transpose)
    {
        decomp.compute(matrix.transpose());
    }
}

void oaf::FirstStageResult::untranspose() const
{
    if (is_transpose)
    {
        decomp.compute(matrix);
    }
}

oaf::FirstStageResult::FirstStageResult(oaf::FirstStageResult &&other_results) : first_stage_cost(other_results.first_stage_cost),
                                                                                 feasible(other_results.feasible),
                                                                                 ray_known(std::move(other_results.ray_known)),
                                                                                 ray_predict(std::move(other_results.ray_predict)),
                                                                                 lexmin_sol(std::move(other_results.lexmin_sol)),
                                                                                 matrix(std::move(other_results.matrix)),
                                                                                 decomp(std::move(other_results.decomp)),
                                                                                 first_included(std::move(other_results.first_included)),
                                                                                 rec_included(std::move(other_results.rec_included)) {}

oaf::FirstStageResult &oaf::FirstStageResult::operator=(oaf::FirstStageResult &&other)
{
    if (this != &other)
    {
        this->first_stage_cost = other.first_stage_cost;
        this->feasible = other.feasible;
        this->ray_known = std::move(other.ray_known);
        this->ray_predict = std::move(other.ray_predict);
        this->lexmin_sol = std::move(other.lexmin_sol);
        this->matrix = std::move(other.matrix);
        this->decomp = std::move(other.decomp);
        this->first_included = std::move(other.first_included);
        this->rec_included = std::move(other.rec_included);
    }
    return other;
}

Eigen::MatrixXd oaf::predict(const FittingResult & myfit, const Eigen::MatrixXd & test_data){
    Eigen::MatrixXd mymatrix = (test_data * myfit.coeff.transpose()).rowwise() + myfit.intercept.transpose();
    int numrows = mymatrix.rows();
    int numcols = mymatrix.cols();
    for(int i=0; i < numrows; i++){
        for(int j=0; j < numcols; j++){
            if(mymatrix(i,j) > -0.000001 && mymatrix(i,j) < 0.000001){
                mymatrix(i,j) = 0;
            }
        }
    }
    return mymatrix;
}

oaf::GRBMasterStruct::GRBMasterStruct(const GRBEnv &myenv, const Eigen::VectorXd &known_rhs,
                                      int n_train, int side_dim, int param_dim) : model(myenv),
                                                                                  loss_vars(),
                                                                                  intercept_vars(),
                                                                                  coeff_vars(),
                                                                                  n_train(n_train),
                                                                                  side_dim(side_dim),
                                                                                  param_dim(param_dim),
                                                                                  known_rhs(known_rhs)
{
    model.set(GRB_IntParam_NumericFocus, 3);
    double loss_obj = 1.0;
    for (int i = 0; i < n_train; i++)
    {
        loss_vars.push_back(model.addVar(0.0, GRB_INFINITY, loss_obj, GRB_CONTINUOUS, "L" + std::to_string(i)));
    }

    for (int i = 0; i < param_dim; i++)
    {
        intercept_vars.push_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "I" + std::to_string(i)));
    }

    for (int i = 0; i < param_dim; i++)
    {
        std::vector<GRBVar> row_coeff_vars;
        for (int j = 0; j < side_dim; j++)
        {
            row_coeff_vars.push_back(model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                                  "C(" + std::to_string(i) + ", " + std::to_string(j) + ")"));
        }
        coeff_vars.push_back(std::move(row_coeff_vars));
    }
    model.update();
}

oaf::FittingResult oaf::GRBMasterStruct::readresult()
{
    if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL)
    {
        model.optimize();
        if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL)
        {
            throw "Error in reading fit coefficients; GRB model not optimizable.";
        }
    }

    Eigen::VectorXd new_intercept(param_dim);
    int i = 0;
    for (const GRBVar &myvar : intercept_vars)
    {
        new_intercept(i) = myvar.get(GRB_DoubleAttr_X);
        i++;
    }

    Eigen::MatrixXd new_coeff(param_dim, side_dim);
    i = 0;
    for (const std::vector<GRBVar> &myvar_row : coeff_vars)
    {
        int j = 0;
        for (const GRBVar &myvar : myvar_row)
        {
            new_coeff(i, j) = myvar.get(GRB_DoubleAttr_X);
            j++;
        }

        i++;
    }
    return FittingResult{new_coeff, new_intercept};
}

oaf::GRBFirstStruct::~GRBFirstStruct()
{
    delete model;
}

oaf::GRBFirstStruct::GRBFirstStruct(GRBFirstStruct &&other) : model(other.model),
                                                              first_costs(std::move(other.first_costs)),
                                                              rec_costs(std::move(other.rec_costs)),
                                                              first_vars(std::move(other.first_vars)),
                                                              rec_vars(std::move(other.rec_vars)),
                                                              known_constr(std::move(other.known_constr)),
                                                              predict_constr(std::move(other.predict_constr))
{
    if (this != &other)
    {
        other.model = nullptr;
    }
}

oaf::GRBFirstStruct &oaf::GRBFirstStruct::operator=(oaf::GRBFirstStruct &&other)
{
    if (this != &other)
    {
        this->model = other.model;
        this->first_costs = std::move(other.first_costs);
        this->rec_costs = std::move(other.rec_costs);
        this->first_vars = std::move(other.first_vars);
        this->rec_vars = std::move(other.rec_vars);
        this->known_constr = std::move(other.known_constr);
        this->predict_constr = std::move(other.predict_constr);
        this->dim_param = other.dim_param;
        other.model = nullptr;
    }

    return *this;
}

oaf::GRBRecDualStruct::~GRBRecDualStruct()
{
    delete model;
}

oaf::GRBRecDualStruct::GRBRecDualStruct(GRBRecDualStruct &&other) : model(other.model),
                                                                    known_dual_vars(std::move(other.known_dual_vars)),
                                                                    predict_dual_vars(std::move(other.predict_dual_vars))
{
    if (this != &other)
    {
        other.model = nullptr;
    }
}

oaf::GRBRecDualStruct &oaf::GRBRecDualStruct::operator=(oaf::GRBRecDualStruct &&other)
{
    if (this != &other)
    {
        this->model = other.model;
        this->known_dual_vars = std::move(other.known_dual_vars);
        this->predict_dual_vars = std::move(other.predict_dual_vars);
        other.model = nullptr;
    }

    return *this;
}

void oaf::addfeascut(GRBMasterStruct &mymastermodel, const Eigen::VectorXd &ray_known, const Eigen::VectorXd &ray_predict,
                     const Eigen::VectorXd &side_info)
{
    GRBLinExpr myexpr = GRBLinExpr(ray_known.dot(mymastermodel.known_rhs));
    myexpr.addTerms(ray_predict.data(), &mymastermodel.intercept_vars[0], mymastermodel.param_dim);
    Eigen::MatrixXd coeff_cut = ray_predict * side_info.transpose();
    int i = 0;
    for (const std::vector<GRBVar> &myvars : mymastermodel.coeff_vars)
    {
        int j = 0;
        for (const GRBVar &myvar : myvars)
        {
            myexpr += coeff_cut(i, j) * myvar;
            j++;
        }

        i++;
    }

    mymastermodel.model.addConstr(myexpr, GRB_LESS_EQUAL, 0);
}

void oaf::addoptcut(GRBMasterStruct &mymastermodel, const Eigen::VectorXd &prediction,
                    const Eigen::VectorXd &side_info, Eigen::VectorXd &subgrad,
                    double loss, int loss_index)
{
    GRBLinExpr myexpr = GRBLinExpr(loss - subgrad.dot(prediction));
    myexpr.addTerms(subgrad.data(), &mymastermodel.intercept_vars[0], mymastermodel.param_dim);
    Eigen::MatrixXd coeff_cut = subgrad * side_info.transpose();
    int i = 0;
    
    for (const std::vector<GRBVar> &myvars : mymastermodel.coeff_vars)
    {
        int j = 0;
        for (const GRBVar &myvar : myvars)
        {
            
            myexpr += coeff_cut(i, j) * myvar;
            j++;
        }
        i++;
    }
    

    mymastermodel.model.addConstr(myexpr, GRB_LESS_EQUAL, mymastermodel.loss_vars[loss_index]);
    mymastermodel.model.update();
}

Eigen::VectorXd oaf::get_subgradient(const FirstStageResult &fs_result, const RecResult &rec_result, int num_basis,
                                     int num_predict)
{

    //proj_adj_costs is the Pi(B)^tr (c-T^tr mu-star)
    fs_result.transpose();
    
    Eigen::VectorXd proj_adj_costs = Eigen::VectorXd::Zero(num_basis);
    int i = 0;
    for (int j : fs_result.first_included)
    {
        proj_adj_costs(i) = rec_result.adj_costs(j);
        i++;
    }
    return fs_result.decomp.solve(proj_adj_costs).tail(num_predict);
}

Eigen::VectorXd oaf::GRBFirstSolver::get_basis_costs(const FirstStageResult &initial_solve)
{
    Eigen::VectorXd basis_costs = Eigen::VectorXd::Zero(basis_size);
    int basis_cost_index = 0;
    for (int varindex : initial_solve.first_included)
    {
        basis_costs(basis_cost_index) = mymodel.first_costs(varindex);
        basis_cost_index++;
    }
    for (int varindex : initial_solve.rec_included)
    {
        basis_costs(basis_cost_index) = mymodel.rec_costs(varindex);
        basis_cost_index++;
    }
    return basis_costs;
}

Eigen::VectorXd oaf::GRBFirstSolver::get_unit_basis_costs(int var_index, const FirstStageResult &initial_solve)
{
    Eigen::VectorXd basis_costs = Eigen::VectorXd::Zero(basis_size);
    int basis_cost_index = 0;
    for (int basisvar_index : initial_solve.first_included)
    {
        if (basisvar_index == var_index)
        {
            basis_costs(basis_cost_index) = 1;
        }
        basis_cost_index++;
    }
    return basis_costs;
}

bool oaf::GRBFirstSolver::is_already_solved(int j, const std::map<int, Eigen::VectorXd> &first_tableau_columns,
                                            const std::map<int, Eigen::VectorXd> &rec_tableau_columns,
                                            const std::set<int> &first_remaining, const std::set<int> &rec_remaining,
                                            const FirstStageResult &initial_solve, const std::set<int> &first_basis,
                                            const std::set<int> &rec_basis)
{
    Eigen::VectorXd basis_costs = get_unit_basis_costs(j, initial_solve);

    for (int i : first_remaining)
    {
        if (first_basis.count(i) == 0)
        {
            double reduced_cost = -basis_costs.dot(first_tableau_columns.at(i));
            if (i == j)
            {
                reduced_cost += 1;
            }
            if (reduced_cost < 0)
            {
                return false;
            }
        }
    }
    for (int i : rec_remaining)
    {
        if (rec_basis.count(i) == 0)
        {

            double reduced_cost = -basis_costs.dot(rec_tableau_columns.at(i));
            if (reduced_cost < 0)
            {
                return false;
            }
        }
    }
    return true;
}

void oaf::GRBFirstSolver::zero_obj()
{

    for (GRBVar &myvar : mymodel.first_vars)
    {
        myvar.set(GRB_DoubleAttr_Obj, 0.0);
    }
    for (GRBVar &myvar : mymodel.rec_vars)
    {
        myvar.set(GRB_DoubleAttr_Obj, 0.0);
    }
}

void oaf::GRBFirstSolver::reset_model()
{
    int i = 0;
    for (GRBVar &myvar : mymodel.first_vars)
    {
        myvar.set(GRB_DoubleAttr_UB, GRB_INFINITY);
        myvar.set(GRB_DoubleAttr_Obj, mymodel.first_costs(i));
        i++;
    }
    i = 0;
    for (GRBVar &myvar : mymodel.rec_vars)
    {
        myvar.set(GRB_DoubleAttr_UB, GRB_INFINITY);
        myvar.set(GRB_DoubleAttr_Obj, mymodel.rec_costs(i));
        i++;
    }
}

oaf::FirstStageResult oaf::GRBFirstSolver::get_lexmin(const Eigen::VectorXd &prediction)
{
    FirstStageResult initialsolve = solve(prediction, false);
    if (!initialsolve.feasible)
    {
        return initialsolve;
    }
    int num_first = mymodel.first_vars.size();
    std::set<int> first_remaining{};
    for (int i = 0; i < num_first; i++)
    {
        first_remaining.insert(i);
    }

    int num_rec = mymodel.rec_vars.size();
    std::set<int> rec_remaining{};
    for (int i = 0; i < num_rec; i++)
    {
        rec_remaining.insert(i);
    }

    std::set<int> first_basis{};
    for (int varindex : initialsolve.first_included)
    {
        first_basis.insert(varindex);
    }

    std::set<int> rec_basis{};
    for (int varindex : initialsolve.rec_included)
    {
        rec_basis.insert(varindex);
    }


    Eigen::VectorXd basis_costs = get_basis_costs(initialsolve);
    std::map<int, Eigen::VectorXd> first_tableau_columns{};
    std::set<int> iter_indices = first_remaining;
    for (int i : iter_indices)
    {
        if (first_basis.count(i) == 0)
        {
            first_tableau_columns[i] = initialsolve.decomp.solve(first_matrix.col(i));
            double reduced_cost = mymodel.first_costs(i) - basis_costs.dot(first_tableau_columns[i]);
            if (reduced_cost > basis_tol)
            {
                first_remaining.erase(i);
                mymodel.first_vars[i].set(GRB_DoubleAttr_UB, 0);
            }
        }
    }

    std::map<int, Eigen::VectorXd> rec_tableau_columns{};
    iter_indices = rec_remaining;
    for (int i : iter_indices)
    {
        if (rec_basis.count(i) == 0)
        {
            rec_tableau_columns[i] = initialsolve.decomp.solve(rec_matrix.col(i));
            double reduced_cost = mymodel.rec_costs(i) - basis_costs.dot(rec_tableau_columns[i]);

            if (reduced_cost > basis_tol)
            {
                rec_remaining.erase(i);
                mymodel.rec_vars[i].set(GRB_DoubleAttr_UB, 0);
            }
        }
        
    }

    zero_obj();
    for (int j = 0; j < num_first; j++)
    {   
            mymodel.first_vars[j].set(GRB_DoubleAttr_Obj, 1.0);
            if(!is_already_solved(j, first_tableau_columns, rec_tableau_columns, first_remaining, rec_remaining, initialsolve,
                               first_basis, rec_basis)){
                initialsolve = solve(false);
            }
            basis_costs = get_unit_basis_costs(j, initialsolve);
            std::set<int> newfirstbasis{};
        
            for (int varindex : initialsolve.first_included)
            {
                newfirstbasis.insert(varindex);
            }
            std::set<int> newrecbasis{};
            for (int varindex : initialsolve.rec_included)
            {
                newrecbasis.insert(varindex);
            }

            iter_indices = first_remaining;
            for (int i : iter_indices)
            {
                if (newfirstbasis.count(i) == 0)
                {
                    first_tableau_columns[i] = initialsolve.decomp.solve(first_matrix.col(i));
                    double reduced_cost = -basis_costs.dot(first_tableau_columns[i]);
                    if (i == j)
                    {
                        reduced_cost += 1;
                    }
                    if (reduced_cost > basis_tol)
                    {
                        first_remaining.erase(i);
                        mymodel.first_vars[i].set(GRB_DoubleAttr_UB, 0);
                    }
                }
            }
            iter_indices = rec_remaining;
            for (int i : iter_indices)
            {
                if (newrecbasis.count(i) == 0)
                {
                    rec_tableau_columns[i] = initialsolve.decomp.solve(rec_matrix.col(i));
                    double reduced_cost = -basis_costs.dot(rec_tableau_columns[i]);

                    if (reduced_cost > basis_tol)
                    {
                        rec_remaining.erase(i);
                        mymodel.rec_vars[i].set(GRB_DoubleAttr_UB, 0);
                    }
                }
            }
            mymodel.first_vars[j].set(GRB_DoubleAttr_Obj, 0.0);
        }

    reset_model();
    return initialsolve;
}

oaf::FirstStageResult oaf::GRBFirstSolver::solve(const Eigen::VectorXd &prediction, bool transpose)
{
    update_rhs(prediction);
    return solve(transpose);
}
oaf::FirstStageResult oaf::GRBFirstSolver::solve(bool transpose)
{
    mymodel.model->optimize();
    if (mymodel.model->get(GRB_IntAttr_Status) == GRB_INFEASIBLE ||
        mymodel.model->get(GRB_IntAttr_Status) == GRB_INF_OR_UNBD)
    {
        return read_infeasible();
    }
    else if (mymodel.model->get(GRB_IntAttr_Status) != GRB_OPTIMAL)
    {
        throw "Model is neither infeasible or optimal. Not sure how that happened.";
    }

    int n_basic_found = 0;

    std::vector<int> var_included{};
    Eigen::VectorXd lexmin_sol = Eigen::VectorXd::Zero(mymodel.first_vars.size());
    Eigen::MatrixXd matrix{basis_size, basis_size};
    int i = 0;

    for (const GRBVar &myvar : mymodel.first_vars)
    {
        if (myvar.get(GRB_IntAttr_VBasis) == 0 || myvar.get(GRB_IntAttr_VBasis) == -3)
        {
            var_included.push_back(i);
            matrix.col(n_basic_found) = first_matrix.col(i);
            lexmin_sol(i) = myvar.get(GRB_DoubleAttr_X);
            n_basic_found++;
        }
        i++;
    }

    std::vector<int> rec_included{};
    i = 0;
    for (const GRBVar &myvar : mymodel.rec_vars)
    {
        if (myvar.get(GRB_IntAttr_VBasis) == 0 || myvar.get(GRB_IntAttr_VBasis) == -3)
        {
            rec_included.push_back(i);
            matrix.col(n_basic_found) = rec_matrix.col(i);
            n_basic_found++;
        }
        i++;
    }
    return FirstStageResult{mymodel.first_costs.dot(lexmin_sol),
                            std::move(lexmin_sol), std::move(matrix),
                            std::move(var_included), std::move(rec_included), transpose};
}

oaf::FirstStageResult oaf::GRBFirstSolver::read_infeasible()
{
    Eigen::VectorXd known_dual_sol(mymodel.known_constr.size());
    Eigen::VectorXd predict_dual_sol(mymodel.predict_constr.size());
    int i = 0;
    for (const GRBConstr &constr : mymodel.known_constr)
    {
        known_dual_sol(i) = -constr.get(GRB_DoubleAttr_FarkasDual);
        i++;
    }

    i = 0;
    for (const GRBConstr &constr : mymodel.predict_constr)
    {
        predict_dual_sol(i) = -constr.get(GRB_DoubleAttr_FarkasDual);
        i++;
    }
    return FirstStageResult{std::move(known_dual_sol), std::move(predict_dual_sol)};
}

double oaf::GRBFirstSolver::optimal_value(const Eigen::VectorXd &prediction)
{
    update_rhs(prediction);
    mymodel.model->optimize();
    return mymodel.model->get(GRB_DoubleAttr_ObjVal);
}

void oaf::GRBFirstSolver::update_rhs(const Eigen::VectorXd &new_rec)
{
    int i = 0;
    for (GRBConstr &constr : mymodel.predict_constr)
    {
        constr.set(GRB_DoubleAttr_RHS, new_rec(i));
        i++;
    }
    mymodel.model->update();
    return;
}

int oaf::GRBFirstSolver::get_basis_size()
{
    return basis_size;
}

int oaf::GRBFirstSolver::get_n_predict()
{
    return n_predict;
}

Eigen::VectorXd oaf::GRBFirstSolver::get_known_rhs()
{
    return known_rhs;
}

double oaf::GRBFirstSolver::get_fs_costs(const Eigen::VectorXd & fs_sol)
{
    return fs_sol.dot(mymodel.first_costs);
}

Eigen::MatrixXd oaf::GRBFirstSolver::solve_all(const Eigen::MatrixXd & predictions){
    
    int num_rows = predictions.rows();
    Eigen::MatrixXd solutions(num_rows, mymodel.first_vars.size());
    for(int i=0; i < num_rows; i++){
        FirstStageResult myresult = solve(predictions.row(i), false);
        solutions.row(i) = myresult.lexmin_sol;
    }
    return solutions;
}
