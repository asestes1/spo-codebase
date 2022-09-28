#include <Eigen/Dense>
#include <oafitting/oafit.hpp>
#include <oafitting/vendor.hpp>

oaf::GRBRecDualStruct VendorInitializer::init_rec_model(GRBEnv &env, const Eigen::VectorXd &sol, const Eigen::VectorXd &rhs) const
{
    std::cout << "SOL: " << sol << std::endl;
    std::cout << "RHS: " << rhs << std::endl;
    GRBModel *mymodel = new GRBModel(env);
    mymodel->set(GRB_IntAttr_ModelSense, GRB_MAXIMIZE);
    mymodel->set(GRB_IntParam_OutputFlag, 0);

    std::vector<GRBVar> known_dual_vars{};
    for (int i = 0; i < num_items; i++)
    {
        known_dual_vars.push_back(mymodel->addVar(-GRB_INFINITY, 0.0, sol(i), GRB_CONTINUOUS, ""));
    }

    std::vector<GRBVar> predict_dual_vars{};
    for (int i = 0; i < num_items; i++)
    {
        predict_dual_vars.push_back(mymodel->addVar(-GRB_INFINITY, 0.0, rhs(i), GRB_CONTINUOUS, ""));
    }

    for (int i = 0; i < num_items; i++)
    {
        mymodel->addConstr(known_dual_vars[i] + predict_dual_vars[i], GRB_LESS_EQUAL, -sell_costs(i));
    }

    mymodel->update();
    return oaf::GRBRecDualStruct{mymodel, known_dual_vars, predict_dual_vars};
}

void VendorInitializer::update_rec_model(oaf::GRBRecDualStruct &mymodel, const Eigen::VectorXd &sol,
                                         const Eigen::VectorXd &rhs) const
{

    int i = 0;
    for (GRBVar myvar : mymodel.known_dual_vars)
    {
        myvar.set(GRB_DoubleAttr_Obj, sol(i));
        i++;
    }

    i = 0;
    for (GRBVar myvar : mymodel.predict_dual_vars)
    {
        myvar.set(GRB_DoubleAttr_Obj, rhs(i));
        i++;
    }

    return;
}

Eigen::VectorXd VendorInitializer::get_adj_costs(const Eigen::VectorXd &known_dual_sol, const Eigen::VectorXd &predict_dual_sol) const
{
    Eigen::VectorXd adj_costs = Eigen::VectorXd::Zero(buy_costs.size() + 2);
    adj_costs.head(buy_costs.size()) = buy_costs + known_dual_sol;
    return adj_costs;
}

oaf::GRBFirstStruct VendorInitializer::init_model(GRBEnv &env, const Eigen::VectorXd &rhs) const
{
    GRBModel *mymodel = new GRBModel(env);
    mymodel->set(GRB_IntParam_OutputFlag, 0);

    std::vector<GRBVar> first_vars{};
    std::vector<GRBVar> rec_vars{};

    std::vector<GRBVar> buy_vars{};
    for (int i = 0; i < num_items; i++)
    {
        GRBVar newvar = mymodel->addVar(0.0, GRB_INFINITY, buy_costs(i) - salvage_values(i), GRB_CONTINUOUS, "");
        buy_vars.push_back(newvar);
        first_vars.push_back(newvar);
    }
    GRBVar budget_slack_var = mymodel->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    first_vars.push_back(budget_slack_var);

    GRBVar vol_slack_var = mymodel->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    first_vars.push_back(vol_slack_var);

    std::vector<GRBVar> sell_vars{};
    for (int i = 0; i < num_items; i++)
    {
        GRBVar newvar = mymodel->addVar(0.0, GRB_INFINITY, -(sell_costs(i)-salvage_values(i)), GRB_CONTINUOUS, "");
        sell_vars.push_back(newvar);
        rec_vars.push_back(newvar);
    }

    std::vector<GRBVar> buy_slack_vars{};
    for (int i = 0; i < num_items; i++)
    {
        GRBVar newvar = mymodel->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "");
        buy_slack_vars.push_back(newvar);
        rec_vars.push_back(newvar);
    }

    std::vector<GRBVar> demand_slack_vars{};
    for (int i = 0; i < num_items; i++)
    {
        GRBVar newvar = mymodel->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "");
        demand_slack_vars.push_back(newvar);
        rec_vars.push_back(newvar);
    }

    std::vector<GRBConstr> known_constr{};
    Eigen::VectorXd known_rhs = Eigen::VectorXd::Zero(num_items + 2);

    GRBLinExpr myexpr{};
    myexpr.addTerms(buy_costs.data(), &buy_vars[0], num_items);
    myexpr += budget_slack_var;
    known_constr.push_back(mymodel->addConstr(myexpr, GRB_EQUAL, budget));
    known_rhs(0) = budget;

    GRBLinExpr volexpr{};
    Eigen::VectorXd volcoeff = Eigen::VectorXd::Ones(num_items);
    volexpr.addTerms(volcoeff.data(), &buy_vars[0], num_items);
    volexpr += vol_slack_var;
    known_constr.push_back(mymodel->addConstr(volexpr, GRB_EQUAL, volume));
    known_rhs(1) = volume;

    
    for (int i = 0; i < num_items; i++)
    {
        known_constr.push_back(mymodel->addConstr(sell_vars[i] + buy_slack_vars[i], GRB_EQUAL, buy_vars[i]));
    }

    std::vector<GRBConstr> predict_constrs;
    for (int i = 0; i < num_items; i++)
    {
        predict_constrs.push_back(mymodel->addConstr(sell_vars[i] + demand_slack_vars[i], GRB_EQUAL, rhs(i)));
    }
    mymodel->update();

    Eigen::VectorXd first_costs = Eigen::VectorXd::Zero(num_items + 2);
    first_costs.head(num_items) = buy_costs;

    Eigen::VectorXd rec_costs = Eigen::VectorXd::Zero(3 * num_items);
    rec_costs.head(num_items) = -sell_costs;
    return oaf::GRBFirstStruct{mymodel, std::move(first_costs), std::move(rec_costs),
                          std::move(first_vars), std::move(known_constr), std::move(known_rhs),
                          std::move(rec_vars), std::move(predict_constrs)};
}

Eigen::MatrixXd VendorInitializer::make_first_matrix() const
{
    Eigen::MatrixXd mymatrix = Eigen::MatrixXd::Zero(2 * num_items + 2, num_items + 2);

    //Add budget constraint.
    for (int i = 0; i < num_items; i++)
    {
        mymatrix(0, i) = buy_costs(i);
    }
    mymatrix(0, num_items) = 1;

    //Add volume constraint.
    for (int i = 0; i < num_items; i++)
    {
        mymatrix(1, i) = 1;
    }
    mymatrix(1, num_items+1) = 1;

    //Add constraints that sales are less than bought inventory
    int constr_index = 2;
    for (int i = 0; i < num_items; i++)
    {
        mymatrix(constr_index, i) = -1;
        constr_index++;
    }

    return mymatrix;
}

Eigen::MatrixXd VendorInitializer::make_rec_matrix() const
{
    Eigen::MatrixXd mymatrix = Eigen::MatrixXd::Zero(2 * num_items + 2, 3 * num_items);

    //Add constraints that sales are less than bought inventory
    int constr_index = 2;
    int sell_slack_index = num_items;
    for (int i = 0; i < num_items; i++)
    {
        mymatrix(constr_index, i) = 1;
        mymatrix(constr_index, sell_slack_index) = 1;
        sell_slack_index++;
        constr_index++;
    }

    //Add constraints that sales are less than demand.
    sell_slack_index = 2 * num_items;
    for (int i = 0; i < num_items; i++)
    {
        mymatrix(constr_index, i) = 1;
        mymatrix(constr_index, sell_slack_index) = 1;
        sell_slack_index++;
        constr_index++;
    }

    return mymatrix;
}

Eigen::VectorXd VendorInitializer::fixed_rhs() const
{
    Eigen::VectorXd myvector = Eigen::VectorXd::Zero(num_items + 2);
    myvector(0) = budget;
    myvector(1) = volume;
    return myvector;
}

Eigen::VectorXd VendorInitializer::default_pred() const
{
    return Eigen::VectorXd::Zero(num_items);
}



void VendorInitializer::add_scenario(GRBModel &mymodel,
                                     const std::vector<GRBVar> &buyvars,
                                     std::vector<std::vector<GRBVar>> &sellvars,
                                     std::vector<std::vector<GRBConstr>> &demand_constrs,
                                     std::vector<std::vector<GRBConstr>> &buysell_constr,
                                     const oaf::Scenario &scenario) const
{
    int scenid = sellvars.size();
    std::vector<GRBVar> scen_sellvars{};
    for (int i = 0; i < num_items; i++)
    {
        GRBVar newvar = mymodel.addVar(0.0, GRB_INFINITY, scenario.probability * (-(sell_costs(i)-salvage_values(i))),
                                       GRB_CONTINUOUS, "SELL" + std::to_string(scenid) + "," + std::to_string(i));
        scen_sellvars.push_back(newvar);
    }
   sellvars.push_back(scen_sellvars);
   mymodel.update();

    std::vector<GRBConstr> scen_demandconstr{};
    for (int i=0; i < num_items; i++){
        scen_demandconstr.push_back(mymodel.addConstr(scen_sellvars[i], GRB_LESS_EQUAL, scenario.parameters(i)));
    }
    demand_constrs.push_back(scen_demandconstr);

    std::vector<GRBConstr> scen_buysellconstr{};
    for (int i=0; i < num_items; i++){
        scen_buysellconstr.push_back(mymodel.addConstr(scen_sellvars[i], GRB_LESS_EQUAL, buyvars[i]));
    }
    buysell_constr.push_back(scen_buysellconstr);
    mymodel.update();
}

void VendorInitializer::init_sp(GRBModel &mymodel,
                                std::vector<GRBVar> &buyvars,
                                GRBVar& budgetslackvar,
                                GRBVar& volslackvar,
                                std::vector<std::vector<GRBVar>> &sellvars,
                                std::vector<std::vector<GRBConstr>> &demand_constrs,
                                std::vector<std::vector<GRBConstr>> &buysell_constrs,
                                const std::vector<oaf::Scenario> &instance) const
{

    for (int i = 0; i < num_items; i++)
    {
        GRBVar newvar = mymodel.addVar(0.0, GRB_INFINITY, buy_costs(i) - salvage_values(i), GRB_CONTINUOUS, "P" + std::to_string(i));
        buyvars.push_back(newvar);
    }
    budgetslackvar = mymodel.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    volslackvar = mymodel.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    mymodel.update();

    GRBLinExpr myexpr{};
    myexpr.addTerms(buy_costs.data(), &buyvars[0], num_items);
    myexpr += budgetslackvar;
    mymodel.addConstr(myexpr, GRB_EQUAL, budget);
    

    GRBLinExpr volexpr{};
    Eigen::VectorXd volcoeff = Eigen::VectorXd::Ones(num_items);
    volexpr.addTerms(volcoeff.data(), &buyvars[0], num_items);
    volexpr += volslackvar;
    mymodel.addConstr(volexpr, GRB_EQUAL, volume);
    
    
    mymodel.update();
    for (const oaf::Scenario &scenario : instance)
    {
        add_scenario(mymodel, buyvars, sellvars, demand_constrs, buysell_constrs, scenario);
    }
    mymodel.update();
    return;
}

Eigen::VectorXd VendorInitializer::read_sp_sol(const std::vector<GRBVar> &buyvars, const GRBVar& budgetslackvar, 
const GRBVar& volslackvar) const
{
    int solsize = num_items+2;
    Eigen::VectorXd mysol(solsize);
    for (int i = 0; i < num_items; i++)
    {
        mysol(i) = buyvars[i].get(GRB_DoubleAttr_X);
    }
    mysol(num_items) = budgetslackvar.get(GRB_DoubleAttr_X);
    mysol(num_items+1) = volslackvar.get(GRB_DoubleAttr_X);
    return mysol;
}

void VendorInitializer::pop_scenario(GRBModel &mymodel,
                                         std::vector<std::vector<GRBVar>> &sellvars,
                                         std::vector<std::vector<GRBConstr>> &demand_constrs,
                                         std::vector<std::vector<GRBConstr>> &buysell_constrs) const
{
    for (GRBVar &myvar : sellvars.back())
    {
        mymodel.remove(myvar);
    }
    sellvars.pop_back();



    for (GRBConstr &myconstr : demand_constrs.back())
    {
        mymodel.remove(myconstr);
    }
    demand_constrs.pop_back();

    for (GRBConstr &myconstr : buysell_constrs.back())
    {
        mymodel.remove(myconstr);
    }
    buysell_constrs.pop_back();
    mymodel.update();
}

void VendorInitializer::update_sp(GRBModel &mymodel,
                                    const std::vector<GRBVar> &buyvars,
                                         std::vector<std::vector<GRBVar>> &sellvars,
                                         std::vector<std::vector<GRBConstr>> &demand_constrs,
                                         std::vector<std::vector<GRBConstr>> &buysell_constrs,
                                        const std::vector<oaf::Scenario> &instance) const
{
    int old_num_scen = sellvars.size();
    int new_num_scen = instance.size();

    int num_revise;

    if (new_num_scen > old_num_scen)
    {
        num_revise = old_num_scen;
        for (int scenid = old_num_scen; scenid < new_num_scen; scenid++)
        {
            add_scenario(mymodel, buyvars, sellvars, demand_constrs, buysell_constrs, instance.at(scenid));
        }
    }
    else if (new_num_scen < old_num_scen)
    {
        num_revise = new_num_scen;
        for (int scendid = new_num_scen; scendid < old_num_scen; scendid++)
        {
            pop_scenario(mymodel, sellvars, demand_constrs, buysell_constrs);
        }
    }else{
        num_revise = new_num_scen;
    }

    for (int scenid = 0; scenid < num_revise; scenid++)
    {
        double myprob = instance[scenid].probability;
        for (int i = 0; i < num_items; i++)
        {
            sellvars[scenid][i].set(GRB_DoubleAttr_Obj, myprob * (-(sell_costs(i)-salvage_values(i))));
            demand_constrs[scenid][i].set(GRB_DoubleAttr_RHS, instance[scenid].parameters(i));
        }
    }
    mymodel.update();
    return;
}


Eigen::MatrixXd VendorInitializer::solve_instances(const std::vector<std::vector<oaf::Scenario>> &instances) const
{
    GRBEnv myenv{};
    GRBModel mymodel = GRBModel(myenv);
    mymodel.set(GRB_IntParam_OutputFlag, 0);

    Eigen::MatrixXd solutions(instances.size(), num_items+2);


    const std::vector<oaf::Scenario> &first_instance = instances.at(0);
    std::vector<GRBVar> buyvars{};
    GRBVar budgetslackvar;
    GRBVar volslackvar;
    std::vector<std::vector<GRBVar>> sellvars{};
    std::vector<std::vector<GRBConstr>> demand_constrs{};
    std::vector<std::vector<GRBConstr>> buysell_constrs{};
    init_sp(mymodel, buyvars, budgetslackvar, volslackvar, sellvars, demand_constrs, buysell_constrs, first_instance);

    mymodel.optimize();

    solutions.row(0) = read_sp_sol(buyvars, budgetslackvar, volslackvar);

    
    int num_instances = instances.size();
    for (int i = 1; i < num_instances; i++)
    {
        update_sp(mymodel, buyvars, sellvars, demand_constrs, buysell_constrs, instances[i]);
        mymodel.optimize();
        solutions.row(i) = read_sp_sol(buyvars, budgetslackvar, volslackvar);

    }
    return solutions;
}
// Eigen::VectorXd VendorInitializer::get_dual_rhs(Eigen::VectorXd ray_known, Eigen::VectorXd ray_predict)
// {
//     double budget_dual = ray_known(0);
//     Eigen::VectorXd buy_dual = ray_known.tail(num_items);
//     Eigen::VectorXd buy_rhs = Eigen::VectorXd::Constant(num_items, ray_known(0)) - buy_dual;
//     Eigen::VectorXd sell_rhs = buy_dual + ray_predict;
//     Eigen::VectorXd budget_slack_rhs = Eigen::VectorXd::Constant(1, budget_dual);

//     Eigen::VectorXd dual_rhs(4 * num_items + 1);
//     dual_rhs << buy_rhs, budget_slack_rhs, sell_rhs, buy_dual, ray_predict;
//     return dual_rhs;
// }

//------------------------------------


