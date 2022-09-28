#include <Eigen/Dense>
#include <oafitting/oafit.hpp>
#include <oafitting/transport.hpp>

Eigen::VectorXd TransportationInitializer::get_adj_costs(const Eigen::VectorXd &known_dual_sol, const Eigen::VectorXd &predict_dual_sol) const
{
    Eigen::VectorXd adj_costs = production_costs + known_dual_sol;
    return adj_costs;
}

oaf::GRBFirstStruct TransportationInitializer::init_model(GRBEnv &env, const Eigen::VectorXd &rhs) const
{
    GRBModel *mymodel = new GRBModel(env);
    mymodel->set(GRB_IntParam_OutputFlag, 0);

    Eigen::VectorXd first_costs = production_costs;
    std::vector<GRBVar> first_vars{};
    std::vector<GRBVar> produce_vars{};
    for (int i = 0; i < n_source; i++)
    {
        GRBVar newvar = mymodel->addVar(0.0, GRB_INFINITY, production_costs(i), GRB_CONTINUOUS, "P" + std::to_string(i));
        produce_vars.push_back(newvar);
        first_vars.push_back(newvar);
    }

    Eigen::VectorXd rec_costs = Eigen::VectorXd::Zero(n_sink + n_source + n_sink * n_source);
    int cost_vec_index = 0;
    std::vector<GRBVar> rec_vars{};
    std::vector<GRBVar> scrap_vars{};
    for (int i = 0; i < n_source; i++)
    {
        GRBVar newvar = mymodel->addVar(0.0, GRB_INFINITY, scrap_costs(i), GRB_CONTINUOUS, "SCR" + std::to_string(i));
        scrap_vars.push_back(newvar);
        rec_vars.push_back(newvar);
        rec_costs(cost_vec_index) = scrap_costs(i);
        cost_vec_index++;
    }

    std::vector<GRBVar> unmet_vars{};
    for (int i = 0; i < n_sink; i++)
    {
        GRBVar newvar = mymodel->addVar(0.0, GRB_INFINITY, unmet_costs(i), GRB_CONTINUOUS, "UM" + std::to_string(i));
        unmet_vars.push_back(newvar);
        rec_vars.push_back(newvar);
        rec_costs(cost_vec_index) = unmet_costs(i);
        cost_vec_index++;
    }

    std::vector<std::vector<GRBVar>> transport_vars{};
    for (int i = 0; i < n_source; i++)
    {
        std::vector<GRBVar> myvector{};
        for (int j = 0; j < n_sink; j++)
        {
            GRBVar newvar = mymodel->addVar(0.0, GRB_INFINITY, transport_costs(i, j), GRB_CONTINUOUS,
                                            "T" + std::to_string(i) + "," + std::to_string(j));
            myvector.push_back(newvar);
            rec_vars.push_back(newvar);
            rec_costs(cost_vec_index) = transport_costs(i, j);
            cost_vec_index++;
        }
        transport_vars.push_back(myvector);
    }
    mymodel->update();

    std::vector<GRBConstr> known_constrs{};
    Eigen::VectorXd known_rhs = Eigen::VectorXd::Zero(n_source);

    for (int i = 0; i < n_source; i++)
    {
        GRBLinExpr myexpr{};
        myexpr -= produce_vars[i];
        myexpr += scrap_vars[i];
        for (int j = 0; j < n_sink; j++)
        {
            myexpr += transport_vars[i][j];
        }
        known_constrs.push_back(mymodel->addConstr(myexpr, GRB_EQUAL, 0.0));
    }

    std::vector<GRBConstr> predict_constrs{};
    for (int i = 0; i < n_sink; i++)
    {
        GRBLinExpr myexpr{};
        myexpr += unmet_vars[i];
        for (int j = 0; j < n_source; j++)
        {
            myexpr += transport_vars[j][i];
        }
        predict_constrs.push_back(mymodel->addConstr(myexpr, GRB_EQUAL, rhs(i)));
    }
    mymodel->update();

    return oaf::GRBFirstStruct{mymodel, std::move(first_costs), std::move(rec_costs),
                               std::move(first_vars), std::move(known_constrs), std::move(known_rhs),
                               std::move(rec_vars), std::move(predict_constrs)};
}

Eigen::MatrixXd TransportationInitializer::make_first_matrix() const
{
    Eigen::MatrixXd mymatrix = Eigen::MatrixXd::Zero(n_sink + n_source, n_source);

    //Add budget constraint.
    for (int i = 0; i < n_source; i++)
    {
        mymatrix(i, i) = -1;
    }
    return mymatrix;
}

Eigen::MatrixXd TransportationInitializer::make_rec_matrix() const
{
    Eigen::MatrixXd mymatrix = Eigen::MatrixXd::Zero(n_sink + n_source, n_source + n_sink + n_source * n_sink);

    int transport_index = n_source + n_sink;
    for (int i = 0; i < n_source; i++)
    {
        mymatrix(i, i) = 1;
        for (int j = 0; j < n_sink; j++)
        {
            mymatrix(i, transport_index) = 1;
            transport_index++;
        }
    }

    int unmet_index = n_source;
    for (int i = 0; i < n_sink; i++)
    {
        mymatrix(unmet_index, unmet_index) = 1;
        unmet_index++;
    }

    int var_index = n_source + n_sink;
    for (int i = 0; i < n_source; i++)
    {
        int constr_index = n_source;
        for (int j = 0; j < n_sink; j++)
        {
            mymatrix(constr_index, var_index) = 1;
            var_index++;
            constr_index++;
        }
    }
    return mymatrix;
}

Eigen::VectorXd TransportationInitializer::fixed_rhs() const
{
    Eigen::VectorXd myvector = Eigen::VectorXd::Zero(n_source);
    return myvector;
}

oaf::GRBRecDualStruct TransportationInitializer::init_rec_model(GRBEnv &env, const Eigen::VectorXd &sol, const Eigen::VectorXd &rhs) const
{
    GRBModel *mymodel = new GRBModel(env);
    mymodel->set(GRB_IntAttr_ModelSense, GRB_MAXIMIZE);
    mymodel->set(GRB_IntParam_OutputFlag, 0);

    std::vector<GRBVar> known_dual_vars{};
    for (int i = 0; i < n_source; i++)
    {
        known_dual_vars.push_back(mymodel->addVar(-GRB_INFINITY, scrap_costs(i), sol(i), GRB_CONTINUOUS, ""));
    }

    std::vector<GRBVar> predict_dual_vars{};
    for (int i = 0; i < n_sink; i++)
    {
        predict_dual_vars.push_back(mymodel->addVar(-GRB_INFINITY, unmet_costs(i), rhs(i), GRB_CONTINUOUS, ""));
    }
    mymodel->update();

    for (int i = 0; i < n_source; i++)
    {
        for (int j = 0; j < n_sink; j++)
        {
            mymodel->addConstr(known_dual_vars[i] + predict_dual_vars[j], GRB_LESS_EQUAL, transport_costs(i, j));
        }
    }

    mymodel->update();
    return oaf::GRBRecDualStruct{mymodel, known_dual_vars, predict_dual_vars};
}

void TransportationInitializer::update_rec_model(oaf::GRBRecDualStruct &mymodel, const Eigen::VectorXd &sol,
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

Eigen::VectorXd TransportationInitializer::default_pred() const
{
    return Eigen::VectorXd::Zero(n_sink);
}

void TransportationInitializer::add_scenario(GRBModel &mymodel,
                                             const std::vector<GRBVar> &prodvars,
                                             std::vector<std::vector<GRBVar>> &scrapvars,
                                             std::vector<std::vector<GRBVar>> &unmetvars,
                                             std::vector<std::vector<std::vector<GRBVar>>> &transportvars,
                                             std::vector<std::vector<GRBConstr>> &src_constr,
                                             std::vector<std::vector<GRBConstr>> &dest_constr,
                                             const oaf::Scenario &scenario) const
{
    int scenid = scrapvars.size();
    std::vector<GRBVar> scen_scrapvars{};
    for (int i = 0; i < n_source; i++)
    {
        GRBVar newvar = mymodel.addVar(0.0, GRB_INFINITY, scenario.probability * scrap_costs(i),
                                       GRB_CONTINUOUS, "SCR" + std::to_string(scenid) + "," + std::to_string(i));
        scen_scrapvars.push_back(newvar);
    }
    scrapvars.push_back(scen_scrapvars);

    std::vector<GRBVar> scen_unmetvars{};
    for (int i = 0; i < n_sink; i++)
    {
        GRBVar newvar = mymodel.addVar(0.0, GRB_INFINITY, scenario.probability * unmet_costs(i), GRB_CONTINUOUS,
                                       "UM" + std::to_string(scenid) + "," + std::to_string(i));
        scen_unmetvars.push_back(newvar);
    }
    unmetvars.push_back(scen_unmetvars);

    std::vector<std::vector<GRBVar>> scen_transportvars{};
    for (int i = 0; i < n_source; i++)
    {
        std::vector<GRBVar> src_transportvars{};
        for (int j = 0; j < n_sink; j++)
        {
            GRBVar newvar = mymodel.addVar(0.0, GRB_INFINITY, scenario.probability * transport_costs(i, j), GRB_CONTINUOUS,
                                           "T" + std::to_string(scenid) + "," + std::to_string(i) + "," + std::to_string(j));
            src_transportvars.push_back(newvar);
        }
        scen_transportvars.push_back(src_transportvars);
    }
    transportvars.push_back(scen_transportvars);
    mymodel.update();

    std::vector<GRBConstr> scen_src_constr{};
    for (int i = 0; i < n_source; i++)
    {
        GRBLinExpr myexpr{};
        myexpr -= prodvars[i];
        myexpr += scrapvars[scenid][i];
        for (int j = 0; j < n_sink; j++)
        {
            myexpr += transportvars[scenid][i][j];
        }
        scen_src_constr.push_back(mymodel.addConstr(myexpr, GRB_EQUAL, 0.0));
    }
    src_constr.push_back(scen_src_constr);

    std::vector<GRBConstr> scen_dest_constr{};
    for (int i = 0; i < n_sink; i++)
    {
        GRBLinExpr myexpr{};
        myexpr += unmetvars[scenid][i];
        for (int j = 0; j < n_source; j++)
        {
            myexpr += transportvars[scenid][j][i];
        }

        scen_dest_constr.push_back(mymodel.addConstr(myexpr, GRB_EQUAL, scenario.parameters(i)));
    }
    dest_constr.push_back(scen_dest_constr);
    mymodel.update();
}

void TransportationInitializer::pop_scenario(GRBModel &mymodel,
                                             std::vector<std::vector<GRBVar>> &scrapvars,
                                             std::vector<std::vector<GRBVar>> &unmetvars,
                                             std::vector<std::vector<std::vector<GRBVar>>> &transportvars,
                                             std::vector<std::vector<GRBConstr>> &src_constr,
                                             std::vector<std::vector<GRBConstr>> &dest_constr) const
{
    for (GRBVar &myvar : scrapvars.back())
    {
        mymodel.remove(myvar);
    }
    scrapvars.pop_back();

    for (GRBVar &myvar : unmetvars.back())
    {
        mymodel.remove(myvar);
    }
    unmetvars.pop_back();

    for (std::vector<GRBVar> &myvars : transportvars.back())
    {

        for (GRBVar &myvar : myvars)
        {
            mymodel.remove(myvar);
        }
    }
    transportvars.pop_back();
    mymodel.update();

    for (GRBConstr &myconstr : src_constr.back())
    {
        mymodel.remove(myconstr);
    }
    src_constr.pop_back();

    for (GRBConstr &myconstr : dest_constr.back())
    {
        mymodel.remove(myconstr);
    }
    dest_constr.pop_back();
    mymodel.update();
}

void TransportationInitializer::init_sp(GRBModel &mymodel,
                                        std::vector<GRBVar> &prodvars,
                                        std::vector<std::vector<GRBVar>> &scrapvars,
                                        std::vector<std::vector<GRBVar>> &unmetvars,
                                        std::vector<std::vector<std::vector<GRBVar>>> &transportvars,
                                        std::vector<std::vector<GRBConstr>> &src_constr,
                                        std::vector<std::vector<GRBConstr>> &dest_constr,
                                        const std::vector<oaf::Scenario> &instance) const
{
    for (int i = 0; i < n_source; i++)
    {
        GRBVar newvar = mymodel.addVar(0.0, GRB_INFINITY, production_costs(i), GRB_CONTINUOUS, "P" + std::to_string(i));
        prodvars.push_back(newvar);
    }

    for (const oaf::Scenario &scenario : instance)
    {
        add_scenario(mymodel, prodvars, scrapvars, unmetvars, transportvars, src_constr, dest_constr, scenario);
    }
    mymodel.update();
    return;
}

void TransportationInitializer::update_sp(GRBModel &mymodel,
                                          const std::vector<GRBVar> &prodvars,
                                          std::vector<std::vector<GRBVar>> &scrapvars,
                                          std::vector<std::vector<GRBVar>> &unmetvars,
                                          std::vector<std::vector<std::vector<GRBVar>>> &transportvars,
                                          std::vector<std::vector<GRBConstr>> &src_constr,
                                          std::vector<std::vector<GRBConstr>> &dest_constr,
                                          const std::vector<oaf::Scenario> &instance) const
{
    int old_num_scen = scrapvars.size();
    int new_num_scen = instance.size();

    int num_revise;

    if (new_num_scen > old_num_scen)
    {
        num_revise = old_num_scen;
        for (int scenid = old_num_scen; scenid < new_num_scen; scenid++)
        {
            add_scenario(mymodel, prodvars, scrapvars, unmetvars, transportvars,
                         src_constr, dest_constr, instance.at(scenid));
        }
    }
    else if (new_num_scen < old_num_scen)
    {
        num_revise = new_num_scen;
        for (int scendid = new_num_scen; scendid < old_num_scen; scendid++)
        {
            pop_scenario(mymodel, scrapvars, unmetvars, transportvars, src_constr, dest_constr);
        }
    }else{
        num_revise = new_num_scen;
    }

    for (int scenid = 0; scenid < num_revise; scenid++)
    {
        double myprob = instance[scenid].probability;
        for (int i = 0; i < n_source; i++)
        {
            scrapvars[scenid][i].set(GRB_DoubleAttr_Obj, myprob * scrap_costs(i));
        }
        for (int i = 0; i < n_sink; i++)
        {
            unmetvars[scenid][i].set(GRB_DoubleAttr_Obj, myprob * unmet_costs(i));
            dest_constr[scenid][i].set(GRB_DoubleAttr_RHS, instance[scenid].parameters(i));
        }

        for (int i = 0; i < n_source; i++)
        {
            for (int j = 0; j < n_sink; j++)
            {
                transportvars[scenid][i][j].set(GRB_DoubleAttr_Obj, myprob * transport_costs(i, j));
            }
        }
    }
    mymodel.update();
    return;
}

Eigen::VectorXd TransportationInitializer::read_sp_sol(const std::vector<GRBVar> &prodvars) const
{
    int solsize = prodvars.size();
    Eigen::VectorXd mysol(solsize);
    for (int i = 0; i < solsize; i++)
    {
        mysol(i) = prodvars[i].get(GRB_DoubleAttr_X);
    }
    return mysol;
}

Eigen::MatrixXd TransportationInitializer::solve_instances(const std::vector<std::vector<oaf::Scenario>> &instances) const
{
    GRBEnv myenv{};
    GRBModel mymodel = GRBModel(myenv);
    mymodel.set(GRB_IntParam_OutputFlag, 0);

    Eigen::MatrixXd solutions(instances.size(), n_source);

    std::vector<GRBVar> producevars{};

    const std::vector<oaf::Scenario> &first_instance = instances.at(0);
    std::vector<std::vector<GRBVar>> scrapvars{};
    std::vector<std::vector<GRBVar>> unmetvars{};
    std::vector<std::vector<std::vector<GRBVar>>> transportvars{};
    std::vector<std::vector<GRBConstr>> src_constr{};
    std::vector<std::vector<GRBConstr>> dest_constr{};

    init_sp(mymodel, producevars, scrapvars, unmetvars, transportvars, src_constr, dest_constr,
            instances.at(0));

    mymodel.optimize();

    solutions.row(0) = read_sp_sol(producevars);

    int num_instances = instances.size();
    for (int i = 1; i < num_instances; i++)
    {
        update_sp(mymodel, producevars, scrapvars, unmetvars, transportvars,
                  src_constr, dest_constr, instances.at(i));
        mymodel.optimize();
        solutions.row(i) = read_sp_sol(producevars);

    }

    return solutions;
}