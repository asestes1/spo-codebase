#ifndef INSTANCES_H
#define INSTANCES_H
#include <Eigen/Dense>
#include <oafitting/oafit.hpp>

class VendorInitializer: public oaf::ModelInitializer
{
private:
    //TODO: change these to normal vectors or arrays.
    Eigen::VectorXd buy_costs;
    Eigen::VectorXd sell_costs;
    Eigen::VectorXd salvage_values;
    double budget;
    double volume;
    int num_items;

    void init_sp() const;

    void add_scenario(GRBModel &mymodel,
                                    const std::vector<GRBVar> &buyvars,
                                    std::vector<std::vector<GRBVar>> &sellvars,
                                    std::vector<std::vector<GRBConstr>> &demand_constrs,
                                    std::vector<std::vector<GRBConstr>> &buysell_constrs,
                                    const oaf::Scenario &scenario) const;

    void init_sp(GRBModel &mymodel,
                std::vector<GRBVar> &buyvars,
                GRBVar& budgetslackvar,
                GRBVar& volslackvar,
                std::vector<std::vector<GRBVar>> &sellvars,
                std::vector<std::vector<GRBConstr>> &demand_constrs,
                std::vector<std::vector<GRBConstr>> &buysell_constr,
                const std::vector<oaf::Scenario> &instance) const;

    void update_sp(GRBModel &mymodel,
                    const std::vector<GRBVar> &buyvars,
                    std::vector<std::vector<GRBVar>> &sellvars,
                    std::vector<std::vector<GRBConstr>> &demand_constrs,
                    std::vector<std::vector<GRBConstr>> &buysell_constrs,
                    const std::vector<oaf::Scenario> &instance) const;

    void pop_scenario(GRBModel &mymodel,
                                         std::vector<std::vector<GRBVar>> &sellvars,
                                         std::vector<std::vector<GRBConstr>> &demand_constrs,
                                         std::vector<std::vector<GRBConstr>> &buysell_constrs) const;

    Eigen::VectorXd read_sp_sol(const std::vector<GRBVar> &buyvars, const GRBVar& budgetslackvar, const GRBVar& volslackvar) const;

public:
    VendorInitializer(const Eigen::VectorXd &buy_costs,
                      const Eigen::VectorXd &sell_costs,
                      const Eigen::VectorXd &salvage_values,
                      double budget, double volume) : buy_costs(buy_costs),
                                                      sell_costs(sell_costs),
                                                      salvage_values(salvage_values),
                                                      budget(budget),
                                                      num_items(buy_costs.size()),
                                                      volume(volume){};
    oaf::GRBFirstStruct init_model(GRBEnv &, const Eigen::VectorXd &) const;

    Eigen::MatrixXd make_first_matrix() const;
    Eigen::MatrixXd make_rec_matrix() const;
    Eigen::VectorXd fixed_rhs() const;
    Eigen::VectorXd default_pred() const;

    oaf::GRBRecDualStruct init_rec_model(GRBEnv &, const Eigen::VectorXd &sol, const Eigen::VectorXd &rec_rhs) const;
    void update_rec_model(oaf::GRBRecDualStruct &, const Eigen::VectorXd &sol, const Eigen::VectorXd &rec_rhs) const;

    Eigen::VectorXd get_adj_costs(const Eigen::VectorXd &, const Eigen::VectorXd &) const;

    Eigen::MatrixXd solve_instances(const std::vector<std::vector<oaf::Scenario>> & instances) const;

    // Eigen::VectorXd get_dual_rhs(Eigen::VectorXd ray_known, Eigen::VectorXd ray_predict);
};

// class VendorDualInitializer:



#endif