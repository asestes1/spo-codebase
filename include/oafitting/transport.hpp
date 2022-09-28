#ifndef TRANSPORT_H
#define TRANSPORT_H
#include <Eigen/Dense>
#include <oafitting/oafit.hpp>
#include <oafitting/stochprogram.hpp>

class TransportationInitializer : public oaf::ModelInitializer
{
private:
    int n_source;
    int n_sink;
    Eigen::VectorXd production_costs;
    Eigen::VectorXd scrap_costs;
    Eigen::VectorXd unmet_costs;
    Eigen::MatrixXd transport_costs;

    void add_scenario(GRBModel &mymodel,
                      const std::vector<GRBVar> &prodvars,
                      std::vector<std::vector<GRBVar>> &scrapvars,
                      std::vector<std::vector<GRBVar>> &unmetvars,
                      std::vector<std::vector<std::vector<GRBVar>>> &transportvars,
                      std::vector<std::vector<GRBConstr>> &src_constr,
                      std::vector<std::vector<GRBConstr>> &dest_constr,
                      const oaf::Scenario &scenario) const;

    void init_sp(GRBModel &mymodel,
                 std::vector<GRBVar> &prodvars,
                 std::vector<std::vector<GRBVar>> &scrapvars,
                 std::vector<std::vector<GRBVar>> &unmetvars,
                 std::vector<std::vector<std::vector<GRBVar>>> &transportvars,
                 std::vector<std::vector<GRBConstr>> &src_constr,
                 std::vector<std::vector<GRBConstr>> &dest_constr,
                 const std::vector<oaf::Scenario> &instance) const;

    void update_sp(GRBModel &mymodel,
                   const std::vector<GRBVar> &prodvars,
                   std::vector<std::vector<GRBVar>> &scrapvars,
                   std::vector<std::vector<GRBVar>> &unmetvars,
                   std::vector<std::vector<std::vector<GRBVar>>> &transportvars,
                   std::vector<std::vector<GRBConstr>> &src_constr,
                   std::vector<std::vector<GRBConstr>> &dest_constr,
                   const std::vector<oaf::Scenario> &instance) const;

    void pop_scenario(GRBModel &mymodel,
                      std::vector<std::vector<GRBVar>> &scrapvars,
                      std::vector<std::vector<GRBVar>> &unmetvars,
                      std::vector<std::vector<std::vector<GRBVar>>> &transportvars,
                      std::vector<std::vector<GRBConstr>> &src_constr,
                      std::vector<std::vector<GRBConstr>> &dest_constr) const;

    Eigen::VectorXd read_sp_sol(const std::vector<GRBVar> &prodvars) const;

public:
    TransportationInitializer(const Eigen::VectorXd &production_costs,
                              const Eigen::VectorXd &scrap_costs,
                              const Eigen::VectorXd &unmet_costs,
                              const Eigen::MatrixXd &transport_costs) : production_costs(production_costs),
                                                                        scrap_costs(scrap_costs),
                                                                        unmet_costs(unmet_costs),
                                                                        transport_costs(transport_costs),
                                                                        n_source(production_costs.size()),
                                                                        n_sink(unmet_costs.size()){};

    oaf::GRBFirstStruct init_model(GRBEnv &, const Eigen::VectorXd &) const;
    oaf::GRBRecDualStruct init_rec_model(GRBEnv &, const Eigen::VectorXd &sol, const Eigen::VectorXd &rec_rhs) const;

    Eigen::MatrixXd solve_instances(const std::vector<std::vector<oaf::Scenario>> &) const;

    Eigen::MatrixXd make_first_matrix() const;
    Eigen::MatrixXd make_rec_matrix() const;
    Eigen::VectorXd fixed_rhs() const;
    Eigen::VectorXd default_pred() const;

    void update_rec_model(oaf::GRBRecDualStruct &, const Eigen::VectorXd &sol, const Eigen::VectorXd &rec_rhs) const;
    Eigen::VectorXd get_adj_costs(const Eigen::VectorXd &, const Eigen::VectorXd &) const;
};


#endif