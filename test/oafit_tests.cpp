#include <oafitting/oafit.hpp>
#include <oafitting/vendor.hpp>
#include <oafitting/transport.hpp>
#include <catch2/catch_all.hpp>



TEST_CASE("Check solution of newsvendor", "[vendorsolution]")
{
    Eigen::VectorXd buy_costs(2);
    buy_costs << 1, 1;

    Eigen::VectorXd sell_costs(2);
    sell_costs << 4, 3;

    Eigen::VectorXd salvage_costs(2);
    salvage_costs << 0, 0;

    Eigen::VectorXd initial_pred(2);
    initial_pred << 15, 20;

    GRBEnv myenv{false};
    VendorInitializer myinit{buy_costs, sell_costs, salvage_costs,
                             1000.0, 1000.0};
    oaf::GRBFirstSolver mysolver{myenv, myinit, initial_pred};
    oaf::FirstStageResult initresult = mysolver.get_lexmin(initial_pred);
    REQUIRE(initresult.feasible);
    REQUIRE(initresult.lexmin_sol.size() == 4);
    REQUIRE(initresult.lexmin_sol(0) == 15);
    REQUIRE(initresult.lexmin_sol(1) == 20);
    REQUIRE(initresult.lexmin_sol(2) == 965);
    REQUIRE(initresult.lexmin_sol(3) == 965);
    REQUIRE(initresult.first_stage_cost == 35);

    VendorInitializer myinit2{buy_costs, sell_costs, salvage_costs, 30.0,
                              1000.0};
    oaf::GRBFirstSolver mysolver2{myenv, myinit2, initial_pred};
    oaf::FirstStageResult initresult2 = mysolver2.get_lexmin(initial_pred);
    REQUIRE(initresult2.lexmin_sol.size() == 4);
    REQUIRE(initresult2.lexmin_sol(0) == 15);
    REQUIRE(initresult2.lexmin_sol(1) == 15);
    REQUIRE(initresult2.lexmin_sol(2) == 0);
    REQUIRE(initresult2.lexmin_sol(3) == 970);
    REQUIRE(initresult2.first_stage_cost == 30);

    VendorInitializer myinit3{buy_costs, sell_costs, salvage_costs, 10.0,
                                1000.0};
    oaf::GRBFirstSolver mysolver3{myenv, myinit3, initial_pred};
    oaf::FirstStageResult initresult3 = mysolver3.get_lexmin(initial_pred);
    REQUIRE(initresult3.lexmin_sol.size() == 4);
    REQUIRE(initresult3.lexmin_sol(0) == 10);
    REQUIRE(initresult3.lexmin_sol(1) == 0);
    REQUIRE(initresult3.lexmin_sol(2) == 0);
    REQUIRE(initresult3.lexmin_sol(3) == 990);

    REQUIRE(initresult3.first_stage_cost == 10);
}


TEST_CASE("Check dual solution", "[vendorgradient]")
{
    Eigen::VectorXd buy_costs(2);
    buy_costs << 1, 1;

    Eigen::VectorXd sell_costs(2);
    sell_costs << 4, 3;

    Eigen::VectorXd salvage_costs(2);
    salvage_costs << 0, 0;

    Eigen::VectorXd actual_rhs(2);
    actual_rhs << 10, 10;

    Eigen::VectorXd initial_pred(2);
    initial_pred << 15, 20;

    GRBEnv myenv{false};
    VendorInitializer myinit{buy_costs, sell_costs, salvage_costs, 30.0,
                             1000.0};
    oaf::GRBFirstSolver mysolver{myenv, myinit, initial_pred};
    oaf::FirstStageResult initresult = mysolver.get_lexmin(initial_pred);

    oaf::GRBRecSolver mysolve{myenv, myinit, initresult.lexmin_sol, actual_rhs};
    oaf::RecResult myresult = mysolve.solve(initresult.lexmin_sol, actual_rhs);

    REQUIRE(myresult.second_stage_cost == -70);
    
    Eigen::VectorXd gradient = oaf::get_subgradient(initresult, myresult, 6, 2);

    REQUIRE_THAT(gradient(0), Catch::Matchers::WithinAbs(0.0, 1.e-5));
    REQUIRE_THAT(gradient(1), Catch::Matchers::WithinAbs(0.0, 1.e-5));

    initial_pred << 2, 35;
    initresult = mysolver.get_lexmin(initial_pred);

    Eigen::VectorXd other_pred(2);
    other_pred << 3, 35;
    oaf::FirstStageResult other_result = mysolver.get_lexmin(other_pred);

    myresult = mysolve.solve(initresult.lexmin_sol, actual_rhs);

    gradient = get_subgradient(initresult, myresult, 6, 2);


    REQUIRE_THAT(gradient(0), Catch::Matchers::WithinAbs(-4.0, 1.e-5));
    REQUIRE_THAT(gradient(1), Catch::Matchers::WithinAbs(0.0, 1.e-5));

    initial_pred << 2, 2;
    initresult = mysolver.get_lexmin(initial_pred);
    myresult = mysolve.solve(initresult.lexmin_sol, actual_rhs);
    gradient = get_subgradient(initresult, myresult, 6, 2);
    REQUIRE_THAT(gradient(0), Catch::Matchers::WithinAbs(-3.0, 1.e-5));
    REQUIRE_THAT(gradient(1), Catch::Matchers::WithinAbs(-2.0, 1.e-5));

    initial_pred << 2, 12;
    initresult = mysolver.get_lexmin(initial_pred);
    myresult = mysolve.solve(initresult.lexmin_sol, actual_rhs);
    gradient = get_subgradient(initresult, myresult, 6, 2);
    REQUIRE_THAT(gradient(0), Catch::Matchers::WithinAbs(-3.0, 1.e-5));
    REQUIRE_THAT(gradient(1), Catch::Matchers::WithinAbs(1.0, 1.e-5));

    initial_pred << 12, 2;
    initresult = mysolver.get_lexmin(initial_pred);
    myresult = mysolve.solve(initresult.lexmin_sol, actual_rhs);
    gradient = get_subgradient(initresult, myresult, 6, 2);
    REQUIRE_THAT(gradient(0), Catch::Matchers::WithinAbs(1.0, 1.e-5));
    REQUIRE_THAT(gradient(1), Catch::Matchers::WithinAbs(-2.0, 1.e-5));

    initial_pred << 12, 12;
    initresult = mysolver.get_lexmin(initial_pred);
    myresult = mysolve.solve(initresult.lexmin_sol, actual_rhs);
    gradient = get_subgradient(initresult, myresult, 6, 2);
    REQUIRE_THAT(gradient(0), Catch::Matchers::WithinAbs(1.0, 1.e-5));
    REQUIRE_THAT(gradient(1), Catch::Matchers::WithinAbs(1.0, 1.e-5));
}


TEST_CASE("Check find cut", "[vendorfindcut]")
{
    Eigen::VectorXd buy_costs(2);
    buy_costs << 1, 1;

    Eigen::VectorXd sell_costs(2);
    sell_costs << 4, 3;

    Eigen::VectorXd salvage_costs(2);
    salvage_costs << 0, 0;

    Eigen::VectorXd actual_rhs(2);
    actual_rhs << 10, 10;

    Eigen::VectorXd initial_pred(2);
    initial_pred << 15, 20;

    GRBEnv myenv{false};
    VendorInitializer myinit{buy_costs, sell_costs, salvage_costs, 30.0,
                              1000.0};
    oaf::GRBFirstSolver mysolver{myenv, myinit, initial_pred};
    oaf::FirstStageResult initresult = mysolver.get_lexmin(initial_pred);
    double best = mysolver.optimal_value(actual_rhs);

    oaf::GRBRecSolver mysolve{myenv, myinit, initresult.lexmin_sol, actual_rhs};
    oaf::RecResult myresult = mysolve.solve(initresult.lexmin_sol, actual_rhs);
    oaf::OptCut mycut = find_optcut(mysolver, mysolve, initial_pred, actual_rhs, best, initresult,
                               myresult, mysolver.get_basis_size(), mysolver.get_n_predict(), 0.0001);
    REQUIRE_THAT(mycut.loss, Catch::Matchers::WithinAbs(7.5, 1.e-5));
    REQUIRE_THAT(mycut.prediction(0), Catch::Matchers::WithinAbs(12.5, 1.e-5));
    REQUIRE_THAT(mycut.prediction(1), Catch::Matchers::WithinAbs(15, 1.e-5));
    REQUIRE_THAT(mycut.subgrad(0), Catch::Matchers::WithinAbs(1, 1.e-5));
    REQUIRE_THAT(mycut.subgrad(1), Catch::Matchers::WithinAbs(1, 1.e-5));
}


TEST_CASE("Check add feasibility cut", "[vendorfeascut]")
{
    int side_dim = 1;
    int param_dim = 1;

    GRBEnv myenv{false};

    double budget = 25.0;
    double volume = 1000.0;
    Eigen::VectorXd initial_pred(1);
    initial_pred << -10;

    Eigen::VectorXd buy_costs(1);
    buy_costs << 1;

    Eigen::VectorXd sell_costs(1);
    sell_costs << 4;

    Eigen::VectorXd salvage_costs(1);
    salvage_costs << 0;

    Eigen::VectorXd side_info(1);
    side_info << 5;

    VendorInitializer myinit{buy_costs, sell_costs, salvage_costs, budget,
                             volume};
    oaf::GRBFirstSolver mysolver{myenv, myinit, initial_pred};
    oaf::FirstStageResult myresult = mysolver.get_lexmin(initial_pred);

    REQUIRE_FALSE(myresult.feasible);

    Eigen::VectorXd rhs_vector(4);
    rhs_vector << budget, volume, 0, initial_pred;
    Eigen::VectorXd ray(4);
    std::cout << "RK: " << myresult.ray_known << std::endl;
    std::cout << "RP: " << myresult.ray_predict << std::endl;

    ray << myresult.ray_known, myresult.ray_predict;
    REQUIRE(rhs_vector.dot(ray) > 0);
    // Eigen::VectorXd dual_rhs = myinit.get_dual_rhs(myresult.ray_known, myresult.ray_predict);
    // for (int i = 0; i < param_dim * 4 + 1; i++)
    // {
    //     REQUIRE(dual_rhs(i) <= 0);
    // }

    oaf::GRBMasterStruct mymastermodel{myenv, myinit.fixed_rhs(), 1, side_dim, param_dim};
    oaf::addfeascut(mymastermodel, myresult.ray_known, myresult.ray_predict, side_info);
    mymastermodel.model.optimize();

    oaf::FittingResult myfit = mymastermodel.readresult();

    Eigen::VectorXd new_pred = myfit.intercept + myfit.coeff * side_info;
    int i = 0;
    for (i = 0; i < param_dim; i++)
    {
        REQUIRE(new_pred(i) >= 0);
    }
}

/*
TEST_CASE("Check add optimality cut", "[vendoroptcut]")
{
    int side_dim = 1;
    int param_dim = 1;

    GRBEnv myenv{false};

    double budget = 25.0;
    Eigen::VectorXd actual_rhs(1);
    actual_rhs << 20;

    Eigen::VectorXd initial_pred(1);
    initial_pred << 10;

    Eigen::VectorXd buy_costs(1);
    buy_costs << 1;

    Eigen::VectorXd sell_costs(1);
    sell_costs << 4;

    Eigen::VectorXd salvage_costs(1);
    salvage_costs << 0;

    Eigen::VectorXd side_info(1);
    side_info << 5;

    VendorInitializer myinit{buy_costs, sell_costs, salvage_costs, budget,
                              1000.0};
    oaf::GRBFirstSolver mysolver{myenv, myinit, initial_pred};
    oaf::FirstStageResult fs_result = mysolver.get_lexmin(initial_pred);

    REQUIRE(fs_result.feasible);

    oaf::GRBRecSolver rec_solver{myenv, myinit, fs_result.lexmin_sol, actual_rhs};
    oaf::RecResult rec_result = rec_solver.solve(fs_result.lexmin_sol, actual_rhs);

    oaf::GRBMasterStruct mymastermodel{myenv, myinit.fixed_rhs(), 1, side_dim, param_dim};

    double best_obj = mysolver.optimal_value(actual_rhs);

    oaf::OptCut optcut = find_optcut(mysolver, rec_solver, initial_pred, actual_rhs, best_obj, fs_result, rec_result,
                                4, 1, 0.0001);
    
    REQUIRE_THAT(optcut.prediction(0), Catch::Matchers::WithinAbs(10, 0.00001));
    REQUIRE_THAT(optcut.subgrad(0), Catch::Matchers::WithinAbs(-3, 0.00001));
    REQUIRE_THAT(optcut.loss, Catch::Matchers::WithinAbs(30, 0.00001));

    oaf::addoptcut(mymastermodel, optcut.prediction, side_info, optcut.subgrad, optcut.loss, 0);
    oaf::FittingResult myfit = mymastermodel.readresult();
    Eigen::VectorXd newpred = myfit.intercept + myfit.coeff * side_info;
    REQUIRE_THAT(newpred(0), Catch::Matchers::WithinAbs(20, 0.000001));
}
*/

TEST_CASE("Check add optimality cut redux", "[vendoroptcut2]")
{
    int side_dim = 3;
    int param_dim = 4;

    GRBEnv myenv{false};

    double budget = 100.0;
    Eigen::VectorXd actual_rhs(param_dim);
    actual_rhs << 20, 30, 10, 15;

    Eigen::VectorXd initial_pred(4);
    initial_pred << 10, 50, 20, 10;

    Eigen::VectorXd buy_costs(4);
    buy_costs << 1, 1, 1, 1;

    Eigen::VectorXd sell_costs(4);
    sell_costs << 4, 3, 7, 2;

    Eigen::VectorXd salvage_costs(4);
    salvage_costs << 0, 0, 0, 0;

    Eigen::MatrixXd side_info(1, 3);
    side_info << 5, 2, 10;

    Eigen::MatrixXd coeff = Eigen::MatrixXd::Zero(4, 3);
    Eigen::VectorXd intercept = Eigen::VectorXd::Zero(4);
    Eigen::MatrixXd train_parameter(1, 4);
    train_parameter << 20, 30, 10, 15;

    VendorInitializer myinit{buy_costs, sell_costs, salvage_costs, budget,
                             1000.0};
    oaf::GRBFirstSolver mysolver{myenv, myinit, initial_pred};
    oaf::FirstStageResult fs_result = mysolver.get_lexmin(initial_pred);

    REQUIRE(fs_result.feasible);

    oaf::GRBRecSolver rec_solver{myenv, myinit, fs_result.lexmin_sol, actual_rhs};
    oaf::FittingResult myresult = oafit(myenv, side_info, train_parameter, coeff, intercept,
                                   mysolver, rec_solver, 0.05);
    Eigen::MatrixXd newpred = (side_info * myresult.coeff.transpose()).rowwise() + myresult.intercept.transpose();
    REQUIRE_THAT(newpred(0, 0), Catch::Matchers::WithinAbs(20, 0.000001));
    REQUIRE_THAT(newpred(0, 1), Catch::Matchers::WithinAbs(30, 0.000001));
    REQUIRE_THAT(newpred(0, 2), Catch::Matchers::WithinAbs(10, 0.000001));
    REQUIRE_THAT(newpred(0, 3), Catch::Matchers::WithinAbs(15, 0.000001));
}

TEST_CASE("Check lexmin solution of newsvendor", "[vendorsolution]")
{
    Eigen::VectorXd buy_costs(4);
    buy_costs << 2, 2, 2, 2;

    Eigen::VectorXd sell_costs(4);
    sell_costs << 3.0, 3.0, 3.0, 4;

    Eigen::VectorXd salvage_costs(4);
    salvage_costs << 0, 0, 0, 0;

    Eigen::VectorXd initial_pred(4);
    initial_pred << 15, 20, 14, 20;

    GRBEnv myenv{false};
    VendorInitializer myinit{buy_costs, sell_costs, salvage_costs, 60.0,
                              1000.0};
    oaf::GRBFirstSolver mysolver{myenv, myinit, initial_pred};
    oaf::FirstStageResult initresult = mysolver.get_lexmin(initial_pred);
    REQUIRE(initresult.feasible);
    REQUIRE(initresult.lexmin_sol.size() == 6);
    REQUIRE(initresult.lexmin_sol(0) == 0);
    REQUIRE(initresult.lexmin_sol(1) == 0);
    REQUIRE(initresult.lexmin_sol(2) == 10);
    REQUIRE(initresult.lexmin_sol(3) == 20);
    REQUIRE(initresult.lexmin_sol(4) == 0);
    REQUIRE(initresult.lexmin_sol(5) == 970);
    REQUIRE(initresult.first_stage_cost == 60);
}


TEST_CASE("Check transportation problem solutions", "[transportation]")
{
    Eigen::VectorXd prodcost(2);
    prodcost << 1, 2;
    Eigen::VectorXd scrapcost(2);
    scrapcost << 0.25, 0.5;
    Eigen::VectorXd unmetcost(2);
    unmetcost << 100, 200;
    Eigen::MatrixXd tcost(2, 2);
    tcost << 1, 2, 12, 1;

    Eigen::VectorXd pred(2);
    pred << 10, 20;

    Eigen::VectorXd actual(2);
    actual << 8, 18;

    TransportationInitializer myinit{prodcost, scrapcost, unmetcost, tcost};
    GRBEnv myenv{false};
    oaf::GRBFirstSolver mysolver{myenv, myinit, pred};
    oaf::FirstStageResult initresult = mysolver.get_lexmin(pred);
    REQUIRE(initresult.feasible);
    REQUIRE(initresult.lexmin_sol(0) == 10);
    REQUIRE(initresult.lexmin_sol(1) == 20);
    REQUIRE(initresult.first_stage_cost == 50);

    oaf::GRBRecSolver rec_solver{myenv, myinit, initresult.lexmin_sol, actual};
    oaf::RecResult rec_result = rec_solver.solve(initresult.lexmin_sol, actual);
    Eigen::VectorXd gradient = get_subgradient(initresult, rec_result, 4, 2);
    REQUIRE_THAT(gradient(0), Catch::Matchers::WithinAbs(1.25, 0.00001));
    REQUIRE_THAT(gradient(1), Catch::Matchers::WithinAbs(2.5, 0.00001));

    actual << 12, 22;

    rec_result = rec_solver.solve(initresult.lexmin_sol, actual);
    gradient = get_subgradient(initresult, rec_result, 4, 2);
    REQUIRE_THAT(gradient(0), Catch::Matchers::WithinAbs(-98, 0.00001));
    REQUIRE_THAT(gradient(1), Catch::Matchers::WithinAbs(-98, 0.00001));
}

TEST_CASE("Check fitting procedure, transportation problem", "[tfit]")
{
    int n_src = 4;
    int n_sink = 3;
    int n_samples = 10;
    int side_dim = 2;
    int param_dim = n_sink;

    GRBEnv myenv{false};
    Eigen::MatrixXd train_side = Eigen::MatrixXd::Random(n_samples, side_dim);
    train_side = train_side + Eigen::MatrixXd::Constant(n_samples, side_dim, 2);
    Eigen::MatrixXd real_coeff = Eigen::MatrixXd::Ones(param_dim, side_dim);
    Eigen::VectorXd real_intercept{param_dim};
    real_intercept << 5, 7, 2;
    Eigen::MatrixXd train_parameter = (train_side * real_coeff.transpose()).rowwise() + real_intercept.transpose();

    Eigen::MatrixXd coeff_guess = Eigen::MatrixXd::Zero(param_dim, side_dim);
    Eigen::VectorXd intercept_guess(param_dim);
    intercept_guess << -1, -1, -1;

    Eigen::VectorXd initial_pred(param_dim);
    initial_pred << 0, 0, 0;

    Eigen::VectorXd prod_costs = Eigen::VectorXd::Random(n_src) + Eigen::VectorXd::Constant(n_src, 2);
    Eigen::VectorXd scrap_costs = Eigen::VectorXd::Random(n_src) + Eigen::VectorXd::Constant(n_src, 2);
    Eigen::VectorXd unmet_costs = Eigen::VectorXd::Random(n_sink) + Eigen::VectorXd::Constant(n_sink, 100);
    Eigen::MatrixXd transport_costs{n_src, n_sink};
    transport_costs << 1, 100, 100,
        100, 1, 100,
        100, 100, 1,
        100, 100, 100;

    TransportationInitializer myinit = TransportationInitializer(prod_costs, scrap_costs, unmet_costs, transport_costs);
    oaf::GRBFirstSolver mysolver{myenv, myinit, initial_pred};
    oaf::FirstStageResult fs_result = mysolver.get_lexmin(initial_pred);

    oaf::GRBRecSolver rec_solver{myenv, myinit, fs_result.lexmin_sol, initial_pred};

    oaf::FittingResult myresult = oafit(myenv, train_side, train_parameter, coeff_guess, intercept_guess,
                                   mysolver, rec_solver, 0.05);
    Eigen::MatrixXd newpred = (train_side * myresult.coeff.transpose()).rowwise() + myresult.intercept.transpose();
    for (int i = 0; i < n_samples; i++)
    {
        for (int j = 0; j < param_dim; j++)
        {
            REQUIRE_THAT(newpred(i, j) - train_parameter(i, j), Catch::Matchers::WithinAbs(0, 0.00001));
        }
    }
}
