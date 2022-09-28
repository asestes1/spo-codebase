import numpy
import typing
import sys
import os
import json
import subprocess
import argparse
import shutil

from numpy.lib.twodim_base import tri


class StatModelParams(typing.NamedTuple):
    trainset_size: int
    validationset_size: int
    testset_size: int

    param_dim: int

    beta_bounds: typing.Tuple[float, float]
    alpha_bounds: typing.Tuple[float, float]

    side_dim: int
    side_bounds: typing.Tuple[float, float]
    noisebounds: typing.Tuple[float, float]

    power: typing.Optional[float]


def read_statmodel_from_json(filename: str) -> StatModelParams:
    with open(filename, 'r') as paramfile:
        myparams = json.load(paramfile)
        trial_params = StatModelParams(trainset_size=myparams["trainsetsize"],
                                       validationset_size=myparams["validationsetsize"],
                                       testset_size=myparams["testsetsize"],
                                       side_dim=myparams["sidedim"],
                                       param_dim=myparams["paramdim"],
                                       seed=myparams["seed"],
                                       alpha_bounds=myparams["model"]["alphabd"],
                                       beta_bounds=myparams["model"]["betabd"],
                                       side_bounds=myparams["model"]["sidebd"],
                                       noisebounds=myparams["model"]["noisebd"],
                                       power=myparams["model"]["power"])
    return trial_params


def uniform_rand_matrix(bounds: typing.Tuple[float, float], dimensions: typing.List[int],
                        generator: numpy.random.Generator) -> numpy.ndarray:
    return bounds[0] + (bounds[1] - bounds[0]) * generator.random(size=dimensions, dtype=numpy.float64)


class DataStruct(typing.NamedTuple):
    features: numpy.ndarray
    target: numpy.ndarray


class DataSubdivision(typing.NamedTuple):
    train: DataStruct
    validate: DataStruct
    test: DataStruct


class TransportGenParams(typing.NamedTuple):
    num_src: int
    num_dest: int


class TransportModel(typing.NamedTuple):
    num_src: int
    num_dest: int
    prod_costs: typing.List[float]
    unmet_costs: typing.List[float]
    scrap_costs: typing.List[float]
    transport_costs: typing.List[typing.List[float]]


def transport_model_to_jsondict(model: TransportModel) -> typing.Dict:
    return {"modeltype": "transportation", "prod_costs": model.prod_costs, "unmet_costs": model.unmet_costs,
            "scrap_costs": model.scrap_costs, "transport_costs": model.transport_costs}


def generate_transport_model(model_params: TransportGenParams, mygenerator: numpy.random.Generator) -> TransportModel:
    prod_costs = [mygenerator.uniform(0, 1)
                  for _ in range(0, model_params.num_src)]
    scrap_costs = [mygenerator.uniform(0, 1)
                   for _ in range(0, model_params.num_src)]
    transport_costs = [[mygenerator.uniform(0, 1) for _ in range(0, model_params.num_dest)]
                       for _ in range(0, model_params.num_src)]
    prodconst = max(min(prod_costs[j] + transport_costs[j][i] for j in range(0, model_params.num_src))
                    for i in range(0, model_params.num_dest))
    unmet_costs = [
        prodconst+mygenerator.uniform(0, 1) for i in range(0, model_params.num_dest)]

    return TransportModel(num_src=model_params.num_src, num_dest=model_params.num_dest,
                          prod_costs=prod_costs, scrap_costs=scrap_costs,
                          unmet_costs=unmet_costs, transport_costs=transport_costs)


def generate_data_sets(trial_params: StatModelParams, mygenerator: numpy.random.Generator) -> DataSubdivision:
    true_beta = uniform_rand_matrix(bounds=trial_params.beta_bounds,
                                    dimensions=[trial_params.side_dim,
                                                trial_params.param_dim],
                                    generator=mygenerator)
    num_gen = (trial_params.trainset_size + trial_params.validationset_size +
               trial_params.testset_size)
    all_x = uniform_rand_matrix(bounds=trial_params.side_bounds, dimensions=[
                                num_gen, trial_params.side_dim],
                                generator=mygenerator)
    true_alpha = uniform_rand_matrix(
        bounds=trial_params.alpha_bounds, dimensions=[trial_params.param_dim],
        generator=mygenerator)
    train_noise = uniform_rand_matrix(bounds=trial_params.noisebounds, dimensions=[
                                      num_gen, trial_params.param_dim],
                                      generator=mygenerator)

    all_y = all_x.dot(true_beta) + true_alpha + train_noise
    if trial_params.power is not None:
        all_y = numpy.power(all_y, trial_params.power)

    train_x = all_x[:][0:trial_params.trainset_size]
    validate_x = all_x[:][trial_params.trainset_size:(
        trial_params.trainset_size + trial_params.validationset_size)]
    test_x = all_x[:][(trial_params.trainset_size +
                       trial_params.validationset_size):num_gen]

    train_y = all_y[:][0:trial_params.trainset_size]
    validate_y = all_y[:][trial_params.trainset_size:(
        trial_params.trainset_size + trial_params.validationset_size)]
    test_y = all_y[:][(trial_params.trainset_size +
                       trial_params.validationset_size):num_gen]
    return DataSubdivision(train=DataStruct(features=train_x, target=train_y),
                           validate=DataStruct(
                               features=validate_x, target=validate_y),
                           test=DataStruct(features=test_x, target=test_y))


def write_matrix_to_file(filename: str, matrix: numpy.ndarray):
    rows, cols = matrix.shape

    with open(filename, "w") as myfile:
        myfile.write(str(rows)+"\n"+str(cols)+"\n")
        for i in range(0, rows):
            next_line: str = ""
            for j in range(0, cols):
                next_line += str(float(matrix[i, j]))+","
            next_line = next_line[:-1] + "\n"
            myfile.write(next_line)


def run_trial(execpath, timing_outputpath, loss_outputpath, nsrc, ndest, noise, linearity, tempdir, train_size, test_size, side_dim,
              mygenerator: numpy.random.Generator, knn_validation_prop=0.20, knn_levels=[1, 2, 4, 8, 16, 32, 64, 128]) -> None:
    if not os.path.isdir(tempdir):
        os.mkdir(tempdir)
    statmodelparams = StatModelParams(trainset_size=train_size, validationset_size=0, testset_size=test_size,
                                      param_dim=ndest, beta_bounds=(0, 1), alpha_bounds=(0, 1),
                                      side_dim=side_dim, side_bounds=[0, 1], noisebounds=[0, noise],
                                      power=linearity)
    mydata = generate_data_sets(
        trial_params=statmodelparams, mygenerator=mygenerator)

    train_side_name = os.path.abspath(os.path.join(tempdir, "train_side.txt"))
    test_side_name = os.path.abspath(os.path.join(tempdir, "test_side.txt"))
    train_param_name = os.path.abspath(
        os.path.join(tempdir, "train_param.txt"))
    test_param_name = os.path.abspath(os.path.join(tempdir, "test_param.txt"))

    write_matrix_to_file(train_side_name, mydata.train.features)
    write_matrix_to_file(test_side_name, mydata.test.features)
    write_matrix_to_file(train_param_name, mydata.train.target)
    write_matrix_to_file(test_param_name, mydata.test.target)

    transportparams = TransportGenParams(num_src=nsrc, num_dest=ndest)
    mytransport = generate_transport_model(
        model_params=transportparams, mygenerator=mygenerator)
    jsondict = transport_model_to_jsondict(mytransport)
    transport_name = os.path.abspath(os.path.join(tempdir, "transport.json"))
    with open(transport_name, "w") as transport_file:
        json.dump(jsondict, transport_file)

    knn_name = os.path.abspath(os.path.join(tempdir, "knn.json"))
    with open(knn_name, "w") as knn_file:
        json.dump({"tuningmethod": "validate", "validationprop": knn_validation_prop,
                   "possiblevals": knn_levels}, knn_file)
    subprocess.run([execpath, transport_name, train_side_name, train_param_name, test_side_name, test_param_name,
                    knn_name, loss_outputpath, timing_outputpath])
    return


def run_trials(execpath, outputdir, tempdir, nsrc, ndest, linearity, noise, ntrials, trainsize, testsize, sidedim, mygenerator,
               knn_validation_prop=0.2, knn_levels=[1, 2, 5, 10, 100]):
    for i in range(0, ntrials):
        timing_path = os.path.abspath(
            os.path.join(outputdir, "timing"+str(i)+".csv"))
        loss_path = os.path.abspath(
            os.path.join(outputdir, "loss"+str(i)+".csv"))
        run_trial(execpath=execpath,
                  timing_outputpath=timing_path,
                  loss_outputpath=loss_path,
                  nsrc=nsrc, ndest=ndest,
                  noise=noise,
                  linearity=linearity, tempdir=tempdir, train_size=trainsize, test_size=testsize, side_dim=sidedim,
                  knn_validation_prop=knn_validation_prop,
                  knn_levels=knn_levels, mygenerator=mygenerator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("execpath",
                        help="A path to the executable compiled from the c++ file `src/runtrial.cpp`", type=str)
    parser.add_argument("outputdir",
                        help="A path to the output directory", type=str)
    parser.add_argument("tempdir",
                        help="A path to a temporary directory that the python script will write files to", type=str)
    parser.add_argument("testtype", type=str,
                        help="Choice of test to run", choices=["varylinearity", "varynoise", "varysize", "varysizelarge"])
    parser.add_argument("--seed", type=int, help="Seed for RNG", default=None)
    parser.add_argument("--linearity", type=float, help="Linearity, if testtype is not varylinearity", default=1)
    parser.add_argument("--noise", type=float, help="Noise, if testtype is not varynoise", default=1)
    parser.add_argument("--size", type=int, help="Size, if testtype is not varysize", default=6)

    myargs = parser.parse_args()
    if myargs.seed is None:
        mygenerator = numpy.random.default_rng()
    else:
        mygenerator = numpy.random.default_rng(seed=myargs.seed)

    outputdir = os.path.abspath(myargs.outputdir)
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    if myargs.testtype == "varylinearity":
        for linearity in [1,2,4,8]:
            for trainsize in [100, 250, 500, 1000]:
                suboutputdir = os.path.join(outputdir,"lin_"+str(linearity)+"_train_"+str(trainsize))

                if not os.path.isdir(suboutputdir):
                    if os.path.exists(suboutputdir):
                        os.remove(suboutputdir)
                    os.mkdir(suboutputdir)
                run_trials(myargs.execpath, suboutputdir, myargs.tempdir,
                            nsrc=myargs.size, ndest=myargs.size,
                             linearity=linearity, noise=myargs.noise, ntrials=100, trainsize=trainsize, sidedim=3,
                            testsize=50,
                            mygenerator=mygenerator)
    if myargs.testtype == "varynoise":
        for noise in [1,2,4,8]:
            for trainsize in [100, 250, 500, 1000]:
                suboutputdir = os.path.join(outputdir,"noise_"+str(noise)+"_train_"+str(trainsize))

                if not os.path.isdir(suboutputdir):
                    if os.path.exists(suboutputdir):
                        os.remove(suboutputdir)
                    os.mkdir(suboutputdir)
                run_trials(myargs.execpath, suboutputdir, myargs.tempdir,
                            nsrc=myargs.size, ndest=myargs.size,
                            linearity=myargs.linearity,
                            noise=noise, ntrials=100, trainsize=trainsize, sidedim=3,
                            testsize=50,
                            mygenerator=mygenerator)

    if myargs.testtype == "varysize":
        for size in [5,10,20, 30, 40, 50]:
            for trainsize in [100, 250, 500, 1000, 5000]:
                suboutputdir = os.path.join(outputdir,"size_"+str(size)+"_train_"+str(trainsize))

                if not os.path.isdir(suboutputdir):
                    if os.path.exists(suboutputdir):
                        os.remove(suboutputdir)
                    os.mkdir(suboutputdir)
                run_trials(myargs.execpath, suboutputdir, myargs.tempdir,
                            nsrc=size, ndest=size,
                            linearity=myargs.linearity,
                            noise=myargs.noise, ntrials=1, trainsize=trainsize, sidedim=3,
                            testsize=50,
                            mygenerator=mygenerator)
    if myargs.testtype == "varysizelarge":
        for size in [60,70,80,90,100]:
            for trainsize in [100, 250, 500, 1000, 5000, 10000]:
                suboutputdir = os.path.join(outputdir,"size_"+str(size)+"_train_"+str(trainsize))

                if not os.path.isdir(suboutputdir):
                    if os.path.exists(suboutputdir):
                        os.remove(suboutputdir)
                    os.mkdir(suboutputdir)
                run_trials(myargs.execpath, suboutputdir, myargs.tempdir,
                            nsrc=size, ndest=size,
                            linearity=myargs.linearity,
                            noise=myargs.noise, ntrials=1, trainsize=trainsize, sidedim=3,
                            testsize=50,
                            mygenerator=mygenerator)