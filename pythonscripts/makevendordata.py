import argparse
import numpy as np
import os 
import pandas

TRAIN_SPLIT = 0.75
SALE_COSTS = [1,5,10]
BAKERY_NONSPECIFIC = ['is_schoolholiday', 'is_holiday',
       'is_holiday_next2days', 'rain', 'temperature', 'weekday', 'month']
BAKERY_SPECIFIC = ['promotion_currentweek',
       'promotion_lastweek', 'demand__median_7', 'demand__mean_7',
       'demand__standard_deviation_7', 'demand__variance_7',
       'demand__root_mean_square_7', 'demand__maximum_7', 'demand__minimum_7',
        'demand__median_14', 'demand__mean_14',
       'demand__standard_deviation_14', 'demand__variance_14',
       'demand__root_mean_square_14', 'demand__maximum_14', 'demand__minimum_14',
        'demand__median_28', 'demand__mean_28',
       'demand__standard_deviation_28', 'demand__variance_28',
       'demand__root_mean_square_28', 'demand__maximum_28', 'demand__minimum_28', 'demand']

def onehotencode(dataframe: pandas.DataFrame, column: str):
    
    first = True
    for category in dataframe[column].unique():
        if not first:
            dataframe[str(category)+"_ENCODING"] = dataframe[column].apply(lambda x:x == category)
        first = False
    dataframe.drop(columns=column, inplace=True)
    return None

def write_matrix_to_file(filename: str, matrix: np):
    rows, cols = matrix.shape

    with open(filename, "w") as myfile:
        myfile.write(str(rows)+"\n"+str(cols)+"\n")
        for i in range(0, rows):
            next_line: str = ""
            for j in range(0, cols):
                next_line += str(float(matrix[i, j]))+","
            next_line = next_line[:-1] + "\n"
            myfile.write(next_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bakerypath",
                    help="A path to the bakery data csv", type=str)
    parser.add_argument("outputdir", help="Path to the directory where file are written", type=str)
    parser.add_argument("--seed", type=int, help="Seed for RNG", default=None)
    parser.add_argument("--num", type=int, help="Number of datasets to create.", default=50)
    parser.add_argument("--margin", type=float, help="Profit margin of all items", default=0.9)



    myargs = parser.parse_args()
    if myargs.seed is None:
        mygenerator = np.random.default_rng()
    else:
        mygenerator = np.random.default_rng(seed=myargs.seed)

    outputdir = os.path.abspath(myargs.outputdir)
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    inputpath = os.path.abspath(myargs.bakerypath)
    with open(inputpath) as bakeryfile:
        
        data = pandas.read_csv(inputpath)
        data['date'] = pandas.to_datetime(data['date'])
        for store, storedata in data.groupby(["store"]):
            items = storedata["item"].unique()
            nonitem_data = storedata[['date',*BAKERY_NONSPECIFIC]].drop_duplicates().set_index('date')
            onehotencode(nonitem_data, "weekday")
            onehotencode(nonitem_data, "month")
            # onehotencode(nonitem_data, "year")
            nonitem_data.columns = pandas.MultiIndex.from_product([['all'], nonitem_data.columns])
            item_data = storedata.pivot(index="date", columns="item", values=BAKERY_SPECIFIC)
            



            all_data = pandas.merge(left=nonitem_data, right= item_data, how="inner", left_index=True, right_index=True).sort_index()
            all_data = all_data.drop_duplicates()
            num_train = int(len(all_data.index)*TRAIN_SPLIT)
            
            train_side = all_data.iloc[0:num_train, all_data.columns.get_level_values(0)!='demand']
            test_side = all_data.iloc[num_train:, all_data.columns.get_level_values(0)!='demand']
            train_param=all_data.iloc[0:num_train, all_data.columns.get_level_values(0)=='demand']
            test_param= all_data.iloc[num_train:, all_data.columns.get_level_values(0)=='demand']
            purchase_costs = {item: SALE_COSTS[i]*myargs.margin for i,item in enumerate(sorted(items))}

            budgetspent = sum(all_data[("demand",item)]*SALE_COSTS[i] for i,item in enumerate(sorted(items)))
            volumespent = sum(all_data[("demand",item)] for item in sorted(items))
            print(store, budgetspent.quantile(0.75), volumespent.quantile(0.75))
            for dataset, name in zip([train_side, test_side, train_param, test_param],
                            ["trainside","testside","trainparam","testparam"]):

                outputpath = os.path.join(outputdir, "store"+str(store)+"_"+name)
                dataset.to_csv(outputpath+".csv")
                # write_matrix_to_file(outputpath+".txt", 1*dataset.to_numpy())