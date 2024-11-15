import argparse
import csv
import pickle

from UnitedPPC.data_processing import *
from UnitedPPC.pyro_model import generate_pyro_data
from UnitedPPC.utils import parsearg_utils, split_folds_across
from UnitedPPC.cross_validation import saving_data_for_cv

if __name__ == "__main__":
    # ------------------------------------- argparse -------------------------------------
    args = parsearg_utils().parse_args()
    print(args)

    # input options
    # file_rna = "/data1/tanseyw/projects/whitej/UnitedPPC/data/df_count_inner.csv"
    # file_zscore = "/data1/tanseyw/projects/whitej/UnitedPPC/data/mean_aucs.csv"
    # file_meta = "/data1/tanseyw/projects/whitej/UnitedPPC/data/sample_metadata.csv"
    file_rna = args.rna_file
    file_zscore = args.zscore_file
    file_meta = args.metadata_file
    # cross-validation options
    cv_dims = args.crossval_dims
    cv_step = args.crossval_steps
    cv_folds = args.crossval_folds
    # output options
    dir_results = args.results_dir

    # ------------------------------------- init -------------------------------------
    seed = 42
    (plots_dir, embedding_dir) = file_system_init(dir_results)

    # ------------------------------------- main -------------------------------------
    # load and concatenate RNA-seq and z-score data

    df_merge, batch_index_vector, n_batch, n_rna, n_zscore = \
        generate_concatenated_ppc_data(
            rna_file = file_rna,
            zscore_file = file_zscore,
            meta_file = file_meta,
        )
    data = df_merge.to_numpy()

    # ------------------------------------- cv prep -------------------------------------
    n_dims = None
    N, J, K, n_obs, orders, ranks, censor_indicator = \
        generate_pyro_data(
            data, 
            n_dims, 
            n_batch, 
            batch_index_vector
        )

    # subset to only include the z-score features (whose names are 32 characters long)
    #TODO: Should this really be all non-missing data (including RNA-seq too)?
    arr_zscores = df_merge[df_merge.columns[
        df_merge.columns.map(lambda x: len(x) == 32)]].to_numpy()
    training = np.copy(arr_zscores)

    # NOT IN USE
    # subset to only include the metabolite features to save memory
    # ranks = ranks[:, :len(met_names)]

    # this can all be extracted from batch_index_vector
    batch_sizes = [list(batch_index_vector).count(i) for i in set(batch_index_vector)]
    start_row = list(np.cumsum([0] + batch_sizes)[:-1])
    stop_row = list(np.cumsum(batch_sizes))

    saving_data_for_cv(
        dir_results, 
        orders, 
        n_obs, 
        batch_sizes,
        start_row,
        stop_row,
        ranks, 
        n_zscore, 
        n_rna
    )

    # ------------------------------------- init cv -------------------------------------
    print("Beginning cross-validation.")

    out_header = ["n_dims", "fold", "mae"]

    with open(f"{dir_results}/cv_folds_scores.csv", "w+") as file:
        writer = csv.writer(file)
        writer.writerow(out_header)

    # "w": Opens a file for writing. Creates a new file if it does not exist or truncates(empty) the file if it exists.
    open(f"{dir_results}/scores.txt", "w").close()

    # ------------------------------------- split and save -------------------------------------
    np.random.seed(seed)

    folds = split_folds_across(
        training, 
        n_batch, 
        batch_index_vector, 
        cv_folds
    )

    if not os.path.exists(f"{dir_results}/folds"):
        os.makedirs(f"{dir_results}/folds")

    for fidx, fold in enumerate(folds):
        fold_temp = f"{dir_results}/folds/fold_{fidx}.pickle"
        pickle.dump(fold, open(fold_temp, "wb+"))
