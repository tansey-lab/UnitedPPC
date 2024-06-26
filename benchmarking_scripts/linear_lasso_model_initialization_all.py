from sklearn.preprocessing import StandardScaler
import argparse
from Performance_Benchmarking.scripts.data_processing import *


def file_system_init_lasso(file_path, cancer_type, imputation, target, iteration, results_dir):
    # Input Metabolomics and Transcriptomics data directory
    rna_matched_data_dir = f"{file_path}/data/RNA_matched_{cancer_type}"
    met_matched_data_dir = f"{file_path}/data/MET_matched_{cancer_type}"
    rna_imputation_data_dir = f"{file_path}/data/RNA_matched_imputation"
    if not os.path.exists(rna_matched_data_dir) or not os.path.exists(met_matched_data_dir):
        raise OSError(f"Input Metabolomics and Transcriptomics data directory don't exist.")
    if imputation and not os.path.exists(rna_imputation_data_dir):
        raise OSError(f"Input RNA data directory for imputation doesn't exist.")

    # Define results directory and other sub-directories inside it
    if cancer_type == 'ccRCC':
        sub_dir = ['CPTAC', 'CPTAC_val', 'RC18', 'RC20']  # notice that CPTAC is the first one
    elif cancer_type == 'BRCA':
        sub_dir = ['BrCa1', 'BrCa2', 'TNBC']
    proportions = list(repeat(0, len(sub_dir)))
    proportions[target - 1] = 1

    plots_dir = f'{results_dir}/plots'
    iteration_dir = f'{results_dir}/{iteration}'
    embedding_dir = f'{iteration_dir}/embeddings'
    iteration_plots_dir = f'{iteration_dir}/plots'

    for dir in [results_dir, plots_dir, iteration_dir, embedding_dir, iteration_plots_dir]:
        try:
            os.makedirs(dir)
        except FileExistsError:
            print(f"Directory '{dir}' already exists.")
    return rna_matched_data_dir, met_matched_data_dir, rna_imputation_data_dir, sub_dir, proportions, \
        results_dir, plots_dir, iteration_dir, embedding_dir

def tic_normalization_across(data, batch_index_vector):
    normalized_data = np.copy(data)  # Note that, when we modify data, normalized_data changes
    n_batches = batch_index_vector.max() + 1
    for bidx in range(n_batches):  # range(i) include numbers from 0 to i-1
        batch_rows = np.arange(data.shape[0])[
            batch_index_vector == bidx]  # np.arange(n) is same as np.array(range(3))
        batch = data[batch_rows]  # get the separate dataframe for distinct batches
        nan_mask = np.isnan(batch)  # Test element-wise for NaN and return result as a boolean array
        missing = np.all(nan_mask, axis=0)  # boolean array about whether for a whole column is all nan in each row
        min_batch = np.nanmin(batch)  # Return minimum of an array, ignoring any NaNs
        for row in range(batch.shape[0]):
            n_censored = np.sum(nan_mask[row]) - np.sum(missing)  # remember to substrate missing numbers
            row_tic = np.nansum(batch[row]) + 0.5 * n_censored * min_batch
            # values that are NAN but not totally missing are filled with np.nanmin(batch) now
            # get the indices of items in a row which nan_mask[row, ] == True but missing == False
            batch[row, nan_mask[row, :] & ~missing] = 0.5 * min_batch
            batch[row, :] = batch[row, :] / row_tic
        batch = batch / np.nansum(batch, axis=1, keepdims=True)  # make sure the sum of each row is 1
        normalized_data[batch_rows] = batch
    return normalized_data


if __name__ == "__main__":
    # not only take the intersected 86 metabolites, take all available metabolites instead

    ############################################################# Parse command-line options & arguments
    # Create a parser
    parser = argparse.ArgumentParser(description="Impute metabolomics data from transcriptomics data",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-fp', '--file_path', help="the path of the parent directory to read/save all files",
                        required=False, type=str, default="/data1/reznike/xiea1/MetabolicModel/results_RNA_ccRCC/RC18")
    parser.add_argument('-rd', '--results_dir', help="results_dir", required=False, type=str)
    parser.add_argument('-t', '--target', help="target dataset", required=False, type=int, default=1)
    parser.add_argument('-f', '--feature_test_prop',
                        help='proportion of features to be tested in each iteration',
                        required=False, type=float, default=0.5)
    # Options for iteration
    parser.add_argument('-i', '--iteration', help='which iterations it is', required=False,
                        default=1, type=int)
    args = parser.parse_args()
    file_path = args.file_path
    results_dir = args.results_dir
    target = args.target
    f_test_prop = args.feature_test_prop
    iteration = args.iteration

    # ------------------------------------- Initialize Variables/Directories/Files -------------------------------------
    seed = 42
    cancer_type = 'ccRCC'
    imputation = False
    tumor_only = False
    remove_solo = False
    rna_matched_data_dir, met_matched_data_dir, rna_imputation_data_dir, sub_dir, proportions, \
        results_dir, plots_dir, iteration_dir, embedding_dir = file_system_init_lasso(file_path, cancer_type, imputation,
                                                                                 target, iteration, results_dir)
    # Load a master file for mapping between RNA and MET data
    MET_RNA_map = pd.read_csv(f'{file_path}/data/MasterMapping_updated.csv', header=0, index_col='MetabID')

    # -------------------------------------- Load Data ----------------------------------
    # Load MET and RNA data
    met_data, metabolite_map, sample_map_met, met_batch_index_vector = load_met_data(met_matched_data_dir,
                                                                                    tumor_only, MET_RNA_map, sub_dir)
    rna_data, batch_index_vector, gene_map, sample_map_rna, batch_map, batch_sizes, start_row, stop_row, \
        s_test_prop_settings, n_batch = load_rna_data(rna_matched_data_dir, tumor_only, MET_RNA_map, proportions,
                                                      imputation, sub_dir)
    # Check the order of samples in loaded RNA/MET datasets are matched and same as the order in the master map
    check_match(sample_map_met, sample_map_rna, MET_RNA_map, sub_dir, tumor_only, imputation, batch_index_vector)
    # Normalization, Concatenate MET and RNA data, and remove features with all NaNs
    met_data = tic_normalization_across(met_data, met_batch_index_vector)


    if np.all(np.isnan(met_data), axis=0).sum() == 0 & np.all(np.isnan(rna_data), axis=0).sum() == 0:
        print("The aggregate data is clean.")
    else:
        print("Some features need to be removed from the aggregate data.")

    rna_data = np.nan_to_num(rna_data, nan=0)

    # Get the feature, sample names
    features_from_index = {v: k for k, v in metabolite_map.items()}
    met_names = [features_from_index[i] for i in range(len(metabolite_map))]
    features_from_index = {v: k for k, v in gene_map.items()}
    rna_names = [features_from_index[i] for i in range(len(gene_map))]
    samples_from_index = {v: k for k, v in sample_map_met.items()}
    sample_names = [samples_from_index[i] for i in range(len(sample_map_met))]

    # Save data for downstream individual jobs
    pd.DataFrame(met_data, index=sample_names, columns=met_names).to_csv(
        f'{iteration_dir}/met_data.csv')
    pd.DataFrame(rna_data, index=sample_names, columns=rna_names).to_csv(
        f'{iteration_dir}/rna_data.csv')


    # -------------------------------------- Test/Train split ----------------------------------
    # The target dataset is the test set, while the rest are training set.
    # X only includes RNA
    X_test = rna_data[start_row[target-1]:stop_row[target-1], :]
    X_train = np.delete(rna_data, np.arange(start_row[target-1], stop_row[target-1]), axis=0)
    # hold-out f_test_prop metabolites as testing features

    # all available metabolites (non-solo metabolites in the target batch) are used as testing features
    training = np.copy(met_data)
    # Mask all metabolite data of the target batch in the training set
    training[start_row[target-1]:stop_row[target-1], :] = np.nan
    solo = np.all(np.isnan(training), axis=0)
    missing = np.all(np.isnan(met_data[start_row[target-1]:stop_row[target-1], :]), axis=0)
    available_features = np.arange(met_data.shape[1])[(~missing) & (~solo)]
    len_available = len(available_features)
    test_feature_indices = np.random.choice(available_features,
                                            size=int(round(f_test_prop * len(available_features))), replace=False)
    # sort the indices in ascending order
    test_feature_indices = np.sort(test_feature_indices)
    print(f'The number of held-out metabolites is: {len(test_feature_indices)}')

    # y include only held-out metabolites (test_feature_indices)
    y_test = met_data[start_row[target-1]:stop_row[target-1], test_feature_indices]
    y_train = np.delete(met_data[:, test_feature_indices], np.arange(start_row[target-1], stop_row[target-1]), axis=0)


    # -------------------------------------- Standardization: Z-score training data ----------------------------------
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    with open(f'{iteration_dir}/data_for_lasso_run.npy', 'wb') as f:  # 'wb': write as binary
        np.save(f, y_test)  # save arrays in numpy binary .npy files
        np.save(f, X_test)
        np.save(f, y_train)
        np.save(f, X_train)
        np.save(f, test_feature_indices)
        f.close()
