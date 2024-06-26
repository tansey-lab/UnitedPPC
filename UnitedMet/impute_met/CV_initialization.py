import argparse
import csv
from UnitedMet.impute_met.data_processing import *
from UnitedMet.impute_met.pyro_model import generate_pyro_data
from UnitedMet.impute_met.utils import split_folds_across
from UnitedMet.impute_met.cross_validation import saving_data_for_cv
import pickle

if __name__ == "__main__":
    ############################################################# Parse command-line options & arguments
    # Create a parser
    parser = argparse.ArgumentParser(description="Impute metabolomics data from transcriptomics data",
                                     formatter_class=argparse.RawTextHelpFormatter)
    # Add options/arguments
    # Options for cross-validation
    parser.add_argument('-cd', '--crossval_dims', help='Range of dimensions to evaluate in cross-validation',
                        required=False, nargs=2, type=int, default=[1, 202])
    parser.add_argument('-cs', '--crossval_steps', help='Steps of dimensions in cross-validation', required=False,
                        type=int, default=5)
    parser.add_argument('-cf', '--crossval_folds', help='Number of folds for cross-validation', required=False,
                        type=int, default=10)

    # Options for input data
    parser.add_argument('-fp', '--file_path', help="the path of the parent directory for all input files",
                        required=False, type=str)
    parser.add_argument('-rna', '--rna_matched_data_dir', help="input dir for paired RNA-seq reference data", required=False, type=str)
    parser.add_argument('-met', '--met_matched_data_dir', help="input dir for paired metabolomics reference data", required=False, type=str)
    parser.add_argument('-id', '--rna_imputation_data_dir', help="input dir for single-modality RNA-seq data", required=False, type=str, default="TCGA")
    parser.add_argument('-ck', '--check_match', help='whether to check if metabolomics samples are matched with paired RNA-seq samples or not, '
                                                     'only choose yes if using reference datasets we provided',
                        required=False, action='store_true', default=False)

    # Options for output
    parser.add_argument('-rd', '--results_dir', help="the directory to save all output files", required=False, type=str)

    # Parsing arguments from command lines
    args = parser.parse_args()
    cv_dims = args.crossval_dims
    cv_step = args.crossval_steps
    cv_folds = args.crossval_folds
    file_path = args.file_path
    rna_matched_data_dir = args.rna_matched_data_dir
    met_matched_data_dir = args.met_matched_data_dir
    rna_imputation_data_dir = args.rna_imputation_data_dir
    check_match = args.check_match
    results_dir = args.results_dir



    # ------------------------------------- Initialize Variables/Directories/Files -------------------------------------
    seed = 42
    (plots_dir, embedding_dir) = file_system_init(rna_matched_data_dir, met_matched_data_dir, rna_imputation_data_dir, results_dir)

    # -------------------------------------- MAIN ----------------------------------
    # Load MET and RNA data
    met_data, metabolite_map, sample_map_met, met_batch_index_vector = load_met_data(met_matched_data_dir)
    rna_data, batch_index_vector, gene_map, sample_map_rna, batch_map, batch_sizes, start_row, stop_row, \
    n_batch = load_rna_data(rna_matched_data_dir, rna_imputation_data_dir)

    # Check the order of samples in loaded RNA/MET datasets are matched and same as the order in the master map
    if check_match:
        # Load a master file for mapping between RNA and MET samples
        MET_RNA_map = pd.read_csv(f'{file_path}/MasterMapping_ccRCC.csv', header=0, index_col='MetabID')
        check_match(sample_map_met, sample_map_rna, MET_RNA_map, batch_index_vector)
    # Normalization, Concatenate MET and RNA data, and remove features with all NaNs
    data, met_data, rna_data, metabolite_map, gene_map, met_names, rna_names, feature_names, sample_names = \
        data_normalization_cleaning(met_data, rna_data, met_batch_index_vector, metabolite_map,
                                gene_map, sample_map_rna, batch_index_vector)
    n_mets, n_genes = met_data.shape[1], rna_data.shape[1]

    # -------------------------------------- Prepare data for CV ----------------------------------
    n_dims = None
    N, J, K, n_obs, orders, ranks, censor_indicator = generate_pyro_data(data, n_dims, n_batch, batch_index_vector)
    training = np.copy(met_data)
    ranks = ranks[:, :len(met_names)]  # subset to only include the metabolite features to save memory
    saving_data_for_cv(results_dir, orders, n_obs, batch_sizes, start_row, stop_row, ranks, n_mets, n_genes)

    # -------------------------------------- Initialize CV ----------------------------------
    print('Beginning cross-validation.')
    out_header = ["n_dims", "fold", "mae"]
    with open(f'{results_dir}/cv_folds_scores.csv', 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(out_header)
    open(f"{results_dir}/scores.txt",
         "w").close()  # 'w': Opens a file for writing. Creates a new file if it does not exist or truncates(empty) the file if it exists.

    # -------------------------------------- Split and save folds ----------------------------------
    np.random.seed(seed)
    folds = split_folds_across(training, n_batch, batch_index_vector, cv_folds)
    if not os.path.exists(f"{results_dir}/folds"):
        os.makedirs(f"{results_dir}/folds")
    for fidx, fold in enumerate(folds):
        fold_temp = f"{results_dir}/folds/fold_{fidx}.pickle"
        pickle.dump(fold, open(fold_temp, 'wb+'))




















