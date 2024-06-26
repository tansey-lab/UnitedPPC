from numpy.random import MT19937
from itertools import repeat
import torch
import argparse
from UnitedMet.impute_met.data_processing import *
from UnitedMet.impute_met.pyro_model import generate_pyro_data, run_pyro_svi, run_pyro_svi_weighted, svi_loss, pyro_posterior, final_impute_met


if __name__ == "__main__":
    ############################################################# Parse command-line options & arguments
    # Create a parser
    parser = argparse.ArgumentParser(description="Impute metabolomics data from transcriptomics data",
                                     formatter_class=argparse.RawTextHelpFormatter)
    # Add options/arguments
    parser.add_argument('-n', '--ndims', help='Number of embedding dimensions for UnitedMet', required=False, default=30,
                        type=int)
    parser.add_argument('-ve', '--verbose_embeddings', help="Write W, H posterior draws to npy files",
                        action='store_true', default=False)
    # Options for pyro SVI
    parser.add_argument('-ns', '--n_steps', help='Iteration steps of pyro SVI', required=False, type=int, default=4000)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate of pyro SVI', required=False, type=float,
                        default=0.001)

    # Options for input data
    parser.add_argument('-fp', '--file_path', help="the path of the parent directory for all input files",
                        required=False, type=str)
    parser.add_argument('-rna', '--rna_matched_data_dir', help="input dir for paired RNA-seq reference data", required=False, type=str)
    parser.add_argument('-met', '--met_matched_data_dir', help="input dir for paired metabolomics reference data", required=False, type=str)
    parser.add_argument('-id', '--rna_imputation_data_dir', help="input dir for single-modality RBA-seq data", required=False, type=str, default="TCGA")
    parser.add_argument('-ck', '--check_match', help='whether to check if metabolomics samples are matched with paired RNA-seq samples or not, '
                                                     'only choose yes if using reference datasets we provided',
                        required=False, action='store_true', default=False)

    # Options for output
    parser.add_argument('-rd', '--results_dir', help="the directory to save all output files", required=False, type=str)

    # Parsing arguments from command lines
    args = parser.parse_args()
    n_dims = args.ndims
    ve = args.verbose_embeddings
    n_steps = args.n_steps
    lr = args.learning_rate
    file_path = args.file_path
    rna_matched_data_dir = args.rna_matched_data_dir
    met_matched_data_dir = args.met_matched_data_dir
    rna_imputation_data_dir = args.rna_imputation_data_dir
    check_match = args.check_match
    results_dir = args.results_dir


    # ------------------------------------- Initialize Variables/Directories/Files -------------------------------------
    seed = 42
    (plots_dir, embedding_dir) = file_system_init(rna_matched_data_dir, met_matched_data_dir, rna_imputation_data_dir, results_dir)

    # -------------------------------------- Load data and Normalize ----------------------------------
    # Load MET and RNA data
    met_data, metabolite_map, sample_map_met, met_batch_index_vector = load_met_data(met_matched_data_dir)
    rna_data, batch_index_vector, gene_map, sample_map_rna, batch_map, batch_sizes, start_row, stop_row, \
    n_batch = load_rna_data(rna_matched_data_dir, rna_imputation_data_dir)

    # Check the order of samples in loaded RNA/MET datasets are matched and same as the order in the master map
    if check_match:
        # Load a master file for mapping between RNA and MET samples
        MET_RNA_map = pd.read_csv(f'{file_path}/MasterMapping_ccRCC.csv', header=0, index_col='MetabID')
        check_match_ccRCC(sample_map_met, sample_map_rna, MET_RNA_map, batch_index_vector)
    # Normalization, Concatenate MET and RNA data, and remove features with all NaNs
    data, met_data, rna_data, metabolite_map, gene_map, met_names, rna_names, feature_names, sample_names = \
        data_normalization_cleaning(met_data, rna_data, met_batch_index_vector, metabolite_map,
                                gene_map, sample_map_rna, batch_index_vector)
    pd.DataFrame(data, index=sample_names, columns=feature_names).to_csv(
        f'{embedding_dir}/normalized_met_rna_data_pyro.csv')
    n_mets, n_genes = met_data.shape[1], rna_data.shape[1]


    # -------------------------------------- Pyro modeling --------------------------------------
    N, J, K, n_obs, orders, ranks, censor_indicator = generate_pyro_data(data, n_dims, n_batch,
                                                                                batch_index_vector)
    n_obs_pre = np.copy(n_obs)  # save the n_obs of the original data

    torch.manual_seed(seed)
    W_loc, W_scale, H_loc, H_scale, loss_list = run_pyro_svi(N, J, K, n_batch, start_row, stop_row, n_obs, orders, n_steps, lr)
    svi_loss(loss_list, plots_dir)  # plot the loss function of the model

    with open(f'{embedding_dir}/W_H_loc_scale.npy', 'wb') as f:  # 'wb': write as binary
        np.save(f, W_loc)
        np.save(f, W_scale)
        np.save(f, H_loc)
        np.save(f, H_scale)
        f.close()


    #------------------------Posterior Prediction------------------------
    num_samples = 1000  # number of samples to draw from posterior
    rank_hat_draws, rank_hat_mean, rank_hat_std = pyro_posterior(W_loc, W_scale, H_loc, H_scale, n_batch, start_row, stop_row, ve, met_names, embedding_dir, seed, num_samples=1000)
    final_impute_met(embedding_dir, results_dir, batch_sizes, rank_hat_mean, rank_hat_std)



