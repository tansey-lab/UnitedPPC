import numpy as np
import pandas as pd
import argparse
from UnitedMet.impute_met.pyro_model import pyro_posterior, final_impute_met

if __name__ == "__main__":
    ############################################################# Parse command-line options & arguments
    # Create a parser
    parser = argparse.ArgumentParser(description="Impute metabolomics data from transcriptomics data",
                                     formatter_class=argparse.RawTextHelpFormatter)
    # Add options/arguments
    parser.add_argument('-ve', '--verbose_embeddings', help="Write W, and H matrices to .csv files",
                        action='store_true', default=False)
    parser.add_argument('-rd', '--results_directory', help="the path of the results directory", required=False, type=str,
                        default="/data1/reznike/xiea1/MetabolicModel/results/results_RNA_ccRCC/RC18")
    args = parser.parse_args()
    ve = args.verbose_embeddings
    results_dir = args.results_directory

    # ----------------------- Initialize Variables/Directories/Files -----------------------
    seed = 42
    embedding_dir = f'{results_dir}/embeddings'

    # -------------------------------Load data from files--------------------------------
    with open(f'{embedding_dir}/data_for_posterior_pyro.npy', 'rb') as f:
        N = np.load(f)
        K = np.load(f)
        J = np.load(f)
        start_row = np.load(f)
        stop_row = np.load(f)
        batch_sizes = np.load(f)
        f.close()
    n_batch = len(start_row)
    met_names = pd.read_csv(f'{embedding_dir}/met_names_pyro.csv')['met_names'].values


    with open(f'{embedding_dir}/W_H_loc_scale.npy', 'rb') as f:
        W_loc = np.load(f)
        W_scale = np.load(f)
        H_loc = np.load(f)
        H_scale = np.load(f)
        f.close()
    print(f'Reading posterior latent variables from {embedding_dir}/W_H_loc_scale.npy')

    #------------------------Matrix Multiplication and Posterior Prediction------------------------
    num_samples = 1000  # number of samples to draw from posterior
    rank_hat_draws, rank_hat_mean, rank_hat_std = pyro_posterior(W_loc, W_scale, H_loc, H_scale, n_batch, start_row, stop_row, ve, met_names, embedding_dir, seed, num_samples=1000)
    final_impute_met(embedding_dir, results_dir, batch_sizes, rank_hat_mean, rank_hat_std)






















