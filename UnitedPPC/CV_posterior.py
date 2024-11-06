from datetime import datetime
import numpy as np
import pickle
import os
import csv

from UnitedPPC.utils import parsearg_utils
from UnitedPPC.pyro_model import posterior_prediction_3D
from UnitedPPC.cross_validation import cv_score_mae_met_rna

if __name__ == "__main__":
    # ------------------------------------- argparse -------------------------------------
    args = parsearg_utils().parse_args()

    # output options
    dir_results = args.results_dir
    # cross-validation options
    cv_dims = args.crossval_dims
    cv_step = args.crossval_steps
    cv_folds = args.crossval_folds

    # ------------------------------------- init -------------------------------------
    seed = 42

    job_index = os.environ.get("SLURM_ARRAY_TASK_ID")  
    print(f"The job index is: {job_index}")
    job_index = int(job_index)

    # ------------------------------------- read data -------------------------------------
    cv_embedding_dir = f"{dir_results}/cv_embeddings"
    with open(f"{cv_embedding_dir}/W_H_loc_scale_{job_index}.npy", "rb") as f:
        W_loc = np.load(f)
        W_scale = np.load(f)
        H_loc = np.load(f)
        H_scale = np.load(f)
        f.close()
    print(f"Reading posterior latent variables from {cv_embedding_dir}/W_H_loc_scale_{job_index}.npy")

    with open(f"{dir_results}/matrices.npy", "rb") as f:
        orders = np.load(f)
        n_obs_pre = np.load(f)  # since the saved n_obs is the original one, n_ons_pre = n_obs
        batch_sizes = np.load(f)
        start_row = np.load(f)
        stop_row = np.load(f)
        ranks = np.load(f)
        f.close()

    # Calculate the corresponding dimension and fold number from the job index in the array
    n_dims_options = np.arange(start=cv_dims[0], stop=cv_dims[1], step=cv_step)
    didx = (job_index-1)//cv_folds
    n_dims = n_dims_options[didx]
    fidx = (job_index-1) % cv_folds
    print(f"The dimension is: {str(n_dims)}")
    print(f"The fold is: {fidx}")

    n_batch = n_obs_pre.shape[0]
    fold_path = f"{dir_results}/folds/fold_{fidx}.pickle"
    fold = pickle.load(open(fold_path, "rb"))

    # ------------------------------------- main -------------------------------------
    num_samples = 1000
    #mae = False  # test the likelihood way to do CV

    np.random.seed(seed)
    W_draws = np.random.normal(W_loc, W_scale, size=(num_samples, W_loc.shape[0], W_loc.shape[1]))
    np.random.seed(seed)
    H_draws = np.random.normal(H_loc, H_scale, size=(num_samples, H_loc.shape[0], H_loc.shape[1]))
    X_draws = np.matmul(W_draws, H_draws[:, :, :ranks.shape[1]])  # subset to only metabolites to save memory
    rank_hat_draws = posterior_prediction_3D(X_draws, n_batch, start_row, stop_row)
    rank_hat_mean = np.mean(rank_hat_draws, axis=0)
    score = cv_score_mae_met_rna(rank_hat_mean, ranks, n_batch, start_row, stop_row, fold, n_obs_pre)

    # Write cv scores
    with open(f"{dir_results}/scores.txt", "a") as f:  # "a": Append Only
        f.write(str(didx) + " " + str(fidx) + " " + str(score) + "\n")  # remove 30 later
        f.close()
    with open(f"{dir_results}/cv_folds_scores.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow([n_dims, fidx, score])

    # Record the finish time for parallel jobs
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("The time of writing scores =", current_time)
    print("The total cv_score is: ", score)
    print(f"End of job {job_index}.")
