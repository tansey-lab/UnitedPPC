import numpy as np
import pickle
import os
import torch

from UnitedPPC.utils import parsearg_utils
from UnitedPPC.pyro_model import run_pyro_svi, run_pyro_svi_weighted

if __name__ == "__main__":
    # ------------------------------------- argparse -------------------------------------
    args = parsearg_utils().parse_args()
    
    # output options
    dir_results = args.results_dir
    # cross-validation options
    cv_dims = args.crossval_dims
    cv_step = args.crossval_steps
    cv_folds = args.crossval_folds
    # pyro options
    n_steps = args.n_steps
    lr = args.learning_rate

    # ------------------------------------- init -------------------------------------
    seed = 42

    job_index = os.environ.get("SLURM_ARRAY_TASK_ID")  
    print(f"The job index is: {job_index}")
    job_index = int(job_index)

    # ------------------------------------- main -------------------------------------
    with open(f"{dir_results}/matrices.npy", "rb") as f:
        orders = np.load(f)
        n_obs = np.load(f)
        batch_sizes = np.load(f)
        start_row = np.load(f)
        stop_row = np.load(f)
        ranks = np.load(f)
        n_mets = np.load(f)
        n_genes = np.load(f)
        f.close()

    # calculate the corresponding dimension and fold number from the job index in the array
    n_dims_options = np.arange(start=cv_dims[0], stop=cv_dims[1], step=cv_step)
    didx = (job_index-1)//cv_folds
    n_dims = n_dims_options[didx]
    fidx = (job_index-1) % cv_folds
    print(f"The dimension is: {n_dims}")
    print(f"The fold is: {fidx}")

    n_batch = n_obs.shape[0]
    fold_path = f"{dir_results}/folds/fold_{fidx}.pickle"
    fold = pickle.load(open(fold_path, "rb"))
    # save the original n_obs of the training data
    n_obs_pre = np.copy(n_obs)
    for b in range(n_batch):
        # reset the n_obs according to the fold!!
        n_obs[b, fold[b]] = 0  
    # number of samples
    N = orders.shape[0]
    # number of features
    J = orders.shape[1]
    K = n_dims

    torch.manual_seed(seed)
    W_loc, W_scale, H_loc, H_scale, loss_list = run_pyro_svi(
        N, J, K, n_batch, start_row, stop_row, n_obs, orders, n_steps, lr
    )
    cv_embedding_dir = f"{dir_results}/cv_embeddings"
    if not os.path.exists(cv_embedding_dir):
        os.makedirs(cv_embedding_dir)
    # "wb": write as binary
    with open(f"{cv_embedding_dir}/W_H_loc_scale_{job_index}.npy", "wb") as f:
        np.save(f, W_loc)
        np.save(f, W_scale)
        np.save(f, H_loc)
        np.save(f, H_scale)
        f.close()
