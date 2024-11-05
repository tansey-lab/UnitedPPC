from datetime import datetime
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
import pyro.poutine as poutine
import tqdm
from UnitedMet.impute_met.utils import gumbel_sampling_3D, smart_perm_2D
from UnitedMet.impute_met.utils import count_obs, order_and_rank
from Performance_Benchmarking.scripts.utils import re_rank_2D

def generate_pyro_data(data, n_dims, n_batch, batch_index_vector):
    """
    This function is the same as 'generate_stan_data_across' function in MetabolicModel directory, but
    it is renamed here to match the pyro topic.
    """
    N = data.shape[0]  # number of samples
    J = data.shape[1]  # number of features
    K = n_dims
    # Record how many values in each column were fully observed (i.e. value is recorded and not left-censored); n_obs = 0 means the column is full of missing values
    # 0 in n_obs means that this metabolite is missing (all NA) in that batch
    censor_indicator = np.where(np.isnan(data), 1, 0)  # 1 means censored, 0 means not censored
    n_obs = count_obs(data, n_batch, J, batch_index_vector)
    orders, ranks = order_and_rank(data, n_obs, N, J, n_batch, batch_index_vector)
    return N, J, K, n_obs, orders, ranks, censor_indicator

class PlackettLuce_2D(TorchDistribution):
    """
        Plackett-Luce distribution for 2D permutation matrix (logits)
    """
    arg_constraints = {"logits": constraints.real, "n_obs": constraints.nonnegative_integer}
    def __init__(self, logits, n_obs):
        # last dimension is for scores of plackett luce
        self.logits = logits
        self.n_obs = n_obs
        self.size = self.logits.size()
        super(PlackettLuce_2D, self).__init__()

    def sample(self, num_samples=1):
        # sample permutations using Gumbel-max trick to avoid cycles
        with torch.no_grad():
            u = torch.distributions.utils.clamp_probs(torch.rand_like(self.logits))
            z = self.logits - torch.log(-torch.log(u))
            samples = torch.sort(z, descending=True, stable=True, dim=0)[1]  # return the indices of the sorted values
        return samples

    def log_prob(self, orders):
        # orders (2 d permutation, the indices of the sorted values from largest to smallest)
        assert self.size == orders.size()
        logits = smart_perm_2D(self.logits, orders)
        logp_matrix = (logits - torch.flip(torch.logcumsumexp(torch.flip(logits, dims=(0, )), dim=0), dims=(0, )))
        # get mask for observed values
        mask = torch.arange(orders.shape[0]).unsqueeze(1).expand(orders.shape[0], orders.shape[1]) < \
               self.n_obs.view(1, orders.shape[1])
        # sum over the observed values in all columns of samples
        logp = logp_matrix[mask].sum()
        return logp

class PlackettLuce_2D_weighted(TorchDistribution):
    """
        Plackett-Luce distribution for 2D permutation matrix (logits)
    """
    arg_constraints = {"logits": constraints.real, "n_obs": constraints.nonnegative_integer}
    def __init__(self, logits, n_obs, n_mets, n_genes):
        # last dimension is for scores of plackett luce
        self.logits = logits
        self.n_obs = n_obs
        self.n_mets = n_mets
        self.n_genes = n_genes
        self.size = self.logits.size()
        super(PlackettLuce_2D_weighted, self).__init__()

    def sample(self, num_samples=1):
        # sample permutations using Gumbel-max trick to avoid cycles
        with torch.no_grad():
            u = torch.distributions.utils.clamp_probs(torch.rand_like(self.logits))
            z = self.logits - torch.log(-torch.log(u))
            samples = torch.sort(z, descending=True, stable=True, dim=0)[1]  # return the indices of the sorted values
        return samples

    def log_prob(self, orders):
        # orders (2 d permutation, the indices of the sorted values from largest to smallest)
        assert self.size == orders.size()
        logits = smart_perm_2D(self.logits, orders)
        logp_matrix = (logits - torch.flip(torch.logcumsumexp(torch.flip(logits, dims=(0, )), dim=0), dims=(0, )))
        # get mask for observed values
        mask = torch.arange(orders.shape[0]).unsqueeze(1).expand(orders.shape[0], orders.shape[1]) < \
               self.n_obs.view(1, orders.shape[1])
        # create the weights matrix for metabolites and genes
        weights = torch.zeros_like(logp_matrix)
        weights[:, :self.n_mets] = 1  # metabolites
        weights[:, self.n_mets:] = self.n_mets/self.n_genes  # genes
        weighted_logp_matrix = logp_matrix * weights
        # sum over the observed values in all columns of samples
        logp = weighted_logp_matrix[mask].sum()
        return logp

def run_pyro_svi(N, J, K, n_batch, start_row, stop_row, n_obs, orders, n_steps=2000, lr=0.001):
    orders = torch.tensor(orders)
    n_obs = torch.tensor(n_obs)
    pyro.set_rng_seed(42)
    pyro.clear_param_store()

    # 2D Plackett Luce distribution version
    # torch.full((K, J), 10)
    def model(N, J, K, n_batch, start_row, stop_row, n_obs, orders):
        W = pyro.sample('W', dist.Normal(torch.zeros(N, K), torch.ones(N, K)).to_event(2))
        H = pyro.sample('H', dist.Normal(torch.zeros(K, J), torch.ones(K, J)).to_event(2))
        X = torch.mm(W, H)

        for b in range(n_batch):
            X_temp = X[start_row[b]:stop_row[b], :]
            temp_order = orders[start_row[b]:stop_row[b], :]
            pyro.sample("R_{}".format(b), PlackettLuce_2D(X_temp, n_obs[b, :]), obs=temp_order)

    # pyro.render_model(model, model_args=(N, J, K, n_batch, start_row, stop_row, n_obs, orders), filename="model.pdf")
    guide = AutoNormal(poutine.block(model, expose=['W', 'H']))
    # run inference
    optimizer = Adam({"lr": lr, 'betas': [0.95, 0.999]})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    tol_e = 0.01
    loss_list = []
    for step in tqdm.trange(n_steps):
        loss = svi.step(N, J, K, n_batch, start_row, stop_row, n_obs, orders)
        loss_list.append(loss)
        if step % 100 == 0:
            print("step {}: loss = {}".format(step, loss))
        if step > 0:
            if (np.abs(loss - loss_list[-2]) <= tol_e):
                print("Converged at step {}".format(step))
                break

    W_loc = pyro.param("AutoNormal.locs.W").detach().numpy()
    W_scale = pyro.param("AutoNormal.scales.W").detach().numpy()
    H_loc = pyro.param("AutoNormal.locs.H").detach().numpy()
    H_scale = pyro.param("AutoNormal.scales.H").detach().numpy()

    return W_loc, W_scale, H_loc, H_scale, loss_list

def run_pyro_svi_weighted(N, J, K, n_batch, start_row, stop_row, n_obs, orders, n_mets, n_genes, n_steps=2000, lr=0.001):
    orders = torch.tensor(orders)
    n_obs = torch.tensor(n_obs)
    pyro.set_rng_seed(42)
    pyro.clear_param_store()

    # 2D Plackett Luce distribution version
    # torch.full((K, J), 10)
    def model(N, J, K, n_batch, start_row, stop_row, n_obs, orders, n_mets, n_genes):
        W = pyro.sample('W', dist.Normal(torch.zeros(N, K), torch.ones(N, K)).to_event(2))
        H = pyro.sample('H', dist.Normal(torch.zeros(K, J), torch.ones(K, J)).to_event(2))
        X = torch.mm(W, H)

        for b in range(n_batch):
            X_temp = X[start_row[b]:stop_row[b], :]
            temp_order = orders[start_row[b]:stop_row[b], :]
            pyro.sample("R_{}".format(b), PlackettLuce_2D_weighted(X_temp, n_obs[b, :], n_mets, n_genes), obs=temp_order)

    # pyro.render_model(model, model_args=(N, J, K, n_batch, start_row, stop_row, n_obs, orders), filename="model.pdf")
    guide = AutoNormal(poutine.block(model, expose=['W', 'H']))
    # run inference
    optimizer = Adam({"lr": lr, 'betas': [0.95, 0.999]})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    tol_e = 0.01
    loss_list = []
    for step in tqdm.trange(n_steps):
        loss = svi.step(N, J, K, n_batch, start_row, stop_row, n_obs, orders, n_mets, n_genes)
        loss_list.append(loss)
        if step % 100 == 0:
            print("step {}: loss = {}".format(step, loss))
        if step > 0:
            if (np.abs(loss - loss_list[-2]) <= tol_e):
                print("Converged at step {}".format(step))
                break

    W_loc = pyro.param("AutoNormal.locs.W").detach().numpy()
    W_scale = pyro.param("AutoNormal.scales.W").detach().numpy()
    H_loc = pyro.param("AutoNormal.locs.H").detach().numpy()
    H_scale = pyro.param("AutoNormal.scales.H").detach().numpy()

    return W_loc, W_scale, H_loc, H_scale, loss_list


def svi_loss(loss_list, plots_dir):
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.plot(loss_list)
    plt.xticks(np.arange(0, len(loss_list), 500))
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.grid()
    plt.title('ELBO loss in SVI')
    plt.savefig(f'{plots_dir}/elbo_loss.pdf')
    plt.close()

def posterior_prediction_3D(X_draws, n_batch, start_row, stop_row):
    rank_hat_draws = np.full([X_draws.shape[0], X_draws.shape[1], X_draws.shape[2]], np.nan)
    for b in range(n_batch):
        X_temp = X_draws[:, start_row[b]:stop_row[b], :]
        # rank_temp = np.apply_along_axis(categorical_sampling, 1, X_temp)
        order_temp = gumbel_sampling_3D(X_temp)
        # rank (largest item has rank 0, second largest has rank 1, etc.) pay attention to the axis
        rank_hat_draws[:, start_row[b]:stop_row[b], :] = order_temp.argsort(axis=1, kind='stable')
    return rank_hat_draws

def pyro_posterior(W_loc, W_scale, H_loc, H_scale, n_batch, start_row, stop_row, ve, met_names, embedding_dir, seed, num_samples=1000):
    # generate posterior samples
    np.random.seed(seed)
    W_draws = np.random.normal(W_loc, W_scale, size=(num_samples, W_loc.shape[0], W_loc.shape[1]))
    np.random.seed(seed)
    H_draws = np.random.normal(H_loc, H_scale, size=(num_samples, H_loc.shape[0], H_loc.shape[1]))
    X_draws = np.matmul(W_draws, H_draws[:, :, 0:len(met_names)])  # subset H and X to only metabolites to save memory
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logging.warning(f'The time of matrix multiplication being done is = {current_time}')

    rank_hat_draws = posterior_prediction_3D(X_draws, n_batch, start_row, stop_row)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logging.warning(f'The time of posterior predictioin being done is = {current_time}')
    if ve:
        np.save(f'{embedding_dir}/W_draws.npy', W_draws)
        np.save(f'{embedding_dir}/H_draws.npy', H_draws)
        np.save(f'{embedding_dir}/rank_hat_draws_met.npy',
                rank_hat_draws)  # save arrays in numpy binary .npy files

    rank_hat_draws_mean = np.mean(rank_hat_draws, axis=0)
    rank_hat_draws_std = np.std(rank_hat_draws, axis=0)
    np.save(f'{embedding_dir}/rank_hat_draws_mean_met.npy', rank_hat_draws_mean)
    np.save(f'{embedding_dir}/rank_hat_draws_std_met.npy', rank_hat_draws_std)

    return rank_hat_draws, rank_hat_draws_mean, rank_hat_draws_std

def final_impute_met(embedding_dir, results_dir, batch_sizes, rank_hat_mean, rank_hat_std):
    '''
    get metabolite ranks in the flipped way
    in pyro modeling rank 0 means the largest value, rank 1 means the second largest value, etc.
    but in the downstream analysis, it's more intuitive to have rank 0 mean the smallest value, rank 1 means the second smallest value, etc
    '''
    # subset predictions to single-modality target dataset
    imputation_batch_size = batch_sizes[0]
    rank_hat_draws_mean = rank_hat_mean[:imputation_batch_size, :]
    rank_hat_draws_std = rank_hat_std[:imputation_batch_size, :] / imputation_batch_size
    # rerank metabolite ranks
    rank_hat_draws_mean = rank_hat_draws_mean.argsort(
        axis=0, kind='stable').argsort(axis=0, kind='stable')
    # flip the ranks
    rank_hat_draws_mean = re_rank_2D(rank_hat_draws_mean)
    rank_hat_draws_mean = rank_hat_draws_mean / rank_hat_draws_mean.shape[0]  # transformed all ranks to [0,1)
    normalized_data_met = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv',
                                      index_col=0, header=0).iloc[:imputation_batch_size, :rank_hat_draws_mean.shape[1]]
    imputed_mean_met = pd.DataFrame(rank_hat_draws_mean, index=normalized_data_met.index,
                                    columns=normalized_data_met.columns)
    imputed_std_met = pd.DataFrame(rank_hat_draws_std, index=normalized_data_met.index,
                                   columns=normalized_data_met.columns)
    imputed_mean_met.to_csv(f'{results_dir}/target_imputed_met_mean.csv')
    imputed_std_met.to_csv(f'{results_dir}/target_imputed_met_std.csv')
