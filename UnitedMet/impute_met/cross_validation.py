import numpy as np
import matplotlib.pyplot as plt
from UnitedMet.impute_met.pyro_model import PlackettLuce_2D
import torch
from kneed import KneeLocator

def saving_data_for_cv(results_dir, orders, n_obs, batch_sizes, start_row, stop_row, ranks, n_mets, n_genes):
    with open(f'{results_dir}/matrices.npy', 'wb') as f:  # 'wb': write as binary
        np.save(f, orders)  # save arrays in numpy binary .npy files
        np.save(f, n_obs)
        np.save(f, batch_sizes)
        np.save(f, start_row)
        np.save(f, stop_row)
        np.save(f, ranks)
        np.save(f, n_mets)
        np.save(f, n_genes)
        f.close()


# FUNCTIONS FOR CV
def cv_score_mae_met_rna(rank_hat_mean, ranks, n_batch, start_row, stop_row, fold, n_obs_pre):
    """
    The difference between this function and the one in PL_across_cv.py is that I added a condition check to
    skip the empty fold of the target dataset, which will result in empty slices of rank and NaN of mae.
    """
    mae = 0
    for b in range(n_batch):
        if fold[b].shape[0] == 0:
            continue
        # get predicted ranks of the fold
        rank_hat_mean_test = rank_hat_mean[start_row[b]:stop_row[b], fold[b]]
        rerank_hat_mean_test = rank_hat_mean_test.argsort(
            axis=0, kind='stable').argsort(axis=0, kind='stable')  # re-rank the predicted values
        # if rank > n_obs, set rank = n_obs (get censored rank)
        rerank_hat_mean_test = np.where(
            rerank_hat_mean_test > n_obs_pre[b, fold[b]], n_obs_pre[b, fold[b]],
            rerank_hat_mean_test)
        ranks_test = ranks[start_row[b]:stop_row[b], fold[b]]
        mae += np.abs(rerank_hat_mean_test - ranks_test).mean()
    return mae

def summarize_cv_results(cv_dims, cv_step, n_folds, results_dir, plots_dir):
    n_dims_options = np.arange(start=cv_dims[0], stop=cv_dims[1], step=cv_step)
    scores = np.zeros((len(n_dims_options), n_folds))
    with open(f"{results_dir}/scores.txt", "r") as f:
        for line in f:
            sp = line.split()
            scores[int(sp[0]), int(sp[1])] = float(sp[2])  # mae
            #scores[int(sp[0]), int(sp[1])] = float(sp[2].replace('tensor(', '').rstrip(','))
            # likelihood (since scores are in tensor string formats)
        f.close()

    # get best params and one_se_rule params
    scores_means = np.mean(scores, axis=1)
    scores_se = np.std(scores, axis=1) / np.sqrt(n_folds)

    # find the knee point
    kneedle = KneeLocator(n_dims_options, scores_means, curve='convex', direction='decreasing')
    n_dims_knee = kneedle.knee
    knee_y = kneedle.knee_y
    print(f'The knee dims is {n_dims_knee}')
    plot_cv_scores(scores_means, scores_se, n_dims_options, n_dims_knee, knee_y, results_dir, plots_dir)

    return n_dims_knee


# plot the cv_score
def plot_cv_scores(scores_means, scores_se, n_dims_options, n_dims_knee, knee_y, results_dir, plots_dir):
    plt.rcParams['figure.figsize'] = [6, 6]
    plt.plot(n_dims_options, scores_means, color='black')
    plt.fill_between(n_dims_options, scores_means + 1 * scores_se, scores_means - 1 * scores_se, alpha=.1)
    plt.axvline(x=n_dims_knee, linestyle='dotted', linewidth=2.0)
    plt.scatter([n_dims_knee], [knee_y], marker='o', color='red', label='Elbow point')
    plt.legend()
    plt.xlabel("# embedding dimensions")  # add X-axis label
    plt.ylabel("fold-average MAE")  # add Y-axis label
    plt.title(results_dir.rsplit("/", 1)[-1])  # add title
    plt.grid()
    plt.savefig(f'{plots_dir}/cv_score.pdf')
    plt.close()


