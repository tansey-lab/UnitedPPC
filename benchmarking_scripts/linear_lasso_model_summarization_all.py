import pandas as pd
from scipy.stats import spearmanr
import argparse
from statsmodels.stats.multitest import multipletests
from Performance_Benchmarking.scripts.data_processing import *

if __name__ == "__main__":
    # Create a parser
    parser = argparse.ArgumentParser(description="Impute metabolomics data from transcriptomics data",
                                  formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-id', '--iteration_dir', help="iteration_dir", required=False, type=str)
    parser.add_argument('-i', '--iteration', help='which iterations it is', required=False,
                        default=1, type=int)
    args = parser.parse_args()
    iteration_dir = args.iteration_dir
    iteration = args.iteration

    # Read in data
    met_data = pd.read_csv(f'{iteration_dir}/met_data.csv', index_col=0)
    rna_data = pd.read_csv(f'{iteration_dir}/rna_data.csv', index_col=0)
    with open(f'{iteration_dir}/data_for_lasso_run.npy', 'rb') as f:
        y_test = np.load(f)
        X_test = np.load(f)
        y_train = np.load(f)
        X_train = np.load(f)
        test_feature_indices = np.load(f)
        f.close()

    # --------------------- Summarize the best coefficient matrix ---------------------
    n_metabolites = len(test_feature_indices)
    n_genes = rna_data.shape[1]
    coef_matrix = pd.DataFrame(np.zeros((n_genes, n_metabolites)), index=rna_data.columns,
                               columns=met_data.iloc[:, test_feature_indices].columns)
    for i in range(n_metabolites):
        best_coefs = pd.read_csv(f'{iteration_dir}/embeddings/best_coefs_{i+1}.csv', index_col=0)
        coef_matrix.iloc[:, i] = best_coefs['coefficient']
    coef_matrix.to_csv(f'{iteration_dir}/best_coef_matrix.csv')

    # --------------------- Summarize predicted metabolomics results ---------------------
    y_pred_matrix = np.zeros((y_test.shape[0], y_test.shape[1]))
    for i in range(n_metabolites):
        with open(f'{iteration_dir}/embeddings/met_predicted_{i+1}.npy', 'rb') as f:
            y_pred = np.load(f)
            f.close()
        y_pred_matrix[:, i] = y_pred

    with open(f'{iteration_dir}/met_predicted.npy', 'wb') as f:  # 'wb': write as binary
        np.save(f, y_pred_matrix)  # save arrays in numpy binary .npy files
        f.close()

    # --------------------- Compare y_pred_matrix and y_test (compare true and predicted values) ---------------------
    rho_matrix = pd.DataFrame(np.zeros((n_metabolites, 2)),
                              index=met_data.iloc[:, test_feature_indices].columns, columns=['rho', 'pval'])
    for i in range(n_metabolites):
        y_pred = y_pred_matrix[:, i]
        y_true = y_test[:, i]
        rho_matrix.loc[rho_matrix.index[i], 'rho'] = spearmanr(y_pred, y_true).statistic
        rho_matrix.loc[rho_matrix.index[i], 'pval'] = spearmanr(y_pred, y_true).pvalue

    # FDR correction
    # replace all p-values that are NaN with 1
    rho_matrix['rho'] = rho_matrix['rho'].fillna(0)
    rho_matrix['pval'] = rho_matrix['pval'].fillna(1)
    rho_matrix['p_adj'] = multipletests(pvals=rho_matrix['pval'], method="fdr_bh", alpha=0.1)[1]
    rho_matrix['sig'] = rho_matrix['p_adj'] < 0.1
    rho_matrix['iteration'] = iteration

    rho_matrix.to_csv(f'{iteration_dir}/actual_pred_rho_matrix.csv')  # final output

