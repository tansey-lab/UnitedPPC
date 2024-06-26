from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import argparse
from Performance_Benchmarking.scripts.data_processing import *
import time


def lasso_cv(seed, X_train, y_train, iteration_dir, job_index):
    # (0) Roughly explore the effects of alpha
    alphas = np.linspace(0.01, 100, 10)
    lasso = Lasso(max_iter=1000, random_state=seed)
    coefs = []

    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X_train, y_train)
        coefs.append(lasso.coef_)

    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('Standardized Coefficients')
    plt.title('Lasso coefficients as a function of alpha')
    plt.savefig(f'{iteration_dir}/plots/alpha_function_{job_index}.pdf')
    plt.close()

    # (1) Lasso with 5 fold cross-validation
    start_time = time.time()
    model = LassoCV(cv=5, random_state=seed, max_iter=1000, verbose=True, n_jobs=-1)
    # Fit model
    model.fit(X_train, y_train)
    print(f'Best alpha chosen by cross validation: {model.alpha_}')
    fit_time = time.time() - start_time

    # (2) Plot the cross-validation score as a function of alpha
    plt.semilogx(model.alphas_, model.mse_path_, linestyle=":")
    plt.plot(
        model.alphas_,
        model.mse_path_.mean(axis=-1),
        color="black",
        label="Average across the folds",
        linewidth=2,
    )
    plt.axvline(model.alpha_, linestyle="--", color="black", label="alpha: CV estimate")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Mean square error")
    plt.legend()
    plt.title(
        f"Mean square error on each fold: coordinate descent (train time: {fit_time:.2f}s)"
    )
    plt.savefig(f'{iteration_dir}/plots/lasso_cv_{job_index}.pdf')
    plt.close()

    return model


if __name__ == "__main__":
    ############################################################# Parse command-line options & arguments
    # Create a parser
    parser = argparse.ArgumentParser(description="Impute metabolomics data from transcriptomics data",
                                  formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-id', '--iteration_dir', help="iteration_dir", required=False, type=str)
    args = parser.parse_args()
    iteration_dir = args.iteration_dir

    seed = 123
    cv = True
    # Read in data
    rna_data = pd.read_csv(f'{iteration_dir}/rna_data.csv', index_col=0)
    with open(f'{iteration_dir}/data_for_lasso_run.npy', 'rb') as f:
        y_test = np.load(f)
        X_test = np.load(f)
        y_train = np.load(f)
        X_train = np.load(f)
        test_feature_indices = np.load(f)
        f.close()

    # Pick a metabolite to predict according to the job index
    # the total number of job indices is the number of available metabolites for testing
    # job_index = os.environ.get("LSB_JOBINDEX")  # LSF: Use $LSB_JOBINDEX in the shell script, don't include the $ in py file
    job_index = os.environ.get("SLURM_ARRAY_TASK_ID")  # SLURM: get the SLURM_ARRAY_TASK_ID from the environment
    print('The job index is: ', job_index)
    job_index = int(job_index)

    y_test = y_test[:, job_index-1]
    y_train = y_train[:, job_index-1]  # with nan values
    # get the sample indices with labels
    train_sample_indices = np.where(~np.isnan(y_train))[0]  # get the indices of the features that are not nan
    y_train = y_train[train_sample_indices]
    X_train = X_train[train_sample_indices, :]


    # -------------------------------------- Lasso Regression ----------------------------------
    if cv:
        model = lasso_cv(seed, X_train, y_train, iteration_dir, job_index)
        alpha = model.alpha_
    else:
        alpha = 0

    # (3) Use the best alpha to fit the model again
    lasso_best = Lasso(alpha=alpha, max_iter=1000, random_state=seed)  # linear model with lasso regularization
    lasso_best.fit(X_train, y_train)

    # (4) Save and Plot the coefficients of the best model
    best_coefs = pd.DataFrame(lasso_best.coef_, index=rna_data.columns, columns=["coefficient"])
    best_coefs.to_csv(f'{iteration_dir}/embeddings/best_coefs_{job_index}.csv')

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(f"Number of non-zero coefficients: {non_zero}")

    if non_zero > 0:
        non_zero_coefs = best_coefs.query("coefficient != 0")
        coef_order = non_zero_coefs.abs().sort_values("coefficient").index

        _, ax = plt.subplots(figsize=(6, 8))
        non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
        ax.set_xlabel("coefficient")
        ax.grid(True)
        plt.savefig(f'{iteration_dir}/plots/best_coefs_{job_index}.pdf')
        plt.close()

    # (5) Predict on the test set using the best model and evaluate the performance
    y_pred = lasso_best.predict(X_test)
    with open(f'{iteration_dir}/embeddings/met_predicted_{job_index}.npy', 'wb') as f:  # 'wb': write as binary
        np.save(f, y_pred)  # save arrays in numpy binary .npy files
        f.close()





