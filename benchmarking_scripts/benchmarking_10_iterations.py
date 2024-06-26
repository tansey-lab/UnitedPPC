import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import scipy.stats
import seaborn as sns
from scipy.stats import chi2_contingency

if __name__ == "__main__":
    # -----------------------------Summarize results across 10 iterations-----------------------------
    # Lasso
    file_path = "/data1/reznike/xiea1"
    file_name = "actual_pred_rho_matrix.csv"
    results_dir = f'{file_path}/MetabolicModel/results_RNA_lasso_met'
    sub_dir = ['CPTAC', 'CPTAC_val', 'RC18', 'RC20']
    n_iter = 10

    for i in sub_dir:
        rho_matrix_list = []
        for j in range(n_iter):
            rho_matrix_list.append(pd.read_csv(f'{results_dir}/{i}/{j}/{file_name}', header=0))
        by_iter_rho = pd.concat(rho_matrix_list, axis=0)
        by_iter_rho = by_iter_rho.rename(columns={by_iter_rho.columns[0]: 'feature'})
        g_median_by_feature = (by_iter_rho
                               .assign(z_score=lambda dataframe: dataframe['rho'].map(lambda rho: np.arctanh(rho)))
                               .groupby(["feature"], group_keys=False))

        # compute median z-score and inverse z-transform to obtain median rho
        median_rho_feature = (pd.DataFrame({"median_z": g_median_by_feature.median()['z_score'],
                                            "sig_in": g_median_by_feature.sum()['sig'] / g_median_by_feature.size()})
                              .assign(median_rho=lambda dataframe: dataframe['median_z'].map(lambda z: np.tanh(z)))
                              # flag well-predicted metabolites
                              .assign(sig_in_most=(lambda dataframe: dataframe['sig_in']
                                                   # note that if a flag for "well-predicted metabolite" is desired, a condition for positive rho must be added here
                                                   .map(lambda sig_in: True if sig_in >= 0.9 else False)))
                              .reset_index()
                              )
        median_rho_feature.to_csv(f'{results_dir}/{i}/median_rho_feature.csv', index=False)

    # UnitedMet
    file_path = "/data1/reznike/xiea1"
    file_name = "median_rho_feature.csv"
    results_dir = f'{file_path}/MetabolicModel/results_RNA_ccRCC_benchmarking_met_weighted'
    sub_dir = ['CPTAC', 'CPTAC_val', 'RC18', 'RC20']
    n_iter = 10

    for i in sub_dir:
        rho_matrix_list = []
        for j in range(n_iter):
            rho_matrix_list.append(pd.read_csv(f'{results_dir}/{i}/{j}/{file_name}', header=0, index_col=0))
        by_iter_rho = pd.concat(rho_matrix_list, axis=0)
        # delete columns "ave_ave_std", "censor_prop", "sig_in_most", "ave_ave_se"
        g_median_by_feature = (by_iter_rho
                               .assign(z_score=lambda dataframe: dataframe['median_rho'].map(lambda rho: np.arctanh(rho)))
                               .groupby(["f_test_prop", "feature"], group_keys=False))

        # compute median z-score and inverse z-transform to obtain median rho
        median_rho_feature = (pd.DataFrame({"median_z": g_median_by_feature.median()['median_z'],
                                            "sig_in": g_median_by_feature.sum()['sig_in'] / g_median_by_feature.size()})
                              .assign(median_rho=lambda dataframe: dataframe['median_z'].map(lambda z: np.tanh(z)))
                              # flag well-predicted metabolites
                              .assign(sig_in_most=(lambda dataframe: dataframe['sig_in']
                                                   # note that if a flag for "well-predicted metabolite" is desired, a condition for positive rho must be added here
                                                   .map(lambda sig_in: True if sig_in >= 0.9 else False)))
                              .reset_index()
                              )
        median_rho_feature.to_csv(f'{results_dir}/{i}/median_rho_feature.csv', index=False)

    # MIRTH

    # ------------------------------------- Benchmarking and comparing ----------------------------------
    # ------------------------------------- Unitedmet vs Lasso -------------------------------------------
    file_path = "/data1/reznike/xiea1"
    sub_dir = ['CPTAC', 'CPTAC_val', 'RC18', 'RC20']
    for i in sub_dir:
        df_lasso = pd.read_csv(f'{file_path}/MetabolicModel/results_RNA_lasso_met/{i}/median_rho_feature.csv', header=0, index_col=0)
        df_lasso = df_lasso.rename(columns={'median_rho': 'rho_lasso'})  # avoid duplicate column names
        df_lasso = df_lasso.rename(columns={'sig_in_most': 'sig_lasso'})
        df_unitedmet = pd.read_csv(f'{file_path}/MetabolicModel/results_RNA_ccRCC_benchmarking_met_weighted/{i}/median_rho_feature.csv', header=0, index_col='feature')
        # concatenate the three dataframes
        df = pd.concat([df_lasso[['rho_lasso']], df_unitedmet[["median_rho"]]], axis=1, join='inner')

        df['high_in_unitedmet'] = df['median_rho'] > df['rho_lasso']
        print(f'{i}: {sum(df["high_in_unitedmet"])/len(df)*100}%')        # create a new column 'best_method' if the row is largest in UnitedMet, then the value is UnitedMet, if the row is largest in MIRTH, then the value in MIRTH, else the value is Lasso

        df['best_method'] = df[['rho_lasso', 'median_rho']].idxmax(axis=1)
        print(df['best_method'].value_counts()/df.shape[0]*100)

        # plot box plot of rho and median rho
        sns.boxplot(data=df[['rho_lasso', 'median_rho']], width=0.5)
        sns.stripplot(data=df[['rho_lasso', 'median_rho']], jitter=True, color=".3", alpha=0.5)
        plt.xticks([0, 1], ["Lasso", "UnitedMet"])
        plt.xlabel('Methods')
        plt.ylabel('Spearman correlation')
        plt.title(i)
        plt.savefig(f'{file_path}/MetabolicModel/results_RNA_lasso_met/{i}/plots/benchmark_box_plot.pdf')
        plt.close()

        # perform paired wilcoxon test (whitney manny u test)
        diff = np.mean((df['median_rho'].to_numpy())-(df['rho_lasso'].to_numpy()))
        pvalue = scipy.stats.wilcoxon(df['rho_lasso'], df['median_rho']).pvalue
        print(f'{i}: diff={diff}, pvalue={pvalue}')

        # bar plot compare the number of features with rho > 0 and significant
        data = pd.concat([df_lasso[['rho_lasso']], df_lasso[['sig_lasso']],
                               df_unitedmet[["median_rho"]], df_unitedmet[["sig_in_most"]]], axis=1, join='inner')
        # calculate the number of features with rho > 0 and sig == ture
        number_lasso = data.loc[(data['rho_lasso'] > 0) & (data['sig_lasso'] == True)].shape[0]
        number_unitedmet = data.loc[(data['median_rho'] > 0) & (data['sig_in_most'] == True)].shape[0]
        # plot
        plt.bar([0, 1], [number_lasso, number_unitedmet], width=0.5)
        plt.xticks([0, 1], ["Lasso", "UnitedMet"])
        plt.xlabel('Methods')
        plt.ylabel('# well-predicted features')
        plt.title(i)
        plt.savefig(f'{file_path}/MetabolicModel/results_RNA_lasso_met/{i}/plots/benchmark_bar_plot.pdf')
        plt.close()


    # --------------------------- Unitedmet vs Lasso vs MIRTH --------------------------------
    file_path = "/data1/reznike/xiea1"
    sub_dir = ['CPTAC', 'CPTAC_val', 'RC18', 'RC20']
    for i in sub_dir:
        df_lasso = pd.read_csv(f'{file_path}/MetabolicModel/results_RNA_lasso_met/{i}/median_rho_feature.csv',
                               header=0, index_col=0)
        df_lasso = df_lasso.rename(columns={'median_rho': 'rho_lasso'})  # avoid duplicate column names
        df_lasso = df_lasso.rename(columns={'sig_in_most': 'sig_lasso'})
        df_mirth = pd.read_csv(f'{file_path}/MIRTH/results_MET_RNA_4_met/{i}/median_rho_feature.csv', header=0, index_col='feature')
        df_mirth = df_mirth.rename(columns={'median_rho': 'rho_mirth'})  # avoid duplicate column names
        df_mirth = df_mirth.rename(columns={'sig_in_most': 'sig_mirth'})
        df_unitedmet = pd.read_csv(
            f'{file_path}/MetabolicModel/results_RNA_ccRCC_benchmarking_met_weighted/{i}/median_rho_feature.csv', header=0,
            index_col='feature')
        # concatenate the three dataframes
        df = pd.concat([df_lasso[['rho_lasso']], df_mirth[['rho_mirth']], df_unitedmet[["median_rho"]]], axis=1, join='inner')
        df['high_in_unitedmet'] = df['median_rho'] > df['rho_lasso']
        print(
            f'{i}: {sum(df["high_in_unitedmet"]) / len(df) * 100}%')  # create a new column 'best_method' if the row is largest in UnitedMet, then the value is UnitedMet, if the row is largest in MIRTH, then the value in MIRTH, else the value is Lasso

        df['best_method'] = df[['rho_lasso', 'rho_mirth', 'median_rho']].idxmax(axis=1)
        print(df['best_method'].value_counts() / df.shape[0] * 100)

        # plot box plot of rho and median rho
        sns.boxplot(data=df[['rho_lasso', 'rho_mirth', 'median_rho']], width=0.5)
        sns.stripplot(data=df[['rho_lasso', 'rho_mirth', 'median_rho']], jitter=True, color=".3", alpha=0.5)
        plt.xticks([0, 1, 2], ["Lasso", "MIRTH", "UnitedMet"])
        plt.xlabel('Methods')
        plt.ylabel('Spearman correlation')
        plt.title(i)
        plt.savefig(f'{file_path}/MetabolicModel/results_RNA_lasso_met/{i}/plots/benchmark_box_plot_3.pdf')
        plt.close()

        # perform paired wilcoxon test (whitney manny u test)
        diff = np.mean((df['median_rho'].to_numpy()) - (df['rho_lasso'].to_numpy()))
        pvalue = scipy.stats.wilcoxon(df['rho_lasso'], df['median_rho']).pvalue
        print(f'{i}, Lasso vs UM: diff={diff}, pvalue={pvalue}')

        diff = np.mean((df['median_rho'].to_numpy())-(df['rho_mirth'].to_numpy()))
        pvalue = scipy.stats.wilcoxon(df['rho_mirth'], df['median_rho']).pvalue
        print(f'{i}, MIRTH vs UM: diff={diff}, pvalue={pvalue}')

        # bar plot compare the number of features with rho > 0 and significant
        data = pd.concat([df_lasso[['rho_lasso']], df_lasso[['sig_lasso']],
                          df_mirth[['rho_mirth']], df_mirth[['sig_mirth']],
                          df_unitedmet[["median_rho"]], df_unitedmet[["sig_in_most"]]], axis=1, join='inner')
        # calculate the number of features with rho > 0 and sig == ture
        number_lasso = data.loc[(data['rho_lasso'] > 0) & (data['sig_lasso'] == True)].shape[0]
        number_mirth = data.loc[(data['rho_mirth'] > 0) & (data['sig_mirth'] == True)].shape[0]
        number_unitedmet = data.loc[(data['median_rho'] > 0) & (data['sig_in_most'] == True)].shape[0]
        # plot
        plt.bar([0, 1, 2], [number_lasso, number_mirth, number_unitedmet], width=0.5)
        plt.xticks([0, 1, 2], ["Lasso", "MIRTH", "UnitedMet"])
        plt.xlabel('Methods')
        plt.ylabel('# well-predicted features')
        plt.title(i)
        plt.savefig(f'{file_path}/MetabolicModel/results_RNA_lasso_met/{i}/plots/benchmark_bar_plot_3.pdf')
        plt.close()

    # --------------------------- Unitedmet vs Lasso vs MIRTH (average rho) --------------------------------
        for i in sub_dir:
            df_lasso = pd.read_csv(
                f'{file_path}/MetabolicModel/results_RNA_lasso_met/{i}/median_rho_feature.csv',
                header=0, index_col=0)
            df_lasso = df_lasso.rename(columns={'median_rho': 'rho_lasso'})  # avoid duplicate column names
            df_lasso = df_lasso.rename(columns={'sig_in_most': 'sig_lasso'})
            df_mirth = pd.read_csv(f'{file_path}/MIRTH/results_MET_RNA_4_met/{i}/ave_rho_padj.csv', header=0,
                                   index_col='feature')
            df_unitedmet = pd.read_csv(
                f'{file_path}/MetabolicModel/results_RNA_ccRCC_benchmarking_met_weighted/{i}/median_rho_feature.csv',
                header=0,
                index_col='feature')


            # bar plot compare the number of features with rho > 0 and significant
            data = pd.concat([df_lasso[['rho_lasso']], df_lasso[['sig_lasso']],
                              df_mirth[['rho']], df_mirth[['sig']],
                              df_unitedmet[["median_rho"]], df_unitedmet[["sig_in_most"]]], axis=1, join='inner')
            # calculate the number of features with rho > 0 and sig == ture
            number_lasso = data.loc[(data['rho_lasso'] > 0) & (data['sig_lasso'] == True)].shape[0]
            number_mirth = data.loc[(data['rho'] > 0) & (data['sig'] == True)].shape[0]
            number_unitedmet = data.loc[(data['median_rho'] > 0) & (data['sig_in_most'] == True)].shape[0]
            # plot
            plt.bar([0, 1, 2], [number_lasso, number_mirth, number_unitedmet], width=0.5)
            plt.xticks([0, 1, 2], ["Lasso", "MIRTH", "UnitedMet"])
            plt.xlabel('Methods')
            plt.ylabel('# well-predicted features')
            plt.title(i)
            plt.savefig(f'{file_path}/MetabolicModel/results_RNA_lasso_met/{i}/plots/benchmark_bar_plot_3_mirth_ave.pdf')
            plt.close()