import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import scipy.stats
import seaborn as sns
from scipy.stats import chi2_contingency

if __name__ == "__main__":
    # -----------------------------Benchmarking: Summarize 100% held-out results-----------------------------
    file_path = "/data1/reznike/xiea1"
    sub_dir = ['CPTAC', 'CPTAC_val', 'RC18', 'RC20']  # notice that CPTAC is the first one

    benchmark_bar_df = pd.DataFrame(columns=['dataset', 'method', 'well_predicted_percent'])
    for i in range(len(sub_dir)):
        df_lasso = pd.read_csv(f'{file_path}/MetabolicModel/results_RNA_lasso/{sub_dir[i]}/0/actual_pred_rho_matrix.csv', header=0, index_col=0)
        df_lasso = df_lasso.rename(columns={'rho': 'rho_lasso'})  # avoid duplicate column names
        df_lasso = df_lasso.rename(columns={'sig': 'sig_lasso'})
        df_mirth = pd.read_csv(f'{file_path}/MIRTH/results_MET_RNA_4_final/{sub_dir[i]}/ave_rho_padj_final.csv', header=0, index_col='feature')
        df_unitedmet = pd.read_csv(f'{file_path}/MetabolicModel/results_RNA_ccRCC/{sub_dir[i]}/median_rho_feature.csv', header=0, index_col='feature')
        # concatenate the three dataframes
        df = pd.concat([df_lasso[['rho_lasso']], df_mirth[['rho']], df_unitedmet[["median_rho"]]], axis=1, join='inner')
        # replace nan with 0
        df = df.fillna(0)
        df['high_in_unitedmet'] = df['median_rho'] > df['rho_lasso']
        print(f'{sub_dir[i]}: {sum(df["high_in_unitedmet"])/len(df)*100}%')        # create a new column 'best_method' if the row is largest in UnitedMet, then the value is UnitedMet, if the row is largest in MIRTH, then the value in MIRTH, else the value is Lasso

        df['best_method'] = df[['rho_lasso', 'rho', 'median_rho']].idxmax(axis=1)
        print(df['best_method'].value_counts()/df.shape[0]*100)

        # plot box plot of rho and median rho
        sns.boxplot(data=df[['rho_lasso', 'rho', 'median_rho']], width=0.5)
        sns.stripplot(data=df[['rho_lasso', 'rho', 'median_rho']], jitter=True, color=".3", alpha=0.5)
        plt.xticks([0, 1, 2], ["Lasso", "MIRTH", "UnitedMet"])
        plt.xlabel('Methods')
        plt.ylabel('Spearman correlation')
        plt.title(sub_dir[i])
        plt.savefig(f'{file_path}/MetabolicModel/results_benchmark/benchmark_box_plot_{sub_dir[i]}.pdf')
        plt.close()

        # perform paired wilcoxon test (whitney manny u test)
        diff = np.mean((df['median_rho'].to_numpy())-(df['rho_lasso'].to_numpy()))
        pvalue = scipy.stats.wilcoxon(df['rho_lasso'], df['median_rho']).pvalue
        print(f'{sub_dir[i]}, Lasso vs UM: diff={diff}, pvalue={pvalue}')

        diff = np.mean((df['median_rho'].to_numpy())-(df['rho'].to_numpy()))
        pvalue = scipy.stats.wilcoxon(df['rho'], df['median_rho']).pvalue
        print(f'{sub_dir[i]}, MIRTH vs UM: diff={diff}, pvalue={pvalue}')

        # bar plot compare the number of features with rho > 0 and significant
        data = pd.concat([df_lasso[['rho_lasso']], df_lasso[['sig_lasso']],
                               df_mirth[['rho']], df_mirth[['sig']],
                               df_unitedmet[["median_rho"]], df_unitedmet[["sig_in_most"]]], axis=1, join='inner')
        # fill all nan in rho_lasso with 0, and fill all nan in sig_lasso 0
        data['rho_lasso'] = data['rho_lasso'].fillna(0)
        data['sig_lasso'] = data['sig_lasso'].fillna(0)
        # calculate the number of features with rho > 0 and sig == ture
        number_lasso = data.loc[(data['rho_lasso'] > 0) & (data['sig_lasso'] == True)].shape[0]
        number_mirth = data.loc[(data['rho'] > 0) & (data['sig'] == True)].shape[0]
        number_unitedmet = data.loc[(data['median_rho'] > 0) & (data['sig_in_most'] == True)].shape[0]
        benchmark_bar_df = benchmark_bar_df._append({'dataset': sub_dir[i], 'method': 'Lasso', 'well_predicted_percent': number_lasso/data.shape[0]*100}, ignore_index=True)
        benchmark_bar_df = benchmark_bar_df._append({'dataset': sub_dir[i], 'method': 'MIRTH', 'well_predicted_percent': number_mirth/data.shape[0]*100}, ignore_index=True)
        benchmark_bar_df = benchmark_bar_df._append({'dataset': sub_dir[i], 'method': 'UnitedMet', 'well_predicted_percent': number_unitedmet/data.shape[0]*100}, ignore_index=True)
        # plot
        plt.bar([0, 1, 2], [number_lasso, number_mirth, number_unitedmet], width=0.5)
        plt.xticks([0, 1, 2], ["Lasso", "MIRTH", "UnitedMet"])
        plt.xlabel('Methods')
        plt.ylabel('# well-predicted features')
        plt.title(sub_dir[i])
        plt.savefig(f'{file_path}/MetabolicModel/results_benchmark/benchmark_bar_plot_{sub_dir[i]}.pdf')
        plt.close()
        benchmark_bar_df.to_csv(f'{file_path}/MetabolicModel/results_RNA_ccRCC/final_benchmark_bar_plot_df_final.csv')


    # get the big table of all the benchmarking results
    benchmark_df = pd.DataFrame(columns=['rho', 'sig', 'method', 'dataset'])
    for i in range(len(sub_dir)):
        df_lasso = pd.read_csv(f'{file_path}/MetabolicModel/results_RNA_lasso/{sub_dir[i]}/0/actual_pred_rho_matrix.csv', header=0, index_col=0)
        df_lasso['method'] = 'Lasso'
        # fill all nan in rho_lasso with 0, and fill all nan in sig_lasso 0
        df_lasso['rho'] = df_lasso['rho'].fillna(0)
        df_lasso['sig'] = df_lasso['sig'].fillna(0)
        df_mirth = pd.read_csv(f'{file_path}/MIRTH/results_MET_RNA_4_benchmarking/{sub_dir[i]}/ave_rho_padj.csv',
                                   header=0, index_col='feature')
        df_mirth['method'] = 'MIRTH'
        df_unitedmet = pd.read_csv(f'{file_path}/MetabolicModel/results_RNA_ccRCC/{sub_dir[i]}/median_rho_feature.csv', header=0, index_col='feature')
        df_unitedmet= df_unitedmet.rename(columns={'median_rho': 'rho'})  # avoid duplicate column names
        df_unitedmet = df_unitedmet.rename(columns={'sig_in_most': 'sig'})
        df_unitedmet['method'] = 'UnitedMet'
        # concatenate the three dataframes
        data = pd.concat([df_lasso[['rho','sig','method']],
                          df_mirth[['rho','sig','method']],
                          df_unitedmet[['rho','sig','method']]], axis=0)

        data['dataset'] = sub_dir[i]
        benchmark_df = pd.concat([benchmark_df, data], axis=0)
    benchmark_df.to_csv(f'{file_path}/MetabolicModel/results_RNA_ccRCC/final_benchmark_long_table_final.csv')
    # all corresponding dir are copied to results_MET_RNA_4_final


    # -----------------------------Summarize 50% held-out results-----------------------------
    # f_test_prop=50%, n_iter=10
    file_path = "/data1/reznike/xiea1"
    sub_dir = ['CPTAC', 'CPTAC_val', 'RC18', 'RC20']  # notice that CPTAC is the first one

    benchmark_bar_df = pd.DataFrame(columns=['dataset', 'method', 'well_predicted_percent'])
    for i in range(len(sub_dir)):
        df_lasso = pd.read_csv(
            f'{file_path}/MetabolicModel/results_RNA_lasso_met/{sub_dir[i]}/0/actual_pred_rho_matrix.csv',
            header=0, index_col=0)
        df_lasso = df_lasso.rename(columns={'rho': 'rho_lasso'})  # avoid duplicate column names
        df_lasso = df_lasso.rename(columns={'sig': 'sig_lasso'})
        df_mirth = pd.read_csv(f'{file_path}/MIRTH/results_MET_RNA_4_met/{sub_dir[i]}/ave_rho_padj.csv',
                               header=0, index_col='feature')
        df_unitedmet = pd.read_csv(f'{file_path}/MetabolicModel/results_RNA_ccRCC_benchmarking_met_weighted/{sub_dir[i]}/median_rho_feature.csv',
                                   header=0, index_col='feature')
        # concatenate the three dataframes
        df = pd.concat([df_lasso[['rho_lasso']], df_mirth[['rho']], df_unitedmet[["median_rho"]]], axis=1, join='inner')
        # replace nan with 0
        df = df.fillna(0)
        df['high_in_unitedmet'] = df['median_rho'] > df['rho_lasso']
        print(
            f'{sub_dir[i]}: {sum(df["high_in_unitedmet"]) / len(df) * 100}%')  # create a new column 'best_method' if the row is largest in UnitedMet, then the value is UnitedMet, if the row is largest in MIRTH, then the value in MIRTH, else the value is Lasso

        df['best_method'] = df[['rho_lasso', 'rho', 'median_rho']].idxmax(axis=1)
        print(df['best_method'].value_counts() / df.shape[0] * 100)


        # perform paired wilcoxon test (whitney manny u test)
        diff = np.mean((df['median_rho'].to_numpy()) - (df['rho_lasso'].to_numpy()))
        pvalue = scipy.stats.wilcoxon(df['rho_lasso'], df['median_rho']).pvalue
        print(f'{sub_dir[i]}, Lasso vs UM: diff={diff}, pvalue={pvalue}')

        diff = np.mean((df['median_rho'].to_numpy()) - (df['rho'].to_numpy()))
        pvalue = scipy.stats.wilcoxon(df['rho'], df['median_rho']).pvalue
        print(f'{sub_dir[i]}, MIRTH vs UM: diff={diff}, pvalue={pvalue}')

        # bar plot compare the number of features with rho > 0 and significant
        data = pd.concat([df_lasso[['rho_lasso']], df_lasso[['sig_lasso']],
                          df_mirth[['rho']], df_mirth[['sig']],
                          df_unitedmet[["median_rho"]], df_unitedmet[["sig_in_most"]]], axis=1, join='inner')
        # fill all nan in rho_lasso with 0, and fill all nan in sig_lasso 0
        data['rho_lasso'] = data['rho_lasso'].fillna(0)
        data['sig_lasso'] = data['sig_lasso'].fillna(0)
        # calculate the number of features with rho > 0 and sig == ture
        number_lasso = data.loc[(data['rho_lasso'] > 0) & (data['sig_lasso'] == True)].shape[0]
        number_mirth = data.loc[(data['rho'] > 0) & (data['sig'] == True)].shape[0]
        number_unitedmet = data.loc[(data['median_rho'] > 0) & (data['sig_in_most'] == True)].shape[0]
        benchmark_bar_df = benchmark_bar_df._append(
            {'dataset': sub_dir[i], 'method': 'Lasso', 'well_predicted_percent': number_lasso / data.shape[0] * 100},
            ignore_index=True)
        benchmark_bar_df = benchmark_bar_df._append(
            {'dataset': sub_dir[i], 'method': 'MIRTH', 'well_predicted_percent': number_mirth / data.shape[0] * 100},
            ignore_index=True)
        benchmark_bar_df = benchmark_bar_df._append({'dataset': sub_dir[i], 'method': 'UnitedMet',
                                                     'well_predicted_percent': number_unitedmet / data.shape[0] * 100},
                                                    ignore_index=True)
        benchmark_bar_df.to_csv(f'{file_path}/MetabolicModel/results_RNA_ccRCC_benchmarking_met_weighted/final_benchmark_bar_plot_df_final.csv')

    # get the big table of all the benchmarking results with f_test_prop=50%, n_iter=10
    benchmark_df = pd.DataFrame(columns=['rho', 'sig', 'method', 'dataset'])
    for i in range(len(sub_dir)):
        df_lasso = pd.read_csv(f'{file_path}/MetabolicModel/results_RNA_lasso_met/{sub_dir[i]}/0/actual_pred_rho_matrix.csv', header=0, index_col=0)
        df_lasso['method'] = 'Lasso'
        # fill all nan in rho_lasso with 0, and fill all nan in sig_lasso 0
        df_lasso['rho'] = df_lasso['rho'].fillna(0)
        df_lasso['sig'] = df_lasso['sig'].fillna(0)
        df_mirth = pd.read_csv(f'{file_path}/MIRTH/results_MET_RNA_4_met/{sub_dir[i]}/ave_rho_padj.csv', header=0, index_col='feature')
        df_mirth['method'] = 'MIRTH'
        df_unitedmet = pd.read_csv(f'{file_path}/MetabolicModel/results_RNA_ccRCC_benchmarking_met_weighted/{sub_dir[i]}/median_rho_feature.csv', header=0, index_col='feature')
        df_unitedmet= df_unitedmet.rename(columns={'median_rho': 'rho'})  # avoid duplicate column names
        df_unitedmet = df_unitedmet.rename(columns={'sig_in_most': 'sig'})
        df_unitedmet['method'] = 'UnitedMet'
        # concatenate the three dataframes
        data = pd.concat([df_lasso[['rho','sig','method']],
                          df_mirth[['rho','sig','method']],
                          df_unitedmet[['rho','sig','method']]], axis=0)

        data['dataset'] = sub_dir[i]
        benchmark_df = pd.concat([benchmark_df, data], axis=0)
    benchmark_df.to_csv(f'{file_path}/MetabolicModel/results_RNA_ccRCC_benchmarking_met_weighted/final_benchmark_long_table_final.csv')







