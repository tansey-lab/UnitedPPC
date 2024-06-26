import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import scipy.stats
from statsmodels.stats.multitest import multipletests
import numpy as np
import umap
from Performance_Benchmarking.scripts.utils import re_rank_2D

if __name__ == "__main__":
    file_path = "/juno/work/reznik/xiea1/MetabolicModel"
    imputation_dir = 'KIPAN'
    results_dir = f'{file_path}/results_RNA_imputation/{imputation_dir}'
    embedding_dir = f'{results_dir}/embeddings'
    plots_dir = f'{results_dir}/plots'
    batch_size = 1020  # KIPAN: 1020

    #--------------------------Get imputed KIPAN isotope results----------------------------------
    rank_hat_draws_mean = np.load(f'{embedding_dir}/rank_hat_draws_mean_met.npy')[:batch_size, :]
    rank_hat_draws_std = np.load(f'{embedding_dir}/rank_hat_draws_std_met.npy')[:batch_size, :] / batch_size
    # rerank metabolite ranks
    rank_hat_draws_mean = rank_hat_draws_mean.argsort(
        axis=0, kind='stable').argsort(axis=0, kind='stable')
    # rerank metabolite ranks in the flipped way (now rank 0 means the smallest value, # 1 means the largest value)
    rank_hat_draws_mean = re_rank_2D(rank_hat_draws_mean)
    rank_hat_draws_mean = rank_hat_draws_mean / rank_hat_draws_mean.shape[0]  # transformed all ranks to [0,1]
    normalized_data_met = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv',
                                  index_col=0, header=0).iloc[:batch_size, :rank_hat_draws_mean.shape[1]]
    imputed_mean_met = pd.DataFrame(rank_hat_draws_mean, index=normalized_data_met.index,columns=normalized_data_met.columns)
    imputed_std_met = pd.DataFrame(rank_hat_draws_std, index=normalized_data_met.index,columns=normalized_data_met.columns)
    imputed_mean_met.to_csv(f'{results_dir}/KIPAN_imputed_mean_met.csv')
    imputed_std_met.to_csv(f'{results_dir}/KIPAN_imputed_std_met.csv')
    imputed_mean_met.to_csv(f'{results_dir}/TCGA_imputed_mean_met_flipped.csv')

    # ---------------------Isotope differential test for actual MITO data--------------------------
    # load the raw unmatched MITO isotope data (110 samples)
    data_path = '/juno/work/reznik/xiea1/MetabolicModel/data/Isotope_raw_mito'
    raw_data = pd.read_csv(f'{data_path}/raw_isotope_MITO.csv', index_col=0, header=0)
    # Normalization by tissue glucose m+6
    raw_data = raw_data[raw_data['Glucose m+6'].notna()]
    # remove metastatic samples
    raw_data = raw_data.drop(['VM28_Tumor_Met', 'VM34_Tumor_Met', 'VM37_Tumor_Met', 'VM38_Tumor_Met',
                             'VM54_Tumor_Met', 'VM66_Tumor_Met'], axis=0)
    normalized_data = pd.DataFrame(raw_data.to_numpy()/(raw_data['Glucose m+6'].to_numpy().reshape(raw_data.shape[0], -1)),
                                   index=raw_data.index, columns=raw_data.columns)

    # Rank transform data
    batch = np.copy(normalized_data)
    n_obs = np.sum(~np.isnan(batch), axis=0)
    batch = np.where(np.isnan(batch), 0, batch)  # important, replace nan with 0
    ranks_temp = batch.argsort(axis=0, kind='stable')[::-1].argsort(
        axis=0, kind='stable')  # rank (largest item has rank 0, second largest has rank 1, etc.)
    ranked_raw_mito = np.where(ranks_temp > n_obs, n_obs, ranks_temp)
    ranked_raw_mito = pd.DataFrame(ranked_raw_mito / ranked_raw_mito.shape[0], index=normalized_data.index,
                               columns=normalized_data.columns)
    ranked_raw_mito.to_csv(f'{embedding_dir}/ranked_raw_mito_isotope.csv')

    # flip rank transformation (largest item has rank 1, smallest has rank 0, etc.)
    batch = np.copy(normalized_data)
    n_obs = np.sum(~np.isnan(batch), axis=0)
    ranks_temp = batch.argsort(axis=0, kind='stable').argsort(
        axis=0, kind='stable')
    ranked_raw_mito = np.where(ranks_temp > n_obs, 0, ranks_temp)  # np.nan will be replaced by 0
    ranked_raw_mito = pd.DataFrame(ranked_raw_mito / ranked_raw_mito.shape[0], index=normalized_data.index,
                               columns=normalized_data.columns)
    ranked_raw_mito = ranked_raw_mito.loc[:, ~ranked_raw_mito.columns.str.endswith('m+0')]
    ranked_raw_mito.to_csv(f'{embedding_dir}/ranked_raw_mito_isotope_flipped.csv')

    # load ranked MITO isotope data
    data = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv', header=0, index_col=0)
    batch = data.iloc[batch_size:, :rank_hat_draws_mean.shape[1]]
    n_obs = np.sum(~np.isnan(batch), axis=0).to_numpy()
    batch = np.where(np.isnan(batch), 0, batch)  # important, replace nan with 0
    ranks_temp = batch.argsort(axis=0, kind='stable')[::-1].argsort(
        axis=0, kind='stable')  # rank (largest item has rank 0, second largest has rank 1, etc.)
    ranked_mito = np.where(ranks_temp > n_obs, n_obs, ranks_temp)
    batch = data.iloc[batch_size:, :rank_hat_draws_mean.shape[1]]
    ranked_mito = pd.DataFrame(ranked_mito/ranked_mito.shape[0], index=batch.index, columns=batch.columns)
    ranked_mito.to_csv(f'{embedding_dir}/ranked_mito_isotope.csv')

    # load MITO isotope sample type info
    data_path = '/juno/work/reznik/xiea1/MetabolicModel/data/Isotope_raw_mito'
    sample_info_mito = pd.read_csv(f'{data_path}/isotope_sample_subtype_info.csv', index_col=0, header=0)

    mito = pd.concat([ranked_mito, sample_info_mito], join='inner', axis=1)
    test = mito
    met_wilcox_df = pd.DataFrame({'mean difference_chromo': [],  'p_chromo': [], 'mean difference_papi': [],  'p_papi': []})
    for isotopologue in list(test.iloc[:, :-1].columns):
        ccrcc = test.loc[test['Subtype'] == 'CCRCC', isotopologue]
        chromo = test.loc[(test['Subtype'] == 'Chromophobe', isotopologue)]
        papi = test.loc[(test['Subtype'] == 'Papillary', isotopologue)]
        met_wilcox_df.loc[isotopologue, 'mean difference_chromo'] = chromo.mean() - (ccrcc.mean())
        met_wilcox_df.loc[isotopologue, 'p_chromo'] = scipy.stats.ranksums(chromo, ccrcc).pvalue
        met_wilcox_df.loc[isotopologue, 'mean difference_papi'] = papi.mean() - (ccrcc.mean())
        met_wilcox_df.loc[isotopologue, 'p_papi'] = scipy.stats.ranksums(papi, ccrcc).pvalue
    met_wilcox_df['p_adj_chromo'] = multipletests(pvals=met_wilcox_df['p_chromo'], method="fdr_bh", alpha=0.1)[1]
    met_wilcox_df['p_adj_papi'] = multipletests(pvals=met_wilcox_df['p_papi'], method="fdr_bh", alpha=0.1)[1]


    #----------------------Isotope differential test for imputed KIPAN data------------------------
    data_path = '/juno/work/reznik/xiea1/MetabolicModel/data/RNA_raw_imputation'
    sample_info_kipan = pd.read_csv(f'{data_path}/KIPAN.clin.merged.picked.txt',
                                    sep='\t', index_col=0, header=0).T
    sample_type_kipan = pd.read_csv(f'{data_path}/KIPAN_sample_type.csv',
                                      index_col=0, header=0)
    imputed_mean_met = pd.read_csv(f'{results_dir}/TCGA_imputed_mean_met_flipped.csv', index_col=0, header=0)
    imputed_mean_met = imputed_mean_met.loc[sample_type_kipan.loc[sample_type_kipan['code'].isin(['01A', '01B'])].index]
    imputed_mean_met.index = [i[:12].lower() for i in imputed_mean_met.index]
    kipan = pd.concat([imputed_mean_met, sample_info_kipan['histological_type']], join='inner', axis=1)

    test = kipan
    met_wilcox_df_kipan = pd.DataFrame({'mean difference_chromo_kipan': [],  'p_chromo_kipan': [],
                                  'mean difference_papi_kipan': [],  'p_papi_kipan': []})
    for isotopologue in list(test.iloc[:, :-1].columns):
        ccrcc = test.loc[test['histological_type'] == 'kidney clear cell renal carcinoma', isotopologue]
        chromo = test.loc[(test['histological_type'] == 'kidney chromophobe', isotopologue)]
        papi = test.loc[(test['histological_type'] == 'kidney papillary renal cell carcinoma', isotopologue)]
        met_wilcox_df_kipan.loc[isotopologue, 'mean difference_chromo_kipan'] = chromo.mean() - (ccrcc.mean())
        met_wilcox_df_kipan.loc[isotopologue, 'p_chromo_kipan'] = scipy.stats.ranksums(chromo, ccrcc).pvalue
        met_wilcox_df_kipan.loc[isotopologue, 'mean difference_papi_kipan'] = papi.mean() - (ccrcc.mean())
        met_wilcox_df_kipan.loc[isotopologue, 'p_papi_kipan'] = scipy.stats.ranksums(papi, ccrcc).pvalue
    met_wilcox_df_kipan['p_adj_chromo_kipan'] = multipletests(pvals=met_wilcox_df_kipan['p_chromo_kipan'], method="fdr_bh", alpha=0.1)[1]
    met_wilcox_df_kipan['p_adj_papi_kipan'] = multipletests(pvals=met_wilcox_df_kipan['p_papi_kipan'], method="fdr_bh", alpha=0.1)[1]


    wilcox_df = pd.concat([met_wilcox_df, met_wilcox_df_kipan],axis=1)
    wilcox_df.to_csv(f'{results_dir}/wilcox_cancer_subtype_mito_kipan.csv')
    wilcox_df.to_csv(f'{results_dir}/wilcox_cancer_subtype_mito_kipan_flipped.csv')

    plt.scatter(x=wilcox_df['mean difference_chromo'], y=wilcox_df['mean difference_chromo_kipan'])
    plt.xlabel("MITO mean difference(T/N) (measured)")
    plt.ylabel('TCGA Log2fc(T/N) (imputed)')
    plt.show()
    rho, pval = spearmanr(wilcox_df['mean difference_chromo'], wilcox_df['mean difference_chromo_kipan'])
    print(f"spearman's rho = {rho}, pval = {pval}")


    # ----------------------MITO actual isotopologue Heatmap------------------------
    ranked_mito = pd.read_csv(f'{embedding_dir}/ranked_raw_mito_isotope.csv', index_col=0, header=0)
    data_path = '/juno/work/reznik/xiea1/MetabolicModel/data/Isotope_raw_mito'
    sample_info_mito = pd.read_csv(f'{data_path}/isotope_sample_subtype_info.csv', index_col=0, header=0)
    sample_info_mito = sample_info_mito.loc[ranked_mito.index]
    # reorder sample_info_mito to put normal samples at the top, ccRCC samples the second, then the papillary, chromophobe and the rest
    sample_info_mito['Subtype'] = sample_info_mito['Subtype'].replace(' ', 'Normal')
    order = ['Normal', 'CCRCC', 'Papillary', 'Chromophobe', 'Oncocytoma',
                                                        'HLRCC']
    sample_info_mito['Subtype'] = pd.Categorical(sample_info_mito['Subtype'], categories=order, ordered=True)
    sample_info_mito = sample_info_mito.sort_values('Subtype')

    ranked_mito = ranked_mito.loc[sample_info_mito.loc[sample_info_mito['Subtype'] != 'Normal'].index]
    # drop metastatic samples
    ranked_mito = ranked_mito.drop(['VM28_Tumor_Met', 'VM34_Tumor_Met','VM66_Tumor_Met'], axis=0)
    # sort the columns by letters
    ranked_mito = ranked_mito[sorted(ranked_mito.columns)]

    sns.clustermap(ranked_mito, col_cluster=False, cmap='coolwarm',
                linewidths=0.1, linecolor='black', cbar=True,
                yticklabels=sample_info_mito.loc[ranked_mito.index, 'Subtype'], figsize=(20, 20))
    plt.xlabel("Isotopologues")
    plt.ylabel("Samples")
    plt.title("Ground truth MITO isotope data")
    plt.savefig(f'{plots_dir}/mito_raw_isotope_heatmap_clustered_row.pdf')
    plt.close()

    # ----------------------Actual Mito Isotopologue v.s. BAP1------------------------
    # Load BAP1 mutation data
    data_path = '/juno/work/reznik/xiea1/MetabolicModel/data/Isotope_raw_mito'
    clinical_df = pd.read_excel(f'{data_path}/bap1_clinical.xlsx',
                         index_col=0, header=1)
    clinical = clinical_df.loc[clinical_df['BAP1 STATUS'].isin(['+', 'LOSS', 'PARTIAL LOSS']), :]
    clinical.loc[clinical['BAP1 STATUS'] == '+', 'BAP1 STATUS'] = 0
    clinical.loc[clinical['BAP1 STATUS'] != 0, 'BAP1 STATUS'] = 1
    ranked_mito = pd.read_csv(f'{embedding_dir}/ranked_raw_mito_isotope.csv', index_col=0, header=0)
    ranked_mito = ranked_mito.loc[:, ~ranked_mito.columns.str.endswith('m+0')]
    data_path = '/juno/work/reznik/xiea1/MetabolicModel/data/Isotope_raw_mito'
    sample_info_mito = pd.read_csv(f'{data_path}/isotope_sample_subtype_info.csv', index_col=0, header=0)
    sample_info_mito = sample_info_mito.loc[ranked_mito.index]
    # reorder sample_info_mito to put normal samples at the top, ccRCC samples the second, then the papillary, chromophobe and the rest
    sample_info_mito['Subtype'] = sample_info_mito['Subtype'].replace(' ', 'Normal')
    order = ['Normal', 'CCRCC', 'Papillary', 'Chromophobe', 'Oncocytoma',
                                                        'HLRCC']
    sample_info_mito['Subtype'] = pd.Categorical(sample_info_mito['Subtype'], categories=order, ordered=True)
    sample_info_mito = sample_info_mito.sort_values('Subtype')
    ranked_mito = ranked_mito.loc[sample_info_mito.loc[sample_info_mito['Subtype'] == 'CCRCC'].index]
    # drop metastatic samples
    #ranked_mito = ranked_mito.drop(['VM34_Tumor_Met', 'VM38_Tumor_Met',
    #                                'VM54_Tumor_Met', 'VM66_Tumor_Met'], axis=0)
    ranked_mito = ranked_mito.drop(['VM34_Tumor_Met', 'VM38_Tumor_Met',
                                    'VM66_Tumor_Met'], axis=0)
    # just keep the string efore first '_' in the index
    ranked_mito.index = [i.split('_')[0] for i in ranked_mito.index]

    isotope_samples = ranked_mito.index
    bap1_samples = clinical.index
    common_samples = list(set(isotope_samples).intersection(set(bap1_samples)))
    ranked_mito = ranked_mito.loc[common_samples]
    clinical = clinical.loc[common_samples]

    # Wilcoxon rank sum test
    gene = 'BAP1 STATUS'
    mean_data = pd.concat([clinical[gene], ranked_mito], axis=1)
    met_wilcox_df = pd.DataFrame({'mean difference': [], 'p': []})
    for metabolite in ranked_mito.columns:
        mut = mean_data.loc[mean_data[gene] == 1, metabolite]
        wt = mean_data.loc[mean_data[gene] == 0, metabolite]
        met_wilcox_df.loc[metabolite, 'mean difference'] = wt.mean() - (
            mut.mean())  # up in mut on the right in plots
        met_wilcox_df.loc[metabolite, 'p'] = scipy.stats.ranksums(mut, wt).pvalue
    met_wilcox_df['p_adj'] = multipletests(pvals=met_wilcox_df['p'], method="fdr_bh", alpha=0.1)[1]
    met_wilcox_df.to_csv(f'{results_dir}/downstream_analysis/wilcox_volcano_mito/wilcox_{gene}.csv')



    # ----------------------Imputed KIRC Isotopologue v.s. Mutation------------------------
    results_dir = f'/juno/work/reznik/xiea1/MetabolicModel/results_RNA_imputation/KIPAN'
    imputed_mean_met = pd.read_csv(f'{results_dir}/KIPAN_imputed_mean_met.csv', index_col=0, header=0)
    imputed_mean_met = imputed_mean_met[sorted(imputed_mean_met.columns)]
    imputed_mean_met.index = imputed_mean_met.index.str[8:15]

    # get mutation data
    mutation = pd.read_csv(
        '/juno/work/reznik/xiea1/MetabolicModel/data/TCGA_downstream/TCGA_KIRC_sample_mutation_matrix.csv',
        index_col=0, header=0)
    imputed_mean_met = imputed_mean_met[imputed_mean_met.index.isin(mutation.index)]
    imputed_mean_met = imputed_mean_met.loc[mutation.index]

    # Wilcoxon rank sum test
    mean_data = pd.concat([mutation, imputed_mean_met], axis=1)
    df_list = []
    for gene in mutation.columns:
        print(gene)
        met_wilcox_df = pd.DataFrame({'mean difference': [], 'p': []})
        for metabolite in imputed_mean_met.columns:
            mut = mean_data.loc[mean_data[gene] == 1, metabolite]
            wt = mean_data.loc[mean_data[gene] == 0, metabolite]
            met_wilcox_df.loc[metabolite, 'mean difference'] = wt.mean() - (
                mut.mean())  # up in mut on the right in plots
            met_wilcox_df.loc[metabolite, 'p'] = scipy.stats.ranksums(mut, wt).pvalue
        met_wilcox_df['p_adj'] = multipletests(pvals=met_wilcox_df['p'], method="fdr_bh", alpha=0.1)[1]
        met_wilcox_df.to_csv(f'{results_dir}/downstream_analysis/wilcox_df_kirc/wilcox_{gene}.csv')
        df_list.append(met_wilcox_df)

    # Check Normalization method for raw mito isotope data
    data = pd.concat([ranked_raw_mito, sample_info_mito], axis=1)
    data.loc[data['Subtype'] != 'CCRCC', 'Subtype'] = 'Other'

    for i in range(4):
        ax = sns.boxplot(data=data, x="Subtype", y=f"Lactate m+{i}")
        ax = sns.stripplot(data=data, x="Subtype", y=f"Lactate m+{i}", jitter=True, alpha=0.5)
        plt.title(f"Lactate m+{i}")
        plt.show()

    # Check Normalization method for imputed kipan isotope data
    kipan.loc[kipan['histological_type'] != 'kidney chromophobe', 'histological_type'] = 'Other'

    for i in range(4):
        ax = sns.boxplot(data=kipan, x="histological_type", y=f"Pyruvate m+{i}")
        ax = sns.stripplot(data=kipan, x="histological_type", y=f"Pyruvate m+{i}", jitter=True, alpha=0.5)
        plt.title(f"Lactate m+{i}")
        plt.show()

    # Check the lactate m+3/pyruvate m+3, citrate m+2/pyruvate m+3 ratios
    for i in ['Alanine m+3', 'Lactate m+3', 'Citrate m+2', 'Pyruvate m+3']:
        ax = sns.boxplot(data=mean_data, x="BAP1", y=i)
        ax = sns.stripplot(data=mean_data, x="BAP1", y=i, jitter=True, alpha=0.5)
        plt.title(i)
        plt.show()

