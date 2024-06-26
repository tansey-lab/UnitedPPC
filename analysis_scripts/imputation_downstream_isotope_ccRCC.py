import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import scipy.stats
from statsmodels.stats.multitest import multipletests
import numpy as np
from Performance_Benchmarking.scripts.utils import re_rank_2D

if __name__ == "__main__":
    file_path = "/juno/work/reznik/xiea1/MetabolicModel"
    imputation_dir = 'KIRC_isotope_pyr'
    results_dir = f'{file_path}/results_RNA_imputation/{imputation_dir}'
    embedding_dir = f'{results_dir}/embeddings'
    plots_dir = f'{results_dir}/plots'
    batch_size = 606 # KIRC_isotope: 606 (same with TCGA KIRC)

    #--------------------------Get imputed KIPAN isotope results----------------------------------
    rank_hat_draws_mean = np.load(f'{embedding_dir}/rank_hat_draws_mean_met.npy')[:batch_size, :]
    rank_hat_draws_std = np.load(f'{embedding_dir}/rank_hat_draws_std_met.npy')[:batch_size, :] / batch_size
    # rerank metabolite ranks
    rank_hat_draws_mean = rank_hat_draws_mean.argsort(
        axis=0, kind='stable').argsort(axis=0, kind='stable')
    # rerank metabolite ranks in the flipped way (now rank 0 means the smallest value, # 1 means the largest value)
    rank_hat_draws_mean = re_rank_2D(rank_hat_draws_mean)
    rank_hat_draws_mean = rank_hat_draws_mean / rank_hat_draws_mean.shape[0]  # transformed all ranks to [0,1)
    normalized_data_met = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv',
                                  index_col=0, header=0).iloc[:batch_size, :rank_hat_draws_mean.shape[1]]
    imputed_mean_met = pd.DataFrame(rank_hat_draws_mean, index=normalized_data_met.index,columns=normalized_data_met.columns)
    imputed_std_met = pd.DataFrame(rank_hat_draws_std, index=normalized_data_met.index,columns=normalized_data_met.columns)
    imputed_mean_met.to_csv(f'{results_dir}/KIRC_isotope_imputed_mean_met_flipped.csv')
    imputed_std_met.to_csv(f'{results_dir}/KIRC_isotope_imputed_std_met.csv')

    # ---------------------load the raw unmatched MITO isotope data--------------------------
    # load the raw unmatched MITO isotope data (110 samples)
    data_path = '/juno/work/reznik/xiea1/MetabolicModel/data/Isotope_raw_mito'
    raw_data = pd.read_csv(f'{data_path}/raw_isotope_MITO.csv', index_col=0, header=0)
    # Normalization by tissue glucose m+6
    raw_data = raw_data[raw_data['Glucose m+6'].notna()]
    # remove metastatic samples
    # 'VM36_Tumor_Met'
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
    ranked_raw_mito = np.where(ranks_temp > n_obs, n_obs, ranks_temp) # np.nan will be replaced by the largest value
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
    ranked_raw_mito.to_csv(f'{embedding_dir}/ranked_raw_mito_isotope_flipped.csv')

    # ----------------------Actual Mito Isotopologue v.s. BAP1------------------------
    # Load BAP1 mutation data
    data_path = '/juno/work/reznik/xiea1/MetabolicModel/data/Isotope_raw_mito'
    clinical_df = pd.read_excel(f'{data_path}/bap1_clinical.xlsx',
                         index_col=0, header=1)
    clinical = clinical_df.loc[clinical_df['BAP1 STATUS'].isin(['+', 'LOSS', 'PARTIAL LOSS']), :]
    clinical.loc[clinical['BAP1 STATUS'] == '+', 'BAP1 STATUS'] = 0
    clinical.loc[clinical['BAP1 STATUS'] != 0, 'BAP1 STATUS'] = 1
    ranked_mito = pd.read_csv(f'{file_path}/results_RNA_imputation/KIPAN/embeddings/ranked_raw_mito_isotope_flipped.csv', index_col=0, header=0)
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
    # just keep the string before first '_' in the index
    ranked_mito.index = [i.split('_')[0] for i in ranked_mito.index]

    isotope_samples = ranked_mito.index
    bap1_samples = clinical.index
    common_samples = list(set(isotope_samples).intersection(set(bap1_samples)))
    ranked_mito = ranked_mito.loc[common_samples]
    clinical = clinical.loc[common_samples]

    # Wilcoxon rank sum test
    gene = 'BAP1 STATUS'
    mean_data = pd.concat([clinical[gene], ranked_mito], axis=1)
    results_dir = f'{file_path}/results_RNA_imputation/KIPAN/downstream_analysis/wilcox_volcano_mito'
    mean_data.to_csv(f'{results_dir}/mut_isotope_glc_flipped.csv')
    met_wilcox_df = pd.DataFrame({'mean difference': [], 'p': []})
    for metabolite in ranked_mito.columns:
        mut = mean_data.loc[mean_data[gene] == 1, metabolite]
        wt = mean_data.loc[mean_data[gene] == 0, metabolite]
        met_wilcox_df.loc[metabolite, 'mean difference'] = wt.mean() - (
            mut.mean())  # up in mut on the right in plots
        met_wilcox_df.loc[metabolite, 'p'] = scipy.stats.ranksums(mut, wt).pvalue
    met_wilcox_df['p_adj'] = multipletests(pvals=met_wilcox_df['p'], method="fdr_bh", alpha=0.1)[1]
    met_wilcox_df.to_csv(f'{results_dir}/downstream_analysis/wilcox_volcano_mito/wilcox_{gene}.csv')



    # ----------------------Imputed TCGA KIRC_isotope only Isotopologue v.s. Mutation------------------------
    results_dir = f'/juno/work/reznik/xiea1/MetabolicModel/results_RNA_imputation/KIRC_isotope_pyr'
    imputed_mean_met = pd.read_csv(f'{results_dir}/KIRC_isotope_imputed_mean_met_flipped.csv', index_col=0, header=0)
    imputed_mean_met = imputed_mean_met[sorted(imputed_mean_met.columns)]
    imputed_mean_met.index = imputed_mean_met.index.str[8:15]

    # get mutation data
    mutation = pd.read_csv(
        '/juno/work/reznik/xiea1/MetabolicModel/data/TCGA_downstream/TCGA_KIRC_sample_mutation_matrix.csv',
        index_col=0, header=0)

    # Wilcoxon rank sum test
    mean_data = pd.concat([mutation, imputed_mean_met], axis=1, join='inner')
    mean_data.to_csv(f'{results_dir}/downstream_analysis/mut_isotope_kirc_flipped.csv')
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




