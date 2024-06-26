import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import scipy.stats
from statsmodels.stats.multitest import multipletests
import numpy as np
from Performance_Benchmarking.scripts.utils import re_rank_2D

if __name__ == "__main__":
    file_path = "/data1/reznike/xiea1/MetabolicModel"
    imputation_dir = 'KICH_isotope_pyr'
    results_dir = f'{file_path}/results_RNA_imputation/{imputation_dir}'
    embedding_dir = f'{results_dir}/embeddings'
    plots_dir = f'{results_dir}/plots'
    batch_size = 91  # KICH_isotope: 91 (TCGA KICH)

    #--------------------------Get imputed KICH isotope results----------------------------------
    rank_hat_draws_mean = np.load(f'{embedding_dir}/rank_hat_draws_mean_met.npy')[:batch_size, :]
    rank_hat_draws_std = np.load(f'{embedding_dir}/rank_hat_draws_std_met.npy')[:batch_size, :] / batch_size
    rank_hat_draws_std = rank_hat_draws_std # divide by np.sqrt(1000), sample size=1000 draws, if getting standard error
    # rerank metabolite ranks in the flipped way (now rank 0 means the smallest value, # 1 means the largest value)
    rank_hat_draws_mean = re_rank_2D(rank_hat_draws_mean)
    rank_hat_draws_mean = rank_hat_draws_mean / rank_hat_draws_mean.shape[0]  # transformed all ranks to [0,1)
    normalized_data_met = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv',
                                  index_col=0, header=0).iloc[:batch_size, :rank_hat_draws_mean.shape[1]]
    imputed_mean_met = pd.DataFrame(rank_hat_draws_mean, index=normalized_data_met.index,columns=normalized_data_met.columns)
    imputed_std_met = pd.DataFrame(rank_hat_draws_std, index=normalized_data_met.index,columns=normalized_data_met.columns)
    imputed_mean_met.to_csv(f'{results_dir}/KICH_isotope_imputed_mean_met_flipped_glcm6.csv')
    imputed_std_met.to_csv(f'{results_dir}/KICH_isotope_imputed_se_met_glcm6.csv')


    # ----------------------Imputed TCGA KICH_isotope only Isotopologue v.s. mtDNA Mutation------------------------
    results_dir = f'/data1/reznike/xiea1/MetabolicModel/results_RNA_imputation/KICH_isotope_pyr'
    imputed_mean_met = pd.read_csv(f'{results_dir}/KICH_isotope_imputed_mean_met_flipped_glcm6.csv', index_col=0, header=0)
    imputed_std_met = pd.read_csv(f'{results_dir}/KICH_isotope_imputed_se_met_glcm6.csv', index_col=0, header=0)
    imputed_mean_met = imputed_mean_met[sorted(imputed_mean_met.columns)]
    imputed_std_met = imputed_std_met[sorted(imputed_std_met.columns)]
    # subset to tumor samples
    sample_type = pd.read_csv(f'{file_path}/data/RNA_raw_imputation/KICH_sample_type.csv',
                                      index_col=0, header=0)
    tumor_sample_indices = sample_type[sample_type['TN'] == 'Tumor'].index
    imputed_mean_met = imputed_mean_met.loc[tumor_sample_indices, :]
    imputed_mean_met.index = imputed_mean_met.index.str[5:12]
    imputed_std_met = imputed_std_met.loc[tumor_sample_indices, :]
    imputed_std_met.index = imputed_std_met.index.str[5:12]
    # subset to tumor samples with mtDNA data sequenced
    sample_info = pd.read_excel(f'{file_path}/data/TCGA_downstream/ChRCC_sample_info.xlsx',
                                index_col=1, header=1, sheet_name='by Patient')
    sample_info.index = sample_info.index.str[5:12]
    mtdna_sample_indices = sample_info.loc[sample_info['mtDNA data'] == 'YES'].index
    imputed_mean_met = imputed_mean_met.loc[mtdna_sample_indices, :]
    imputed_std_met = imputed_std_met.loc[mtdna_sample_indices, :]

    imputed_mean_met.to_csv(f'{results_dir}/KICH_isotope_imputed_mean_met_flipped_glcm6_subset.csv')
    imputed_std_met.to_csv(f'{results_dir}/KICH_isotope_imputed_se_met_glcm6_subset.csv')

    # get mtDNA mutation data
    mutation = pd.read_excel(
        '/data1/reznike/xiea1/MetabolicModel/data/TCGA_downstream/ChRCC_mtdna_mutation.xlsx',
        sheet_name=1, header=3)
    mutation = mutation.iloc[:-2, :]  # remove last two rows

    # ----------------------MT-ND5 mutant v.s. WT samples------------------------
    isotope_list = pd.read_csv(f'{file_path}/results_RNA_MITO/MITO1/median_rho_feature.csv',
                               index_col='feature', header=0).index.to_list()
    mtnd5_indices = mutation.loc[mutation['Gene'] == 'MT-ND5', 'case'].values
    # Wilcoxon rank sum test

    met_wilcox_df = pd.DataFrame({'mean difference': [], 'p': []})
    for metabolite in isotope_list:
        mut = imputed_mean_met.loc[mtnd5_indices, metabolite]
        wt = imputed_mean_met.loc[imputed_mean_met.index.isin(mtnd5_indices) == False, metabolite]
        met_wilcox_df.loc[metabolite, 'mean difference'] = mut.mean() - (
            wt.mean())  # up in mut on the right in plots
        met_wilcox_df.loc[metabolite, 'p'] = scipy.stats.ranksums(mut, wt).pvalue
    met_wilcox_df['p_adj'] = multipletests(pvals=met_wilcox_df['p'], method="fdr_bh", alpha=0.1)[1]
    met_wilcox_df.to_csv(f'{results_dir}/downstream_analysis/wilcox_df/wilcox_mtnd5_glcm6.csv')

    # ----------------------Cx 1 mutant v.s. WT samples------------------------
    # transport chain complex 1: MT-ND 1-6
    cx1_genes = ['MT-ND1', 'MT-ND2', 'MT-ND3', 'MT-ND4', 'MT-ND5', 'MT-ND6']
    cx1_indices = mutation.loc[mutation['Gene'].isin(cx1_genes), 'case'].unique()
    cx1_indel_indices = mutation.loc[(mutation['Gene'].isin(cx1_genes)) & (mutation['SNV/indel']=='indel'), 'case'].unique()
    cx1_snv_indices = mutation.loc[(mutation['Gene'].isin(cx1_genes)) & (mutation['SNV/indel']=='SNV'), 'case'].unique()
    print(cx1_indel_indices)
    print(cx1_snv_indices)

    # Wilcoxon rank sum test
    met_wilcox_df = pd.DataFrame({'mean difference': [], 'p': []})
    for metabolite in isotope_list:
        mut = imputed_mean_met.loc[cx1_indices, metabolite]
        wt = imputed_mean_met.loc[imputed_mean_met.index.isin(cx1_indices) == False, metabolite]
        met_wilcox_df.loc[metabolite, 'mean difference'] = mut.mean() - (
            wt.mean())  # up in mut on the right in plots
        met_wilcox_df.loc[metabolite, 'p'] = scipy.stats.ranksums(mut, wt).pvalue
    met_wilcox_df['p_adj'] = multipletests(pvals=met_wilcox_df['p'], method="fdr_bh", alpha=0.1)[1]
    met_wilcox_df.to_csv(f'{results_dir}/downstream_analysis/wilcox_df/wilcox_cx1_glcm6.csv')

    # ----------------------all mutant v.s. WT samples------------------------
    # transport chain complex 1: MT-ND 1-6
    all_indices = mutation['case'].unique()
    print(all_indices)
    # Wilcoxon rank sum test

    met_wilcox_df = pd.DataFrame({'mean difference': [], 'p': []})
    for metabolite in isotope_list:
        mut = imputed_mean_met.loc[all_indices, metabolite]
        wt = imputed_mean_met.loc[imputed_mean_met.index.isin(all_indices) == False, metabolite]
        met_wilcox_df.loc[metabolite, 'mean difference'] = mut.mean() - (
            wt.mean())  # up in mut on the right in plots
        met_wilcox_df.loc[metabolite, 'p'] = scipy.stats.ranksums(mut, wt).pvalue
    met_wilcox_df['p_adj'] = multipletests(pvals=met_wilcox_df['p'], method="fdr_bh", alpha=0.1)[1]
    met_wilcox_df.to_csv(f'{results_dir}/downstream_analysis/wilcox_df/wilcox_all_glcm6.csv')


    # ---------------------- box plot showing all 1000 draws----------------------
    # reorder the samples to show mtDNA mutant samples on the top
    indel_indices = cx1_indel_indices
    snv_indices = cx1_snv_indices
    mut_indices = cx1_indices

    # pick out pyruvate m+3, lactate m+3, aspartate m+2, malate m+2
    metabolite_list = ['Pyruvate m+3', 'Lactate m+3', 'Aspartate m+2', 'Citrate m+2', 'Malate m+2']
    sort_type= 'separate'
    for met in metabolite_list:
        # sort the samples according to the mean value of the metabolite in mtDNA mutant samples and WT samples separately
        df_met_mean = imputed_mean_met.loc[:, metabolite_list]
        df_met_std = imputed_std_met.loc[:, metabolite_list]
        if sort_type == 'yaxis':
            df_met_mean = df_met_mean.sort_values(by=met, axis=0, ascending=False)
            df_met_std = df_met_std.loc[df_met_mean.index, :]
        elif sort_type == 'separate':
            if met == 'Pyruvate m+3' or met == 'Lactate m+3':
                df_met_mean = pd.concat([df_met_mean.loc[mut_indices].sort_values(by=met, ascending=False),
                                         df_met_mean.loc[df_met_mean.index.isin(mut_indices)==False].sort_values(by=met, ascending=False)],
                                        axis=0)
                df_met_std = df_met_std.loc[df_met_mean.index, :]

            elif met == 'Aspartate m+2' or met == 'Malate m+2' or met == 'Citrate m+2':
                # sort the samples according to the mean value of the metabolite in mtDNA mutant samples and WT samples separately
                df_met_mean = pd.concat([df_met_mean.loc[mut_indices].sort_values(by=met, ascending=True),
                                         df_met_mean.loc[df_met_mean.index.isin(mut_indices)==False].sort_values(by=met, ascending=True)],
                                        axis=0)
                df_met_std = df_met_std.loc[df_met_mean.index, :]
        fig, ax = plt.subplots(figsize=(10, 5))
        # label the samples in mut_indices by different colors
        for i, sample in enumerate(df_met_mean.index):
            if sample in cx1_indel_indices:
                ax.plot(i + 1, df_met_mean.loc[sample, met], marker='o', color='salmon', markersize=8)
            elif sample in cx1_snv_indices:
                ax.plot(i + 1, df_met_mean.loc[sample, met], marker='o', color='palegreen', markersize=8)
            else:
                ax.plot(i + 1, df_met_mean.loc[sample, met], marker='o', color='skyblue', markersize=8)
        for i, (mean, std_dev) in enumerate(zip(df_met_mean[met], df_met_std[met])):
            ax.plot([i + 1, i + 1], [mean - std_dev, mean + std_dev], color='darkgrey', linewidth=2)
        # add annotation legend saying the red dots are mtDNA mutant samples
        ax.plot([], [], 'o', color='salmon', label='Complex 1 indel')
        ax.plot([], [], 'o', color='palegreen', label='Complex 1 SNV')
        ax.plot([], [], 'o', color='skyblue', label='WT')
        ax.legend()
        ax.set_xlabel('Samples')
        ax.set_ylabel(f'{met} abundance')
        ax.set_title(f'Predicted {met} abundance of KICH')
        plt.savefig(f'{results_dir}/downstream_analysis/plots_final/boxplot_{met}_glcm6_cx1_{sort_type}.pdf')
        plt.close()

    # ---------------------- plot heatmap ----------------------
    # subset imputed_mean_met to isotope_list
    imputed_mean_met = imputed_mean_met.loc[:, isotope_list]
    # create sample annotation labels according to mtDNA mutations
    sample_annotation = pd.DataFrame({'mtDNA mutation': []})
    for sample in imputed_mean_met.index:
        if sample in mtnd5_indices:
            sample_annotation.loc[sample, 'mtDNA mutation'] = 'MUT'
        else:
            sample_annotation.loc[sample, 'mtDNA mutation'] = 'WT'

    light_red = (204/255, 153/255, 153/255)  # (R, G, B) values for light red
    light_blue = (173 / 255, 216 / 255, 230 / 255)  # (R, G, B) values for light blue
    color_map = {"MUT": light_red, "WT": light_blue}
    row_colors = sample_annotation['mtDNA mutation'].map(color_map)

    # plot heatmap
    sns.clustermap(imputed_mean_met, row_colors=row_colors, cmap='coolwarm', vmin=0, vmax=1,
                linewidths=0.1, linecolor='black', figsize=(20, 20))
    plt.xlabel("Isotopologues")
    plt.ylabel("Samples")
    plt.title("Predicted isotopogule data of KICH")
    plt.savefig(f'{results_dir}/downstream_analysis/plots/isotopologue_heatmap_mtnd5_glcm6.pdf')
    plt.close()

    # reorder the samples to show mtDNA mutant samples on the top

    # plot box plots for every sample, showing it deviance from the mean





