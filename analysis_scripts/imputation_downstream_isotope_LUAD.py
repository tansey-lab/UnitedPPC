import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from statsmodels.stats.multitest import multipletests
import numpy as np
from Performance_Benchmarking.scripts.utils import re_rank_2D

if __name__ == "__main__":
    file_path = "/data1/reznike/xiea1/MetabolicModel"
    imputation_dir = 'LUAD'
    results_dir = f'{file_path}/results_RNA_imputation/{imputation_dir}'
    embedding_dir = f'{results_dir}/embeddings'
    plots_dir = f'{results_dir}/plots'
    batch_size = 576  # TCGA LUAD: 576

    #--------------------------Get imputed LUAD isotope results----------------------------------
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
    imputed_mean_met.to_csv(f'{results_dir}/LUAD_isotope_imputed_mean_met_flipped.csv')
    imputed_std_met.to_csv(f'{results_dir}/LUAD_isotope_imputed_std_met.csv')


    #------------------------------------GET LUAD MAF----------------------------------
    data_path = "/data1/reznike/xiea1/MetabolicModel/data/TCGA_downstream"
    data = pd.read_csv(f'{data_path}/mc3.v0.2.8.PUBLIC.maf', sep='\t')
    # subset to TCGA LUAD samples
    tcga = pd.read_csv(f'/data1/reznike/xiea1/MetabolicModel/data/RNA_matched_imputation/matched_tpm_LUAD.csv',
                       index_col=0, header=0)
    # only the participant and sample code matters
    data['Tumor_Sample_Barcode_short'] = data['Tumor_Sample_Barcode'].str[8:15]
    tcga['Sample_Barcode_short'] = tcga.index.str[8:15]
    data = data[data['Tumor_Sample_Barcode_short'].isin(tcga['Sample_Barcode_short'])]
    total_samples = set(data['Tumor_Sample_Barcode'])  # 368 total samples
    # subset to 15 ccRCC driver mutations
    driver_mutations = ['EGFR', 'KRAS', 'KEAP1', 'STK11', 'TP53', 'SMARCA4', 'SETD2']
    data = data[data['Hugo_Symbol'].isin(driver_mutations)]
    # excluded ‘silent’, ‘intron’, ‘3utr’, and ‘5 utr’, because they won’t bring a change to gene functions
    driver_mutation_types = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
                             'Splice_Site', 'Frame_Shift_Ins', 'In_Frame_Del',
                             'Translation_Start_Site', 'In_Frame_Ins', 'Nonstop_Mutation']
    data = data[data['Variant_Classification'].isin(driver_mutation_types)]

    # pivot table
    df_wide = pd.pivot_table(data[['Tumor_Sample_Barcode', 'Hugo_Symbol']], index='Tumor_Sample_Barcode',
                             columns='Hugo_Symbol', aggfunc=len, fill_value=0)
    # Convert non-zero values to 1
    df_wide[df_wide > 0] = 1  # shape of df_wide: (290, 14)
    # There are 368 samples with matched RNA-seq and DNA seq, but only 290 samples with mutations
    # in the 15 driver genes, so I need to fill in the missing 78 samples with 0s
    missing_samples = list(total_samples - set(df_wide.index))
    df_missing = pd.DataFrame(0, index=missing_samples, columns=df_wide.columns)
    df = pd.concat([df_wide, df_missing], axis=0)
    df.index = df.index.str[8:15]
    df.to_csv(f'{data_path}/TCGA_LUAD_sample_mutation_matrix.csv')


    # ----------------------Imputed TCGA LUAD Isotopologue v.s. Mutation------------------------
    results_dir = f'/data1/reznike/xiea1/MetabolicModel/results_RNA_imputation/LUAD'
    imputed_mean_met = pd.read_csv(f'{results_dir}/LUAD_isotope_imputed_mean_met_flipped.csv', index_col=0, header=0)
    imputed_mean_met = imputed_mean_met[sorted(imputed_mean_met.columns)]
    imputed_mean_met.index = imputed_mean_met.index.str[8:15]

    # get mutation data
    mutation = pd.read_csv(
        '/data1/reznike/xiea1/MetabolicModel/data/TCGA_downstream/TCGA_LUAD_sample_mutation_matrix.csv',
        index_col=0, header=0)

    # Wilcoxon rank sum test
    mean_data = pd.concat([mutation, imputed_mean_met], axis=1, join='inner')
    mean_data.to_csv(f'{results_dir}/downstream_analysis/mut_isotope_flipped.csv')
    for gene in mutation.columns:
        print(gene)
        met_wilcox_df = pd.DataFrame({'mean difference': [], 'p': []})
        for metabolite in imputed_mean_met.columns:
            mut = mean_data.loc[mean_data[gene] == 1, metabolite]
            wt = mean_data.loc[mean_data[gene] == 0, metabolite]
            met_wilcox_df.loc[metabolite, 'mean difference'] = mut.mean() - (
                wt.mean())  # used flipped data, up in mut on the right in plots
            met_wilcox_df.loc[metabolite, 'p'] = scipy.stats.ranksums(mut, wt).pvalue
        met_wilcox_df['p_adj'] = multipletests(pvals=met_wilcox_df['p'], method="fdr_bh", alpha=0.1)[1]
        met_wilcox_df.to_csv(f'{results_dir}/downstream_analysis/wilcox_df/wilcox_{gene}.csv')


    #------------------------------------Heatmap----------------------------------
    mutation_list = ['EGFR', 'KRAS', 'KEAP1', 'STK11', 'TP53', 'SMARCA4', 'SETD2']
    met_wilcox = pd.DataFrame(dict(zip(mutation_list, [[]] * len(mutation_list))))
    fdr_wilcox = pd.DataFrame(dict(zip(mutation_list, [[]] * len(mutation_list))))
    for gene in mutation_list:
        met_wilcox_df = pd.read_csv(
            f'{results_dir}/downstream_analysis/wilcox_df/wilcox_{gene}.csv',
            index_col=0, header=0)
        met_wilcox[gene] = met_wilcox_df['mean difference']
        fdr_wilcox[gene] = met_wilcox_df['p_adj']

    met_wilcox.to_csv(f'{results_dir}/downstream_analysis/heatmap/wilcox_mean_difference_matrix.csv')
    fdr_wilcox.to_csv(f'{results_dir}/downstream_analysis/heatmap/wilcox_fdr_matrix.csv')
    # grey out insignificant values
    met_wilcox_masked = np.where(fdr_wilcox >= 0.1, np.nan, met_wilcox)
    met_wilcox_masked = pd.DataFrame(met_wilcox_masked, index=met_wilcox.index, columns=met_wilcox.columns)

    # plot heatmap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    fig, ax = plt.subplots(figsize=(30, 20))
    sns.heatmap(met_wilcox_masked, ax=ax, annot=True, cmap=cmap,
                linewidths=0.1, linecolor='black', fmt=".2f")
    ax = plt.gca()
    plt.xlabel("Mutations")
    plt.ylabel("Isotopologues")
    plt.title("FDR value of mutations in metabolite ~ mutation")
    plt.savefig(f'{results_dir}/downstream_analysis/heatmap/LUAD_heatmap.pdf')
    plt.close()


    # heatmap with p value annotations
    def assign_annotation(value):
        if value >= 0.1:
            return ''
        elif value < 0.001:
            return '***'
        elif 0.001 <= value < 0.01:
            return '**'
        elif 0.01 <= value < 0.1:
            return '*'
    fdr_annotations = fdr_wilcox.map(assign_annotation)
    sns.clustermap(data=met_wilcox, cmap="coolwarm",
                   annot=fdr_annotations,
                   annot_kws={'size': 14, 'fontweight': 'bold', "va": "center"},
                   fmt="", figsize=(15, 30), vmin=-0.3,
                   vmax=0.3)
    plt.xlabel("Mutations")
    plt.ylabel("Isotopologues")
    plt.title("FDR value of mutations in isotopologue ~ mutation")
    plt.savefig(f'{results_dir}/downstream_analysis/heatmap/LUAD_heatmap_clustered.pdf')
    plt.close()


