import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import scipy.stats
from statsmodels.stats.multitest import multipletests
import numpy as np
from Performance_Benchmarking.scripts.utils import re_rank_2D
from scripts.data_processing import tic_normalization_across

if __name__ == "__main__":
    file_path = "/juno/work/reznik/xiea1/MetabolicModel"
    imputation_dir = 'TCGA'
    results_dir = f'{file_path}/results_RNA_imputation/{imputation_dir}'
    embedding_dir = f'{results_dir}/embeddings'
    plots_dir = f'{results_dir}/plots'
    batch_size = 606  # TCGA:606, IMmotion151: 823

    # Check standard deviation of TCGA imputed results
    #---------------------------------------------------------------------------------------------------------
    rank_hat_draws_std = np.load(f'{embedding_dir}/rank_hat_draws_std_met.npy')[:batch_size, :]/batch_size
    rank_hat_draws_std_ave = np.mean(rank_hat_draws_std, axis=0)
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.hist(rank_hat_draws_std_ave)
    plt.xlabel('average standard deviation of posterior predicted metabolites')
    plt.ylabel('count')
    plt.title('TCGA histogram of standard deviation')
    #plt.show()
    plt.savefig(f'{plots_dir}/predicted_standard deviation_histogram.pdf')
    plt.close()

    #--------------------------Get imputed TCGA metabolites results----------------------------------
    rank_hat_draws_mean = np.load(f'{embedding_dir}/rank_hat_draws_mean_met.npy')[:batch_size, :]
    rank_hat_draws_std = np.load(f'{embedding_dir}/rank_hat_draws_std_met.npy')[:batch_size, :] / batch_size
    # rerank metabolite ranks (now rank 0 means the largest value, # 1 means the smallest value)
    rank_hat_draws_mean = rank_hat_draws_mean.argsort(
        axis=0, kind='stable').argsort(axis=0, kind='stable')
    rank_hat_draws_mean = rank_hat_draws_mean / rank_hat_draws_mean.shape[0]  # transformed all ranks to [0,1]
    normalized_data_met = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv',
                                  index_col=0, header=0).iloc[:batch_size, :rank_hat_draws_mean.shape[1]]
    imputed_mean_met = pd.DataFrame(rank_hat_draws_mean, index=normalized_data_met.index,columns=normalized_data_met.columns)
    imputed_std_met = pd.DataFrame(rank_hat_draws_std, index=normalized_data_met.index,columns=normalized_data_met.columns)
    imputed_mean_met.to_csv(f'{results_dir}/TCGA_imputed_mean_met.csv')
    imputed_std_met.to_csv(f'{results_dir}/TCGA_imputed_std_met.csv')
    # rerank metabolite ranks in the flipped way (now rank 0 means the smallest value, # 1 means the largest value)
    rank_hat_draws_mean = re_rank_2D(rank_hat_draws_mean)
    rank_hat_draws_mean = rank_hat_draws_mean / rank_hat_draws_mean.shape[0]  # transformed all ranks to [0,1]
    normalized_data_met = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv',
                                  index_col=0, header=0).iloc[:batch_size, :rank_hat_draws_mean.shape[1]]
    imputed_mean_met = pd.DataFrame(rank_hat_draws_mean, index=normalized_data_met.index,columns=normalized_data_met.columns)
    imputed_mean_met.to_csv(f'{results_dir}/TCGA_imputed_mean_met_flipped.csv')



    # -------------------------TCGA: Mutation vs Metabolite Analysis----------------------------------
    imputed_mean_met = pd.read_csv(f'{results_dir}/TCGA_imputed_mean_met.csv', index_col=0, header=0)
    imputed_std_met = pd.read_csv(f'{results_dir}/TCGA_imputed_std_met.csv', index_col=0, header=0)
    # subset imputed data to only include reproducibly well-predicted metabolites
    reproducibly_well_predicted = pd.read_csv(f'{file_path}/results_RNA_ccRCC/reproducibly_well_predicted_metabolites.csv'
                                              , index_col=0, header=0)
    imputed_mean_met = imputed_mean_met[reproducibly_well_predicted.index]
    imputed_std_met = imputed_std_met[reproducibly_well_predicted.index]
    imputed_mean_met.index = imputed_mean_met.index.str[8:15]
    imputed_std_met.index = imputed_std_met.index.str[8:15]

    # get mutation data
    mutation = pd.read_csv('/juno/work/reznik/xiea1/MetabolicModel/data/TCGA_downstream/TCGA_KIRC_sample_mutation_matrix.csv',
                           index_col=0, header=0)
    imputed_mean_met = imputed_mean_met[imputed_mean_met.index.isin(mutation.index)]
    imputed_std_met = imputed_std_met[imputed_std_met.index.isin(mutation.index)]
    imputed_mean_met = imputed_mean_met.loc[mutation.index]
    imputed_std_met = imputed_std_met.loc[mutation.index]

    # Wilcoxon rank sum test
    mean_data = pd.concat([mutation, imputed_mean_met], axis=1)
    mean_data.to_csv(f'{results_dir}/downstream_analysis/wilcox_df_tcga/mut_met_flipped.csv')
    std_data = pd.concat([mutation, imputed_std_met], axis=1)
    df_list = []
    for gene in mutation.columns:
        print(gene)
        met_wilcox_df = pd.DataFrame({'mean difference': [], 'p': [], 'pooled std mut': [], 'pooled std wt': []})
        for metabolite in imputed_mean_met.columns:
            mut = mean_data.loc[mean_data[gene] == 1, metabolite]
            wt = mean_data.loc[mean_data[gene] == 0, metabolite]
            met_wilcox_df.loc[metabolite, 'mean difference'] = wt.mean() - (mut.mean())  # up in mut on the right in plots
            met_wilcox_df.loc[metabolite, 'p'] = scipy.stats.ranksums(mut, wt).pvalue
            mut_std = np.sqrt(np.mean(np.square(std_data.loc[std_data[gene] == 1, metabolite])))
            wt_std = np.sqrt(np.mean(np.square(std_data.loc[std_data[gene] == 0, metabolite])))
            met_wilcox_df.loc[metabolite, 'pooled std mut'] = mut_std
            met_wilcox_df.loc[metabolite, 'pooled std wt'] = wt_std
        met_wilcox_df['p_adj'] = multipletests(pvals=met_wilcox_df['p'], method="fdr_bh", alpha=0.1)[1]
        met_wilcox_df.to_csv(f'{results_dir}/downstream_analysis/wilcox_df_tcga/wilcox_{gene}_62met.csv')
        df_list.append(met_wilcox_df)

    # Calculate the frequency of Mutations in TCGA samples
    mean_data = pd.read_csv(f'{results_dir}/downstream_analysis/wilcox_df_tcga/mut_met_flipped.csv',
                            index_col=0, header=0)

    # Volcano plots
    for gene in mutation.columns:
        df = pd.read_csv(f'{results_dir}/downstream_analysis/wilcox_df_tcga/wilcox_{gene}_62met.csv', index_col=0, header=0)
        # plot volcano plot
        df['significant'] = df['p_adj'] < 0.1
        plt.rcParams['figure.figsize'] = [10, 10]
        scatterplot = sns.scatterplot(x=df['mean difference'], y=-np.log10(df['p_adj']), hue=df['significant'])
        # Add labels to the dots
        for sig_gene in df[df['significant']].index:
            scatterplot.annotate(sig_gene, (df.loc[sig_gene, 'mean difference'], -np.log10(df.loc[sig_gene, 'p_adj'])))
        plt.xlabel('mean difference')
        plt.ylabel('-log10(p-value)')
        plt.title(f'Volcano plot of {gene}')
        plt.savefig(f'{results_dir}/downstream_analysis/volcano_plots_tcga/volcano_{gene}_62met.pdf')
        plt.close()

    # -------------------------CPTAC: Mutation vs Metabolite Analysis----------------------------------
    # Validation in CPTAC
    reproducibly_well_predicted = pd.read_csv(f'{file_path}/results_RNA_ccRCC/reproducibly_well_predicted_metabolites.csv'
                                              , index_col=0, header=0)
    cptac = pd.read_csv(f'/juno/work/reznik/xiea1/MetabolicModel/data/MET_matched_ccRCC/matched_Harmonized_Met_CPTAC.csv',
                        index_col=0, header=0)
    # subset imputed data to only include reproducibly well-predicted metabolites
    cptac = cptac[reproducibly_well_predicted[reproducibly_well_predicted['CPTAC'].notna()].index]  # 61 mets
    cptac.index = cptac.index.str[:-2]
    # TIC
    cptac = pd.DataFrame(tic_normalization_across(cptac.to_numpy(), np.zeros(cptac.shape[0]).astype(int)),
                         index=cptac.index, columns=cptac.columns)
    # rank metabolite ranks in the flipped way (0 means the smallest value, 1 means the largest value)
    cptac = pd.DataFrame(cptac.to_numpy().argsort(axis=0, kind='stable').argsort(
        axis=0, kind='stable'), index=cptac.index, columns=cptac.columns)
    cptac = cptac / cptac.shape[0]  # transformed all ranks to [0,1]
    # get mutation data
    mutation = pd.read_csv('/juno/work/reznik/xiea1/MetabolicModel/data/CPTAC_DNA/CPTAC_KIRC_sample_mutation_matrix.csv',
                           index_col=0, header=0)
    mean_data = pd.concat([mutation, cptac], axis=1, join='inner')
    mean_data.to_csv(f'{results_dir}/downstream_analysis/wilcox_df_cptac/mut_met_flipped.csv')


    # Validation in CPTAC_val
    cptac_val = pd.read_csv(f'/juno/work/reznik/xiea1/MetabolicModel/data/MET_matched_ccRCC/matched_Harmonized_Met_CPTAC_val.csv',
                            index_col=0, header=0)
    cptac_val = cptac_val[reproducibly_well_predicted[reproducibly_well_predicted['CPTAC_val'].notna()].index]  # 59 mets
    # subset cptac_val to only include tumor samples based on the index
    cptac_val = cptac_val[cptac_val.index.str.contains('-T')]
    cptac_val.index = cptac_val.index.str[:-2]
    # TIC
    cptac_val = pd.DataFrame(tic_normalization_across(cptac_val.to_numpy(), np.zeros(cptac_val.shape[0]).astype(int)),
                         index=cptac_val.index, columns=cptac_val.columns)
    # rank metabolite ranks in the flipped way (0 means the smallest value, 1 means the largest value)
    cptac_val = pd.DataFrame(cptac_val.to_numpy().argsort(axis=0, kind='stable').argsort(
        axis=0, kind='stable'), index=cptac_val.index, columns=cptac_val.columns)
    cptac_val = cptac_val / cptac_val.shape[0]  # transformed all ranks to [0,1]
    # get mutation data
    mutation = pd.read_csv('/juno/work/reznik/xiea1/MetabolicModel/data/CPTAC_DNA/CPTAC_val_KIRC_sample_mutation_matrix.csv',
                           index_col=0, header=0)
    mean_data = pd.concat([mutation, cptac], axis=1, join='inner')
    mean_data.to_csv(f'{results_dir}/downstream_analysis/wilcox_df_cptac_val/mut_met_flipped.csv')



    # Wilcoxon rank sum test
    df_list = []
    for gene in mutation.columns:
        print(gene)
        met_wilcox_df = pd.DataFrame({'mean difference': [], 'p': []})
        for metabolite in cptac_val.columns:
            mut = mean_data.loc[mean_data[gene] == 1, metabolite]
            wt = mean_data.loc[mean_data[gene] == 0, metabolite]
            met_wilcox_df.loc[metabolite, 'mean difference'] = wt.mean() - (mut.mean())
            met_wilcox_df.loc[metabolite, 'p'] = scipy.stats.ranksums(mut, wt).pvalue
        met_wilcox_df['p_adj'] = multipletests(pvals=met_wilcox_df['p'], method="fdr_bh", alpha=0.1)[1]
        met_wilcox_df.to_csv(f'{results_dir}/downstream_analysis/wilcox_df_cptac_val/wilcox_{gene}_59met.csv')
        df_list.append(met_wilcox_df)

    #--------------------------IMmotion151: Get imputed TCGA metabolites results----------------------------------
    rank_hat_draws_mean = np.load(f'{embedding_dir}/rank_hat_draws_mean_met.npy')[:batch_size, :]
    rank_hat_draws_std = np.load(f'{embedding_dir}/rank_hat_draws_std_met.npy')[:batch_size, :] / batch_size
    # rerank metabolite ranks
    rank_hat_draws_mean = rank_hat_draws_mean.argsort(
        axis=0, kind='stable').argsort(axis=0, kind='stable')
    rank_hat_draws_mean = rank_hat_draws_mean / rank_hat_draws_mean.shape[0]  # transformed all ranks to [0,1]
    normalized_data_met = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv',
                                  index_col=0, header=0).iloc[:batch_size, :rank_hat_draws_mean.shape[1]]
    imputed_mean_met = pd.DataFrame(rank_hat_draws_mean, index=normalized_data_met.index,columns=normalized_data_met.columns)
    imputed_std_met = pd.DataFrame(rank_hat_draws_std, index=normalized_data_met.index,columns=normalized_data_met.columns)
    imputed_mean_met.to_csv(f'{results_dir}/IMmotion151_imputed_mean_met.csv')
    imputed_std_met.to_csv(f'{results_dir}/IMmotion151_imputed_std_met.csv')

    # subset imputed data to only include reproducibly well-predicted metabolites
    reproducibly_well_predicted = pd.read_csv(
        f'{file_path}/results_RNA_ccRCC/reproducibly_well_predicted_metabolites.csv'
        , index_col=0, header=0)
    imputed_mean_met = imputed_mean_met[reproducibly_well_predicted.index]  # 62 mets
    imputed_std_met = imputed_std_met[reproducibly_well_predicted.index]  # 62 mets

    # get mutation data
    mutation = pd.read_csv(
        '/juno/work/reznik/xiea1/MetabolicModel/data/IMmotion151/IMmotion151_sample_mutation_matrix.csv',
        index_col=0, header=0)
    imputed_mean_met = imputed_mean_met[imputed_mean_met.index.isin(mutation.index)]
    imputed_std_met = imputed_std_met[imputed_std_met.index.isin(mutation.index)]
    imputed_mean_met = imputed_mean_met.loc[mutation.index]
    imputed_std_met = imputed_std_met.loc[mutation.index]

    # Wilcoxon rank sum test
    mean_data = pd.concat([mutation, imputed_mean_met], axis=1)
    std_data = pd.concat([mutation, imputed_std_met], axis=1)
    df_list = []
    for gene in mutation.columns:
        print(gene)
        met_wilcox_df = pd.DataFrame({'mean difference': [], 'p': [], 'pooled std mut': [], 'pooled std wt': []})
        for metabolite in imputed_mean_met.columns:
            mut = mean_data.loc[mean_data[gene] == 1, metabolite]
            wt = mean_data.loc[mean_data[gene] == 0, metabolite]
            met_wilcox_df.loc[metabolite, 'mean difference'] = wt.mean() - (mut.mean())  # up in mut on the right in plots
            met_wilcox_df.loc[metabolite, 'p'] = scipy.stats.ranksums(mut, wt).pvalue
            mut_std = np.sqrt(np.mean(np.square(std_data.loc[std_data[gene] == 1, metabolite])))
            wt_std = np.sqrt(np.mean(np.square(std_data.loc[std_data[gene] == 0, metabolite])))
            met_wilcox_df.loc[metabolite, 'pooled std mut'] = mut_std
            met_wilcox_df.loc[metabolite, 'pooled std wt'] = wt_std
        met_wilcox_df['p_adj'] = multipletests(pvals=met_wilcox_df['p'], method="fdr_bh", alpha=0.1)[1]
        met_wilcox_df.to_csv(f'{file_path}/results_RNA_imputation/TCGA/downstream_analysis/wilcox_df_151/wilcox_{gene}_62met.csv')
        df_list.append(met_wilcox_df)










