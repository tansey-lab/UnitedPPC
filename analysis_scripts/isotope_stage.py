import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import scipy.stats
from statsmodels.stats.multitest import multipletests
import numpy as np
import statsmodels.api as sm
import seaborn as sns

if __name__ == "__main__":
    # ------------------------TCGA KIRC: isotopologue ~ stage------------------------
    file_path = "/juno/work/reznik/xiea1/MetabolicModel"
    results_dir = f'{file_path}/results_RNA_imputation/KIRC_isotope_pyr'
    imputed_mean_met = pd.read_csv(f'{results_dir}/KIRC_isotope_imputed_mean_met_flipped.csv', index_col=0, header=0)
    imputed_mean_met = imputed_mean_met[sorted(imputed_mean_met.columns)]
    # include only ccRCC tumor samples
    sample_type_kipan = pd.read_csv(f'{file_path}/data/RNA_raw_imputation/KIPAN_sample_type.csv',
                                    index_col=0, header=0)
    sample_type_kipan = sample_type_kipan.loc[sample_type_kipan['code'].isin(['01A', '01B'])]
    merged = pd.concat([imputed_mean_met, sample_type_kipan], axis=1, join='inner')
    imputed_mean_met = imputed_mean_met.loc[merged.index]
    imputed_mean_met.index = [brc[:12].lower() for brc in imputed_mean_met.index]
    # subset to 9 interpretable isotopologues
    imputed_mean_met_keep = imputed_mean_met[['Glucose m+6', '3-Phosphoglycerate m+3', 'Lactate m+3', 'Alanine m+3',
                                'Citrate m+2', 'Succinate m+2', 'Malate m+2',
                                'Glutamate m+2', 'Aspartate m+2']]

    # Load sample stage clinical data
    clinical = pd.read_csv(f'{file_path}/data/RNA_raw_imputation/KIRC.clin.merged.picked.txt',
                                 delimiter="\t",header=0, index_col=0).T
    mean_data = pd.concat([imputed_mean_met_keep, clinical], axis=1, join='inner')
    # change the order of x axis to ['stage i', 'stage ii', 'stage iii', 'stage iv']
    mean_data['pathologic_stage'] = mean_data['pathologic_stage'].astype('category')
    mean_data['pathologic_stage'] = mean_data['pathologic_stage'].cat.reorder_categories(['stage i', 'stage ii', 'stage iii', 'stage iv'])
    # filter out rows with nan in pathologic_stage
    mean_data = mean_data.dropna(subset=['pathologic_stage'])
    mean_data.to_csv('/juno/work/reznik/xiea1/MetabolicModel/final_results/data/fig_5/KIRC_isotope_stage.csv')

    # Plot boxplots showing isotopologue abundance by stage
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for i, col in enumerate(imputed_mean_met_keep.columns):
        sns.boxplot(x="pathologic_stage", y=col, data=mean_data, ax=axes[i])
        # differential abundance test of 4 groups (stage i, ii, iii, iv)
        statistic, p_value = scipy.stats.kruskal(mean_data.loc[mean_data['pathologic_stage'] == 'stage i', col],
                                    mean_data.loc[mean_data['pathologic_stage'] == 'stage ii', col],
                                    mean_data.loc[mean_data['pathologic_stage'] == 'stage iii', col],
                                    mean_data.loc[mean_data['pathologic_stage'] == 'stage iv', col])
        # add p values of the Kruskal test to the plot
        axes[i].text(0.05, 1, f'p={p_value:.2e}', transform=axes[i].transAxes, fontsize=12,
                     verticalalignment='top')
        sns.stripplot(x="pathologic_stage", y=col, data=mean_data, ax=axes[i], jitter=True, alpha=0.5, color='grey')
        axes[i].set_title(col)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/KIRC_isotope_stage_boxplots.pdf')
    plt.close()


