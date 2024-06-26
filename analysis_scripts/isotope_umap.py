import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import numpy as np
import umap
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # ----------------------Load W embeddings------------------------
    results_dir = f'/juno/work/reznik/xiea1/MetabolicModel/results_RNA_imputation/KIPAN'
    embedding_dir = f'{results_dir}/embeddings'
    batch_size = 1020  # KIPAN: 1020

    with open(f'{embedding_dir}/W_H_loc_scale.npy', 'rb') as f:
        W_loc = np.load(f)
        W_scale = np.load(f)
        H_loc = np.load(f)
        H_scale = np.load(f)
        f.close()
    normalized_data_met = pd.read_csv(f'{embedding_dir}/normalized_met_rna_data_pyro.csv',
                                      index_col=0, header=0)  # 1091 = 1020 + 71 samples
    pd.DataFrame(W_loc, index=normalized_data_met.index).to_csv(f'{embedding_dir}/W_loc.csv')

    # W embeddings in a UMAP
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(W_loc)
    # W embeddings in a PCA
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(W_loc)
    # Umap based on RNA seq data
    len_metabolite = 51
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(normalized_data_met.iloc[:, len_metabolite:].to_numpy())
    # log RNA in PCA
    len_metabolite = 51
    pca = PCA(n_components=2)
    log_rna = np.log(normalized_data_met.iloc[:batch_size, len_metabolite:].to_numpy())
    rna = np.concatenate([log_rna, normalized_data_met.iloc[batch_size:, len_metabolite:].to_numpy()], axis=0)
    # drop the columns that there are inf, -inf, or nan
    rna = rna[:, ~np.any(np.isinf(rna), axis=0)]
    rna = rna[:, ~np.any(np.isnan(rna), axis=0)]
    embedding = pca.fit_transform(rna)


    # ----------------------Look into W embeddings of imputed KIPAN tumor data------------------------
    embedding_kipan = pd.DataFrame(embedding[:batch_size, :], index=normalized_data_met.iloc[:batch_size, :].index)
    data_path = '/juno/work/reznik/xiea1/MetabolicModel/data/RNA_raw_imputation'
    sample_info_kipan = pd.read_csv(f'{data_path}/KIPAN.clin.merged.picked.txt',
                                    sep='\t', index_col=0, header=0).T
    sample_type_kipan = pd.read_csv(f'{data_path}/KIPAN_sample_type.csv',
                                    index_col=0, header=0)

    # exclude two new primary tumor samples with code '05A', because their sample names are duplicated with two '01A' samples
    sample_type_kipan = sample_type_kipan.loc[sample_type_kipan['code'].isin(['01A', '01B', '11A'])]
    sample_type_kipan_normal = sample_type_kipan.loc[sample_type_kipan['TN'] == 'Normal']
    sample_type_kipan_normal['histological_type'] = 'Normal'
    sample_type_kipan_tumor = sample_type_kipan.loc[sample_type_kipan['TN'] == 'Tumor']
    sample_type_kipan_tumor['histological_type'] = sample_info_kipan.loc[[i[:12].lower() for i in sample_type_kipan_tumor.index], 'histological_type'].to_list()
    sample_type_kipan = pd.concat([sample_type_kipan_normal, sample_type_kipan_tumor], axis=0)
    kipan = pd.concat([embedding_kipan, sample_type_kipan['histological_type']], join='inner', axis=1)

    # plot W embeddings in a UMAP
    colors = {'kidney clear cell renal carcinoma': 'red', 'kidney chromophobe': 'green',
              'kidney papillary renal cell carcinoma': 'blue', 'Normal': 'black'}
    plt.rcParams['figure.figsize'] = [4, 5]
    for color, label in [('red', 'kidney clear cell renal carcinoma'),
                         ('blue', 'kidney papillary renal cell carcinoma'),
                         ('green', 'kidney chromophobe'),
                         ('black', 'Normal')]:
        indices = kipan['histological_type'] == label
        plt.scatter(kipan.to_numpy()[indices, 0], kipan.to_numpy()[indices, 1], c=color, label=label, s=10, alpha=0.3)

    legend_labels = ['kidney clear cell renal carcinoma', 'kidney chromophobe', 'kidney papillary renal cell carcinoma', 'Normal']
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
                       for label, color in colors.items()]
    plt.legend(handles=legend_elements, labels=legend_labels, title='Histological type',
               loc='lower right', bbox_to_anchor=(1, 1))
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig('/juno/work/reznik/xiea1/MetabolicModel/final_results/figures/fig_4/fig_4b_pca_rna_KIPAN_all.pdf')
    plt.close()

    # ----------------------Look into W embeddings of the measured MITO tumor data------------------------
    embedding_mito = pd.DataFrame(embedding[batch_size:, :], index=normalized_data_met.iloc[batch_size:, :].index)
    data_path = '/juno/work/reznik/xiea1/MetabolicModel/data/Isotope_raw_mito'

    sample_info_mito = pd.read_csv(f'{data_path}/isotope_sample_subtype_info.csv', index_col=0, header=0)
    sample_info_mito['Subtype'] = sample_info_mito['Subtype'].replace(' ', 'Normal')
    mito = pd.concat([embedding_mito, sample_info_mito['Subtype']], join='inner', axis=1)

    # Plot the umap
    # plot W embeddings in a UMAP
    colors = {'CCRCC': 'red', 'Chromophobe': 'green', 'Papillary': 'blue', 'Normal': 'black', 'Oncocytoma': 'purple'}
    plt.rcParams['figure.figsize'] = [4, 5]
    for color, label in [('red', 'CCRCC'),
                         ('blue', 'Papillary'),
                         ('green', 'Chromophobe'),
                         ('black', 'Normal'),
                         ('purple', 'Oncocytoma')]:
        indices = mito['Subtype'] == label
        plt.scatter(mito.to_numpy()[indices, 0], mito.to_numpy()[indices, 1], c=color, label=label, s=10, alpha=0.3)

    legend_labels = ['CCRCC', 'Chromophobe', 'Papillary', 'Normal', 'Oncocytoma']
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
                       for label, color in colors.items()]
    plt.legend(handles=legend_elements, labels=legend_labels, title='Histological type',
               loc='lower right', bbox_to_anchor=(1, 1))
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig('/juno/work/reznik/xiea1/MetabolicModel/final_results/figures/fig_4/fig_4b_pca_rna_mito_all.pdf')
    plt.close()


    # ----------------------Look into W embeddings of imputed KIPAN + MITO all data (tumor + normal)------------------------
    colors = ['red'] * 1020 + ['blue'] * 71
    color_map = {'KIPAN': 'red', 'MITO': 'blue'}
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors,  s=10)

    legend_labels = ['KIPAN', 'MITO']
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
                       for label, color in color_map.items()]
    plt.legend(handles=legend_elements, labels=legend_labels, title='Dataset',
               loc='lower right', bbox_to_anchor=(1, 1))
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig('/juno/work/reznik/xiea1/MetabolicModel/final_results/figures/fig_4/fig_4b_pca_rna_KIPAN_MITO_all.pdf')
    plt.close()

    # ----------------------Exclude mismatched samples ------------------------
    # Get the mismatched samples from umap RNA seq data only
    # Umap based on RNA seq data
    len_metabolite = 51
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(normalized_data_met.iloc[:, len_metabolite:].to_numpy())
    embedding_kipan = pd.DataFrame(embedding[:batch_size, :], index=normalized_data_met.iloc[:batch_size, :].index)
    embedding_mito = pd.DataFrame(embedding[batch_size:, :], index=normalized_data_met.iloc[batch_size:, :].index)

    # Get mismatched samples in KIPAN
    kipan = pd.concat([embedding_kipan, sample_type_kipan['histological_type']], join='inner', axis=1)
    # plot W embeddings in a UMAP
    colors = {'kidney clear cell renal carcinoma': 'red', 'kidney chromophobe': 'green',
              'kidney papillary renal cell carcinoma': 'blue', 'Normal': 'black'}
    plt.rcParams['figure.figsize'] = [4, 5]
    for color, label in [('red', 'kidney clear cell renal carcinoma'),
                         ('blue', 'kidney papillary renal cell carcinoma'),
                         ('green', 'kidney chromophobe'),
                         ('black', 'Normal')]:
        indices = kipan['histological_type'] == label
        plt.scatter(kipan.to_numpy()[indices, 0], kipan.to_numpy()[indices, 1], c=color, label=label, s=10, alpha=0.3)

    legend_labels = ['kidney clear cell renal carcinoma', 'kidney chromophobe', 'kidney papillary renal cell carcinoma',
                     'Normal']
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
                       for label, color in colors.items()]
    plt.legend(handles=legend_elements, labels=legend_labels, title='Histological type',
               loc='lower right', bbox_to_anchor=(1, 1))
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.show()

    # plot each cancer type separately to see the mismatched samples
    kipan_prcc = kipan.loc[kipan['histological_type'] == 'kidney papillary renal cell carcinoma']
    plt.scatter(kipan_prcc.iloc[:, :2].to_numpy()[:, 0], kipan_prcc.iloc[:, :2].to_numpy()[:, 1], s=10, alpha=1)
    plt.show()
    kipan_chrcc = kipan.loc[kipan['histological_type'] == 'kidney chromophobe']
    plt.scatter(kipan_chrcc.iloc[:, :2].to_numpy()[:, 0], kipan_chrcc.iloc[:, :2].to_numpy()[:, 1], s=10, alpha=1)
    plt.show()
    kipan_ccrcc = kipan.loc[kipan['histological_type'] == 'kidney clear cell renal carcinoma']
    plt.scatter(kipan_ccrcc.iloc[:, :2].to_numpy()[:, 0], kipan_ccrcc.iloc[:, :2].to_numpy()[:, 1], s=10, alpha=1)
    plt.show()
    kipan_normal = kipan.loc[kipan['histological_type'] == 'Normal']
    plt.scatter(kipan_normal.iloc[:, :2].to_numpy()[:, 0], kipan_normal.iloc[:, :2].to_numpy()[:, 1], s=10, alpha=1)
    plt.show()

    mismatched_kipan = []
    mismatched_kipan.extend(kipan_prcc.loc[(kipan_prcc[0] > 10) | (kipan_prcc[0] < 5) |
                                           (kipan_prcc[1] > 5) | (kipan_prcc[1] < 1.25)].index.to_list())
    mismatched_kipan.extend(kipan_chrcc.loc[(kipan_chrcc[0] > 14)].index.to_list())
    mismatched_kipan.extend(kipan_ccrcc.loc[(kipan_ccrcc[0] < 12)].index.to_list())
    mismatched_kipan.extend(kipan_normal.loc[(kipan_normal[0] > 7.5)].index.to_list())

    # Get mismatched samples in MITO
    mito = pd.concat([embedding_mito, sample_info_mito['Subtype']], join='inner', axis=1)
    # plot W embeddings in a UMAP
    colors = {'CCRCC': 'red', 'Chromophobe': 'green', 'Papillary': 'blue', 'Normal': 'black', 'Oncocytoma': 'purple'}
    plt.rcParams['figure.figsize'] = [4, 5]
    for color, label in [('red', 'CCRCC'),
                         ('blue', 'Papillary'),
                         ('green', 'Chromophobe'),
                         ('black', 'Normal'),
                         ('purple', 'Oncocytoma')]:
        indices = mito['Subtype'] == label
        plt.scatter(mito.to_numpy()[indices, 0], mito.to_numpy()[indices, 1], c=color, label=label, s=10, alpha=0.6)

    legend_labels = ['CCRCC', 'Chromophobe', 'Papillary', 'Normal', 'Oncocytoma']
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
                       for label, color in colors.items()]
    plt.legend(handles=legend_elements, labels=legend_labels, title='Histological type',
               loc='lower right', bbox_to_anchor=(1, 1))
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.show()


    mito_ccrcc = mito.loc[mito['Subtype'] == 'CCRCC']
    mito_normal = mito.loc[mito['Subtype'] == 'Normal']
    mismatched_mito = []
    mismatched_mito.extend(mito_ccrcc.loc[mito_ccrcc[0] > -8.2].index.to_list())
    mismatched_mito.extend(mito_normal.loc[(mito_normal[0] < -8.2)].index.to_list())

    mismatched_samples = mismatched_kipan + mismatched_mito
    mismatched_samples = set(mismatched_samples)  # 58 mismatched samples

    # ----------------------Get sample type annotation for MITO & KIPAN------------------------
    # Get sample type annotation for KIPAN
    data_path = '/juno/work/reznik/xiea1/MetabolicModel/data/RNA_raw_imputation'
    sample_info_kipan = pd.read_csv(f'{data_path}/KIPAN.clin.merged.picked.txt',
                                    sep='\t', index_col=0, header=0).T
    sample_type_kipan = pd.read_csv(f'{data_path}/KIPAN_sample_type.csv',
                                    index_col=0, header=0)
    # sace separately two new primary tumor samples with code '05A', because their sample names are duplicated with two '01A' samples
    sample_type_kipan_dropped = sample_type_kipan.loc[sample_type_kipan['code'] == '05A']
    # two '05A' samples: 'tcga-dv-a4w0', 'tcga-uz-a9ps'
    sample_type_kipan_dropped['histological_type'] = sample_info_kipan.loc[
        ['tcga-dv-a4w0', 'tcga-uz-a9ps'], 'histological_type'].to_list()
    # exclude two new primary tumor samples with code '05A', because their sample names are duplicated with two '01A' samples
    sample_type_kipan = sample_type_kipan.loc[sample_type_kipan['code'].isin(['01A', '01B', '11A'])]
    sample_type_kipan_normal = sample_type_kipan.loc[sample_type_kipan['TN'] == 'Normal']
    sample_type_kipan_normal['histological_type'] = 'Normal'
    sample_type_kipan_tumor = sample_type_kipan.loc[sample_type_kipan['TN'] == 'Tumor']
    sample_type_kipan_tumor['histological_type'] = sample_info_kipan.loc[
        [i[:12].lower() for i in sample_type_kipan_tumor.index], 'histological_type'].to_list()
    sample_type_kipan = pd.concat([sample_type_kipan_normal, sample_type_kipan_tumor, sample_type_kipan_dropped],
                                  axis=0)
    # reorder the index to match the W embeddings
    sample_type_kipan.reindex(normalized_data_met.iloc[:batch_size, :].index, inplace=True)
    sample_type_kipan['batch'] = 'KIPAN'


    # Get sample type annotation for MITO
    data_path = '/juno/work/reznik/xiea1/MetabolicModel/data/Isotope_raw_mito'
    sample_info_mito = pd.read_csv(f'{data_path}/isotope_sample_subtype_info.csv', index_col=0, header=0)
    sample_info_mito['Subtype'] = sample_info_mito['Subtype'].replace(' ', 'Normal')
    sample_info_mito.reindex([normalized_data_met.iloc[batch_size:, :].index], inplace=True)
    sample_info_mito.rename(columns={'Subtype': 'histological_type'}, inplace=True)
    # if histological_type == 'Normal', then TN = 'Normal', else TN = 'Tumor'
    sample_info_mito['TN'] = sample_info_mito['histological_type'].apply(lambda x: 'Normal' if x == 'Normal' else 'Tumor')
    sample_info_mito['batch'] = 'MITO'


    # Concatenate sample type annotation for MITO & KIPAN
    sample_type = pd.concat([sample_type_kipan, sample_info_mito], axis=0)
    # drop mismatched samples
    sample_type = sample_type.loc[~sample_type.index.isin(mismatched_samples)]
    sample_type.to_csv(f'{embedding_dir}/sample_type_MITO_KIPAN_mismatched_dropped.csv')

    W_loc = pd.DataFrame(W_loc, index=normalized_data_met.index)
    W_loc_mismatched_dropped = W_loc.loc[~W_loc.index.isin(mismatched_samples)]
    W_loc_mismatched_dropped.to_csv(f'{embedding_dir}/W_loc_mismatched_dropped.csv')
