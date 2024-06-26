import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import numpy as np


def order_tumor_normal(data, MET_RNA_map, batch_map, sample_map_met, batch_index_vector, sub_dir, metabolite_map):
    ordered_data = np.copy(data)
    n_batches = batch_index_vector.max() + 1
    batch_map_inverse = {v: k for (k, v) in batch_map.items()}
    sample_order = []
    sample_type = pd.DataFrame({'TN': []})
    for bidx in range(n_batches):
        batch_rows = np.arange(ordered_data.shape[0])[batch_index_vector == bidx]
        batch = ordered_data[batch_rows]
        # Sort the master_map by tumor/normal in 'TN' column
        master_map = MET_RNA_map[MET_RNA_map['sub_dir'] == sub_dir[bidx]].sort_values(by=['TN'])
        sample_type = pd.concat([sample_type, pd.DataFrame({'TN': master_map['TN']})], axis=0)
        # Get the order of the sorted values (pay attention that indices in the batch is different from the index in the merged data)
        batch_order = [sample_map_met[i] - batch_rows[0] for i in master_map.index]
        order = [sample_map_met[i] for i in master_map.index]
        sample_order.extend(order)
        batch = batch[batch_order, :]
        ordered_data[batch_rows, :] = batch
        # Differentiate missing/present/censored values
        missing = np.all(np.isnan(batch), axis=0)
        # ordered_data.shape[1]
        for col in range(len(metabolite_map)):
            if missing[col]:
                ordered_data[batch_rows, col] = 0  # missing
                continue
            missing_in_col = np.isnan(batch[:, col])
            rows = batch_rows[~missing_in_col]
            ordered_data[rows, col] = 2  # present
            ordered_data[batch_rows[missing_in_col], col] = 1  # censored
        for col in range(len(metabolite_map), ordered_data.shape[1]):
            if missing[col]:
                ordered_data[batch_rows, col] = 0  # missing
                continue
            missing_in_col = np.isnan(batch[:, col])
            rows = batch_rows[~missing_in_col]
            ordered_data[rows, col] = 3  # present
            ordered_data[batch_rows[missing_in_col], col] = 1  # censored
    # reorder the metabolites for heatmap visualization (put the complete columns at left)
    ordered_data = np.insert(ordered_data, len(sample_map_met), ordered_data.sum(axis=0), axis=0)
    ordered_met = ordered_data[:, 0:len(metabolite_map)]
    ordered_rna = ordered_data[:, len(metabolite_map):]
    ordered_met = ordered_met[:, ordered_met[len(sample_map_met), :].argsort()[::-1]]
    ordered_rna = ordered_rna[:, ordered_rna[len(sample_map_met), :].argsort()[::-1]]
    ordered_data = np.concatenate((ordered_met, ordered_rna), axis=1)
    ordered_data = ordered_data[0:len(sample_map_met), :]
    return ordered_data, sample_order, sample_type

def uniquify(df_columns):
    seen = set()
    for item in df_columns:
        fudge = 1
        newitem = item
        while newitem in seen:
            fudge += 1
            newitem = "{}_{}".format(item, fudge)
        yield newitem
        seen.add(newitem)

if __name__ == "__main__":
    # Check consistency of the results
    #---------------------------------------------------------------------------------------------------------
    file_path = "/juno/work/reznik/xiea1/MetabolicModel"
    cancer_type = 'ccRCC'
    if cancer_type == 'ccRCC':
        sub_dir = ['CPTAC','CPTAC_val', 'RC18', 'RC20']  # notice that CPTAC is the first one
    elif cancer_type == 'BRCA':
        sub_dir = ['BrCa1', 'BrCa2', 'TNBC']

    median_rho_feature_list = []
    for i in range(len(sub_dir)):
        median_rho_feature_list.append(pd.read_csv(
            f'{file_path}/results_RNA_{cancer_type}/{sub_dir[i]}/median_rho_feature.csv', header=0, index_col='feature').iloc[:, 1:])

    df = pd.concat(median_rho_feature_list, axis=1)
    df.columns = list(uniquify(df.columns))  # rename columns because they are the same coming from each dataframe
    df.rename(columns={'median_rho': 'CPTAC', 'median_rho_2': 'CPTAC_val',
                        'median_rho_3': 'RC18',
                       'median_rho_4': 'RC20'}, inplace=True)
    df_rho = df[['CPTAC', 'CPTAC_val', 'RC18', 'RC20']]

    # 1. Pairwise pearson's correlation of median rho values in each dataset
    corr_matrix = df_rho.corr(method='pearson')
    # Heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap="YlOrBr",  annot=True)
    #plt.show()
    plt.title('Pearson correlation of median rho values in each dataset')
    plt.savefig(f'{file_path}/results_RNA_{cancer_type}/plots/heatmap_pearson_rho_June_8.pdf')
    plt.close()

    # IOU scores
    def iou(a, b):
        intersection = len(set(a.index) & set(b.index))
        union = len(set(a.index) | set(b.index))
        return intersection / union if union > 0 else 0


    # 2. calculate the IoU scores between all pairs of variables
    cutoff = 0.5
    scores = pd.DataFrame(index=df_rho.columns, columns=df_rho.columns)
    for col1 in df_rho.columns:
        for col2 in df_rho.columns:
            scores.at[col1, col2] = iou(df_rho[col1][(df_rho[col1] > cutoff)][df_rho[col2].notna()],
                                        df_rho[col2][df_rho[col2] > cutoff][df_rho[col1].notna()])
    scores = scores.astype(float)
    mask = np.tril(np.ones_like(scores, dtype=bool))
    sns.heatmap(scores, mask=mask, cmap="YlOrBr",  annot=True)
    #plt.show()
    plt.title('IoU scores between all pairs of datasets')
    plt.savefig(f'{file_path}/results_RNA_{cancer_type}/plots/heatmap_iou_score_rho>0.5.pdf')
    plt.close()

    # 3. Pairplots
    # standard blue pairplots
    #sns.pairplot(df_rho, vars=['CPTAC', 'CPTAC_val', 'RC18', 'RC20'], dropna=True, corner=True)
    #plt.show()

    # customized pairplots (each scatterplots has a different color)
    # color code: https://xkcd.com/color/rgb/, https://matplotlib.org/stable/gallery/color/named_colors.html
    colors = iter(['xkcd:gray', 'xkcd:gray', 'xkcd:gray','xkcd:gray',
                   'xkcd:sky blue','xkcd:light peach', 'xkcd:greyish pink',
                   'xkcd:pastel pink',
                   'xkcd:pale teal', 'xkcd:muddy green'])
    def my_scatter(x, y, **kwargs):
        kwargs['color'] = next(colors)
        plt.scatter(x, y, **kwargs, s=8)
        # Calculate and plot the regression line
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m * x + b, **kwargs)

    def my_hist(x, **kwargs):
        kwargs['color'] = next(colors)
        plt.hist(x, **kwargs, edgecolor="black", bins=8)

    g = sns.PairGrid(df_rho, vars=['CPTAC', 'CPTAC_val', 'RC18', 'RC20'], dropna=True, corner=True)
    g.map_diag(my_hist)
    g.map_offdiag(my_scatter)
    plt.savefig(f'{file_path}/results_RNA_{cancer_type}/plots_logs/pairplot_4_June_8_reg.pdf')
    plt.close()

    # Scatterplot
    plt.rcParams['figure.figsize'] = [10, 10]
    df_test = df_rho[df_rho['RC18'].notna()][df_rho['RC20'].notna()]
    x = df_test['RC18']
    y = df_test['RC20']
    plt.scatter(x, y)
    plt.axline((0, 0), slope=1)
    plt.xlabel('median_rho in rc18')
    plt.ylabel('median_rho in rc20')
    title = 'median_rho rc18 vs rc20'
    plt.title(title)
    #plt.savefig(f'results_RNA_{cancer_type}/plots/scatterplot_{title}.pdf')
    plt.show()
    #plt.close()
    rho, pval = pearsonr(x, y)
    print(f"pearson's rho = {rho}, pval = {pval}")

    # 4. Comparing the scatter plots between median_rho & standard deviation in each dataset
    df_rho = df[['CPTAC', 'ave_ave_se', 'CPTAC_val', 'ave_ave_se_2', 'RC18', 'ave_ave_se_3', 'RC20', 'ave_ave_se_4']]
    df_rho = df_rho[df_rho['CPTAC'].notna()][df_rho['CPTAC_val'].notna()][df_rho['RC18'].notna()][
        df_rho['RC20'].notna()]
    df_rho['sum'] = df_rho[['CPTAC', 'CPTAC_val', 'RC18', 'RC20']].sum(axis=1)

    # Histograms
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.hist(df_rho['ave_ave_se_4'])
    plt.xlabel('average standard deviation of posterior predicted metabolites')
    plt.ylabel('count')
    plt.title('RC20 histogram of standard deviation')
    #plt.show()
    plt.savefig(f'{file_path}/results_RNA_{cancer_type}/plots/sd_histogram_RC20.pdf')
    plt.close()

    # Scatter plots
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.scatter(df_rho['ave_ave_se_3'], df_rho['RC18'],  c='orange', label='RC18')
    plt.scatter(df_rho['ave_ave_se_4'], df_rho['RC20'],  c='green', label='RC20')
    plt.scatter(df_rho['ave_ave_se_2'], df_rho['CPTAC_val'], c='red', label='CPTAC_val')
    plt.scatter(df_rho['ave_ave_se'], df_rho['CPTAC'], c='blue', label='CPTAC')
    plt.legend()
    plt.xlabel('Average standard deviation')
    plt.ylabel("Spearman's rho")
    #plt.show()
    plt.savefig(f'{file_path}/results_RNA_{cancer_type}/plots/median_rho_sd_4.pdf')
    plt.close()

    # 5. Get a list of reproducibly well-predicted metabolites (rho > 0 & FDR < 0.1 in at least 3 datasets)
    df_rho = df[['CPTAC', 'sig_in_most', 'CPTAC_val', 'sig_in_most_2', 'RC18', 'sig_in_most_3', 'RC20', 'sig_in_most_4']]
    df_rho['average_rho'] = df_rho[['CPTAC', 'CPTAC_val', 'RC18', 'RC20']].mean(axis=1, skipna=True)
    for i in range(4):
        df_rho[f'well_predicted_{i}']=(df_rho[df_rho.columns[2*i]] >= 0).astype(bool) & df_rho[df_rho.columns[2*i+1]]
    df_rho['n_well_predicted'] = df_rho.iloc[:, 9:13].sum(axis=1)
    reproducibly_well_predicted = df_rho[df_rho['n_well_predicted'] >= 3]
    reproducibly_well_predicted.to_csv(f'{file_path}/results_RNA_{cancer_type}/reproducibly_well_predicted_metabolites_June_8.csv')

    # 6. Get a list of rho values and significance values in all 4 datasets and highlight if a metabolite is reproducibly well-predicted
    df_rho['reproducibly_well_predicted'] = df_rho['n_well_predicted'] >= 3
    df_rho.to_csv(f'{file_path}/results_RNA_{cancer_type}/rho_sig_in_all_datasets_June_8.csv')





    #-------------------------------------Schematic of data overview-------------------------------------------
    MET_RNA_map = pd.read_csv(f'{file_path}/data/MasterMapping_updated.csv', header=0, index_col='MetabID')
    data = pd.read_csv("/juno/work/reznik/xiea1/MetabolicModel/results_RNA_ccRCC/CPTAC/embeddings/normalized_met_rna_data_pyro.csv", header=0, index_col=0)
    ordered_data, sample_order, sample_type = order_tumor_normal(data, MET_RNA_map, batch_map, sample_map_met, batch_index_vector, sub_dir, metabolite_map)
    pd.DataFrame(ordered_data).to_csv(f'/juno/work/reznik/xiea1/MetabolicModel/results_RNA_ccRCC/plots/ordered_raw_data.csv')
    sample_type.to_csv(f'/juno/work/reznik/xiea1/MetabolicModel/results_RNA_ccRCC/plots/ordered_sample_type(TN).csv')
    ordered_data = np.delete(ordered_data, np.arange(start_row[2], stop_row[2]), axis=0)
    sample_type.drop(sample_type.index[np.arange(start_row[2], stop_row[2])], inplace=True)

    #-------------------------------------Barplot comparing performance-------------------------------------------
    df = pd.DataFrame({
        'dataset': ['CPTAC', 'CPTAC_val', 'RC18', 'RC20'],
        #'methods': ['Bayesian', 'Bayesian','T-MIRTH','Bayesian','T-MIRTH', 'Bayesian'],
        'significantly predicted metabolites (%)': np.array([83/156, 84/129, 504/709, 478/718])*100
    })
    # create a grouped bar plot using Seaborn
    def colors_from_values(values, palette_name):
        # normalize the values to range [0, 1]
        normalized = (values - min(values)) / (max(values) - min(values))
        # convert to indices
        indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
        # use the indices to get the colors
        palette = sns.color_palette(palette_name, len(values))
        return np.array(palette).take(indices, axis=0)
    sns.set_style("dark")
    sns.barplot(x='dataset', y='significantly predicted metabolites (%)', data=df,
                palette=colors_from_values(df['significantly predicted metabolites (%)'], "YlOrRd"))
    plt.savefig('/juno/work/reznik/xiea1/MetabolicModel/results_RNA_ccRCC/plots/percentage_barplot_4.pdf')

    test = pd.read_csv('/juno/work/reznik/xiea1/MIRTH/results_MET_RNA/RC20_ave_rho_padj.csv', header=0, index_col=0)
    test[(test['p_adj'] < 0.1) & (test['rho']>0)].shape[0]
