import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    # -------------------------------------Barplot comparing performance-------------------------------------------
    df = pd.DataFrame({
        'dataset': ['RCC_G6', 'NSCLC_G6', 'NSCLC_Q6'],
        # 'methods': ['Bayesian', 'Bayesian','T-MIRTH','Bayesian','T-MIRTH', 'Bayesian'],
        'significantly predicted metabolites (%)': np.array([12 / 23, 5 / 9, 5 / 8]) * 100
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
    plt.savefig('/juno/work/reznik/xiea1/MetabolicModel/results_RNA_isotope/plots_logs/percentage_barplot.pdf')
