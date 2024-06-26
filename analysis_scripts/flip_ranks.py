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
    file_path = "/data1/reznike/xiea1/MetabolicModel"
    results_path = f"{file_path}/results_RNA_MITO"
    dir = "MITO1"
    results_dir = f"{results_path}/{dir}"

    actual_pred_res_df = pd.read_csv(f"{results_dir}/actual_vs_predicted_ranks.csv",
                                     index_col=0, header=0)
    # get flipped actual_rank and flipped predicted_rerank (flipped and normalized to [0,1))
    feature_list = actual_pred_res_df['feature'].unique().tolist()
    actual_pred_res_df_flipped = pd.DataFrame(columns=actual_pred_res_df.columns)
    actual_pred_res_df_flipped['flipped_actual_rank'] = np.nan
    actual_pred_res_df_flipped['flipped_predicted_rerank'] = np.nan

    for feature in feature_list:
        feature_df = actual_pred_res_df.loc[actual_pred_res_df['feature'] == feature]
        feature_df['flipped_actual_rank'] = (feature_df['actual_rank'].max() - feature_df['actual_rank'])/feature_df.shape[0]
        feature_df['flipped_predicted_rerank'] = (feature_df['predicted_rerank'].max() - feature_df['predicted_rerank'])/feature_df.shape[0]
        actual_pred_res_df_flipped = pd.concat([actual_pred_res_df_flipped,feature_df], axis=0)

    actual_pred_res_df_flipped.to_csv(f"{results_dir}/actual_vs_predicted_ranks_flipped.csv")

