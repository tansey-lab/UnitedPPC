import numpy as np
from UnitedMet.impute_met.utils import count_obs, order_and_rank
import os
import pandas as pd
from itertools import repeat

def file_system_init(rna_matched_data_dir, met_matched_data_dir, rna_imputation_data_dir, results_dir):
    # Input Metabolomics and Transcriptomics data directory
    if not os.path.exists(rna_matched_data_dir) or not os.path.exists(met_matched_data_dir):
        raise OSError(f"Input Metabolomics and Transcriptomics reference data directory don't exist.")
    if not os.path.exists(rna_imputation_data_dir):
        raise OSError(f"Input RNA data directory for imputation doesn't exist.")

    # Define sub-directories inside the results directory
    plots_dir = f'{results_dir}/plots'
    embedding_dir = f'{results_dir}/embeddings'

    for dir in [embedding_dir, plots_dir]:
        try:
            os.makedirs(dir)
        except FileExistsError:
            print(f"Directory '{dir}' already exists.")
    return (plots_dir,  embedding_dir)

def tic_normalization_across(data, batch_index_vector):
    normalized_data = np.copy(data)  # Note that, when we modify data, normalized_data changes
    n_batches = batch_index_vector.max() + 1
    for bidx in range(n_batches):  # range(i) include numbers from 0 to i-1
        batch_rows = np.arange(data.shape[0])[
            batch_index_vector == bidx]  # np.arange(n) is same as np.array(range(3))
        batch = data[batch_rows]  # get the separate dataframe for distinct batches
        nan_mask = np.isnan(batch)  # Test element-wise for NaN and return result as a boolean array
        missing = np.all(nan_mask, axis=0)  # boolean array about whether for a whole column is all nan in each row
        min_batch = np.nanmin(batch)  # Return minimum of an array, ignoring any NaNs
        for row in range(batch.shape[0]):
            n_censored = np.sum(nan_mask[row]) - np.sum(missing)  # remember to substrate missing numbers
            row_tic = np.nansum(batch[row]) + 0.5 * n_censored * min_batch
            batch[row, :] = batch[row, :] / row_tic
        batch = batch / np.nansum(batch, axis=1, keepdims=True)  # make sure the sum of each row is 1
        normalized_data[batch_rows] = batch
    return normalized_data  # missing values are still NAN but not filled with np.nanmin(batch) now

def data_normalization_cleaning(met_data, rna_data, met_batch_index_vector, metabolite_map,
                                gene_map, sample_map_rna, batch_index_vector):
    # TIC Normalize the MET data
    met_data = tic_normalization_across(met_data, met_batch_index_vector)

    missing_data = np.full((len(batch_index_vector[batch_index_vector == 0]), len(metabolite_map)), np.nan)
    met_data = np.concatenate([missing_data, met_data], axis=0)

    ########################################################## Concatenate MET and RNA data
    data = np.concatenate([met_data, rna_data], axis=1)
    met_names = list(metabolite_map.keys())
    rna_names = list(gene_map.keys())
    if np.all(np.isnan(data), axis=0).sum() == 0:
        print("The aggregate data is clean.")
    else:
        print("Some features are NaN in all samples and need to be removed from the aggregate data.\
              If remove_solo == True, solo metabolites will also be removed in this step.")
        delete_index = np.where(np.all(np.isnan(data), axis=0))[0]
        met_delete_index = delete_index[delete_index < len(metabolite_map)]
        rna_delete_index = delete_index[delete_index >= len(metabolite_map)] - len(metabolite_map)
        # Update the data
        data = np.delete(data, delete_index, axis=1)
        # Update the met_data and rna_data
        met_data = np.delete(met_data, met_delete_index, axis=1)
        rna_data = np.delete(rna_data, rna_delete_index, axis=1)
        # Update the feature names and maps
        met_names = np.delete(np.array(met_names), met_delete_index).tolist()
        rna_names = np.delete(np.array(rna_names), rna_delete_index).tolist()
        print(" Features which are all NAs were removed.")

    metabolite_map = {v: k for k, v in enumerate(met_names)}
    gene_map = {v: k for k, v in enumerate(rna_names)}
    feature_names = met_names + rna_names
    sample_names = list(sample_map_rna.keys())
    return data, met_data, rna_data, metabolite_map, gene_map, met_names, rna_names, feature_names, sample_names


def load_met_data(met_matched_data_dir):
    # First create an empty dataframe (sample*feature) with the shape of merged datasets, establish maps for features/samples/batches
    features = set()  # because we need the characteristic of non-repeating elements in set() to make 'features' non-redundant
    nrows = 0
    samples = []
    dirlist = os.listdir(met_matched_data_dir)
    dirlist.sort()
    for batch_idx, fpath in enumerate(
            dirlist):
        if 'csv' not in fpath:
            continue
        df = pd.read_csv(f'{met_matched_data_dir}/{fpath}', header=0, index_col=0)
        features.update(df.columns)
        samples.extend(df.index)
        nrows += df.shape[0]
    feature_map = {s: i for i, s in enumerate(features)}
    sample_map = {s: i for i, s in enumerate(samples)}
    data = np.full((nrows, len(features)),
                   np.nan)
    batch_index_vector = np.zeros(nrows, dtype=int)

    # Secondly, fill in the data from each dataset to the empty dataframe
    sidx = 0
    batch_idx = 0
    for fpath in dirlist:
        if 'csv' not in fpath:
            continue
        df = pd.read_csv(f'{met_matched_data_dir}/{fpath}', header=0, index_col=0)
        for feature in df.columns:  # fill in the data by feature columns
            fidx = feature_map[feature]
            data[sidx:sidx + df.shape[0], fidx] = df[
                feature].values
        batch_index_vector[sidx:sidx + df.shape[
            0]] = batch_idx
        sidx += df.shape[0]
        batch_idx += 1
    return data, feature_map, sample_map, batch_index_vector

def load_rna_data(rna_matched_data_dir, rna_imputation_data_dir):
    """
    We load the rna data of the imputation set first and then load the other matched rna data.
    Pay attention that the imputation dataset will always be put in the first batch.
    """
    # First create an empty dataframe(sample*feature) with the shape of merged datasets, establish maps for features/samples/batches
    features = set()  # because we need the characteristic of non-repeating elements in set() to make 'features' non-redundant
    nrows = 0
    samples = []
    batch_names = []
    batch_sizes = []
    dirlist = os.listdir(rna_matched_data_dir)
    dirlist.sort()
    imputation_dirlist = os.listdir(rna_imputation_data_dir)
    imputation_dirlist.sort()
    dirlist = imputation_dirlist + dirlist  # put the imputation dataset in the first batch
    for batch_idx, fpath in enumerate(
            dirlist):
        if 'csv' not in fpath:
            continue
        elif batch_idx == 0:
            df = pd.read_csv(f'{rna_imputation_data_dir}/{fpath}', header=0, index_col=0)
        else:
            df = pd.read_csv(f'{rna_matched_data_dir}/{fpath}', header=0, index_col=0)
        # remove the genes that are all zeros or all nans
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.loc[:, (df.notnull()).any(axis=0)]
        if batch_idx == 0:
            features.update(df.columns)
        else:
            features = features.intersection(df.columns)  # only keep the intersection of features
        samples.extend(df.index)
        batch_names.append(
            fpath)
        batch_sizes.append(df.shape[0])
        nrows += df.shape[0]
    feature_map = {s: i for i, s in enumerate(features)}
    sample_map = {s: i for i, s in enumerate(samples)}
    batch_map = {s: i for i, s in enumerate(batch_names)}
    data = np.full((nrows, len(features)),
                   np.nan)
    batch_index_vector = np.zeros(nrows, dtype=int)

    # Secondly, fill in the data from each dataset to the empty dataframe
    sidx = 0
    batch_idx = 0
    start_row = []
    stop_row = []
    for fpath in dirlist:
        if 'csv' not in fpath:
            continue
        elif batch_idx == 0:
            df = pd.read_csv(f'{rna_imputation_data_dir}/{fpath}', header=0, index_col=0)
        else:
            df = pd.read_csv(f'{rna_matched_data_dir}/{fpath}', header=0, index_col=0)
        for feature in features:  # fill in the data by intersected features
            fidx = feature_map[feature]
            data[sidx:sidx + df.shape[0], fidx] = df[
                feature].values
        batch_index_vector[sidx:sidx + df.shape[
            0]] = batch_idx
        start_row.append(sidx)
        stop_row.append(sidx + df.shape[0])
        sidx += df.shape[0]
        batch_idx += 1
    start_row = np.array(start_row)
    stop_row = np.array(stop_row)
    n_batch = len(batch_map)
    batch_settings = pd.DataFrame(list(zip(batch_names, range(n_batch))), columns=["batches", "order"])

    print("The order of datasets in the aggregate matrix is as below:")
    print(batch_settings)

    return data, batch_index_vector, feature_map, sample_map, batch_map, batch_sizes, start_row, stop_row, n_batch

def check_match(sample_map_met, sample_map_rna, MET_RNA_map, batch_index_vector):
    sample_met = list(sample_map_met.keys())
    sample_rna = list(sample_map_rna.keys())[len(batch_index_vector[batch_index_vector == 0]):]
    met_order = []
    rna_order = []
    met_order.extend(MET_RNA_map.index.tolist())
    rna_order.extend(MET_RNA_map['RNAID'].tolist())
    if sample_met == met_order:
        print('MetaID matched')
    else:
        print('MetaID is not matched')
    if sample_rna == rna_order:
        print('RNAID matched')
    else:
        print('RNAID is not matched')

