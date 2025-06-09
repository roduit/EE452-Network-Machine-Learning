# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2025-04-28 -*-
# -*- Last revision: 2025-06-09 by roduit -*-
# -*- python version : 3.10.4 -*-
# -*- Description: Functions to load the project-*-

# Import librairies
import pandas as pd
from seiz_eeg.dataset import EEGDataset
from torch.utils.data import DataLoader, Dataset
import os
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch_geometric
import torch
import numpy as np
from scipy import signal
from scipy.signal import welch
from scipy.signal import coherence
from sklearn.model_selection import StratifiedKFold

# Import modules
import constants
from transform_func import *
from utils import is_mostly_zero_record

def parse_datasets(datasets: list, config: dict, submission: bool= False) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Parse the datasets from the configuration file and support cross-validation via fold index.

    Args:
        datasets (list): List of dataset config dicts.
        config (dict): General config dictionary.
        fold (int): Current fold index for cross-validation.

    Returns:
        tuple: DataLoaders for train, val, and test sets.
    """
    if len(datasets) == 0:
        raise ValueError("No datasets provided in the configuration file.")

    loader_train, loader_val, loader_test = None, None, None

    for i in tqdm(range(len(datasets)), desc="Loading datasets", unit="dataset"):
        dataset = datasets[i]
        set_type = dataset.get("set", None)
        if submission: 
            if set_type == "train":
                loader_train = load_data(dataset_cfg=dataset, config=config, submission=True)
            elif set_type == "test":
                loader_test = load_data(dataset_cfg=dataset, config=config, submission=True)
            else:
                continue
        
        else:
            if set_type == "train":
                loader_train = load_data(dataset_cfg=dataset, config=config)
            elif set_type == "val":
                loader_val = load_data(dataset_cfg=dataset, config=config)
            elif set_type == "test":
                continue
            else:
                raise ValueError(f"Unknown dataset type: {set_type}")
        
    return loader_train, loader_val, loader_test


def load_data(dataset_cfg: dict, config: dict, submission:bool = False) -> DataLoader:
    """Load data and split according to fold for cross-validation."""
    path = dataset_cfg.get("path", None)
    if path is None:
        raise ValueError("No data path provided in the configuration file.")

    clips = pd.read_parquet(os.path.join(path, "segments.parquet"))
    tfm_name = config.get("tfm", None)
    tfm = get_transform(tfm_name=tfm_name)

    batch_size = config.get("batch_size", constants.BATCH_SIZE)
    shuffle = dataset_cfg.get("shuffle", True)
    set_name = dataset_cfg.get("set", None)
    get_id = True if set_name == "test" else False
    sampling = dataset_cfg.get("sampling", False)
    size = int(dataset_cfg.get("size", 1))
    num_workers = config.get("num_workers", constants.NUM_WORKERS)
    n_splits = config.get("n_splits", 0)

    dataloaders = []

    if submission:
        # For submission, we use the entire dataset as training data and a dummy validation set
        n_splits = 1
    for split in range(n_splits):

        if set_name in ["train", "val"]:
            labels = clips["label"].values
            indices = np.arange(len(clips))
            
            if submission:
                train_idx = indices
                val_idx = [0]  # Dummy index for submission
            else:            
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                splits = list(skf.split(indices, labels))
                train_idx, val_idx = splits[split]

            if set_name == "train":
                clips = clips.iloc[train_idx]
            elif set_name == "val":
                clips = clips.iloc[val_idx]

            clips.sort_index(inplace=True)

        dataset = EEGDataset(
            clips,
            signals_root=path,
            signal_transform=tfm,
            prefetch=True,
            return_id=get_id
        )
        
        # Remove zero-value samples
        if tfm_name == "fusion":
            valid_indices = [i for i in range(len(dataset)) if not is_mostly_zero_record(dataset[i][0].fft)]
        else:
            valid_indices = [i for i in range(len(dataset)) if not is_mostly_zero_record(dataset[i][0])]

        dataset = torch.utils.data.Subset(dataset, valid_indices)
        sampler = None

        if sampling:
            n_classes = len(np.unique(dataset.dataset.get_label_array()))
            weights = make_weights_for_balanced_classes(dataset, n_classes)
            sampler = WeightedRandomSampler(weights, size * len(dataset), replacement=True)
            shuffle = False

        graph_cfg = config.get("graph", None)

        if graph_cfg is not None:
            if tfm_name == "fusion":
                dataset = fusion_graph_construction(dataset, graph_cfg, dataset_cfg)
            else: dataset, _ = graph_construction(dataset, graph_cfg, dataset_cfg)

            
            return torch_geometric.loader.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
        dataloaders.append(dataloader)
    if len(dataloaders) == 1:
        return dataloaders[0]
    else:
        return dataloaders

def make_weights_for_balanced_classes(samples:Dataset, nclasses:int) -> list:
    """Code taken from https://stackoverflow.com/questions/67799246/weighted-random-sampler-oversample-or-undersample
    This function is used to create weights for each class in the dataset.

    Args:
        samples (torch.utils.data.Dataset): Dataset containing the samples.
        nclasses (int): Number of classes in the dataset.

    Returns:
        weights (list): A list of weights for each sample in the dataset.
    """
    n_samples = len(samples)
    count_per_class = [0] * nclasses
    for _, sample_class in samples:
        count_per_class[sample_class] += 1
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = float(n_samples) / float(count_per_class[i])
    weights = [0] * n_samples
    for idx, (sample, sample_class) in enumerate(samples):
        weights[idx] = weight_per_class[sample_class]
    return weights

def graph_construction(dataset, graph_cfg, cfg):

    graph_type = graph_cfg.get("type", None)
    distance_path = graph_cfg.get("path", constants.DISTANCE_3D_FILE)
    distance_path = os.path.expandvars(distance_path)
    get_graph_summary = graph_cfg.get("get_graph_summary", None)
    
    data_set = cfg.get("set", None)

    if graph_type == 'distance':
        distance_df = pd.read_csv(distance_path)
        edge_threshold =  graph_cfg.get("edge_threshold", None)

        adj_matrix = distance_df.pivot(index='from', columns='to', values='distance')
        adj_matrix = adj_matrix.reindex(distance_df['from'].unique(), axis=0).reindex(distance_df['from'].unique(),
                                                                                      axis=1).to_numpy()
        with np.errstate(divide='ignore'):
            adj_matrix = 1 / adj_matrix
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = (adj_matrix - np.min(adj_matrix)) / (np.max(adj_matrix) - np.min(adj_matrix))

        adj_matrix = np.where(adj_matrix > edge_threshold, adj_matrix, 0)



        distance_graphs = []
        for i in range(len(dataset)):

            adj_tensor = torch.tensor(adj_matrix)
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_tensor)

            signals = np.asarray(dataset[i][0].T, dtype=np.float32)  # shape: (n_channels, signal_len)

            if get_graph_summary:
                x_tensor = graph_signal_summary(signals)
            else:
                x_tensor = signals

            x_tensor = torch.from_numpy(x_tensor)
            pyg_graph = torch_geometric.data.Data(x=x_tensor, edge_index=edge_index) # edge_weight=edge_weight

            if data_set != "test":
                pyg_graph.y = torch.tensor(dataset[i][1], dtype=torch.int64)
            else:
                pyg_graph.id = dataset[i][1]

            distance_graphs.append(pyg_graph)

        return distance_graphs, adj_matrix


    elif graph_type == 'correlation':
        edge_threshold =  graph_cfg.get("edge_threshold", None)
        correlation_graphs = []
        correlation_adj_matrix = []

        for i in range(len(dataset)):
            data = np.asarray(dataset[i][0].T, dtype=np.float32) 
            adj_matrix = safe_corrcoef(data)
            adj_matrix = np.where(adj_matrix > edge_threshold, adj_matrix, 0)
            np.fill_diagonal(adj_matrix, 0)

            if np.min(adj_matrix) == np.max(adj_matrix):
                adj_matrix = np.zeros_like(adj_matrix)
            else:
                adj_matrix = (adj_matrix - np.min(adj_matrix)) / (np.max(adj_matrix) - np.min(adj_matrix))




            adj_tensor = torch.tensor(adj_matrix)
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_tensor)

            signals = np.asarray(dataset[i][0].T, dtype=np.float32)  # shape: (n_channels, signal_len)

            if get_graph_summary:
                x_tensor = graph_signal_summary(signals)
            else:
                x_tensor = signals

            x_tensor = torch.from_numpy(x_tensor)
            pyg_graph = torch_geometric.data.Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_weight) # edge_weight=edge_weight

            if data_set != "test":
                pyg_graph.y = torch.tensor(dataset[i][1], dtype=torch.int64)
            else:
                pyg_graph.id = dataset[i][1]

            correlation_graphs.append(pyg_graph)
            correlation_adj_matrix.append(adj_matrix)

        return correlation_graphs, correlation_adj_matrix


    elif graph_type == 'coherence':
        edge_threshold =  graph_cfg.get("edge_threshold", None)
        coherence_energie_graphs = []
        coherence_adj_matrix = []


        for i in range(len(dataset)):

            f, coherence_matrix = signal.coherence(x=dataset[i][0][:, :, np.newaxis], y=dataset[i][0][:, np.newaxis, :], fs=250, axis=0)
            coherence_matrix = np.nan_to_num(coherence_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            adj_matrix = coherence_matrix[np.logical_and(f >= 0.5, f <= 30), : , :].mean(axis=0)
            adj_matrix = np.where(adj_matrix >= edge_threshold, adj_matrix, 0)
            np.fill_diagonal(adj_matrix, 0)

            if np.min(adj_matrix) == np.max(adj_matrix):
                adj_matrix = np.zeros_like(adj_matrix)
            else:
                adj_matrix = (adj_matrix - np.min(adj_matrix)) / (np.max(adj_matrix) - np.min(adj_matrix))



            adj_tensor = torch.tensor(adj_matrix)
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_tensor)

            signals = np.asarray(dataset[i][0].T, dtype=np.float32)  # shape: (n_channels, signal_len)

            if get_graph_summary:
                x_tensor = graph_signal_summary(signals)
            else:
                x_tensor = signals # np.logical_and(freqs >= 0.5, freqs <= 30)

            x_tensor = torch.from_numpy(x_tensor)
            pyg_graph = torch_geometric.data.Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_weight) # edge_weight=edge_weight

            if data_set != "test":
                pyg_graph.y = torch.tensor(dataset[i][1], dtype=torch.int64)
            else:
                pyg_graph.id = dataset[i][1]

            coherence_energie_graphs.append(pyg_graph)
            coherence_adj_matrix.append(adj_matrix)

        return coherence_energie_graphs, coherence_adj_matrix

    else:
        raise ValueError("This graph construction method is not implemented.")


def fusion_graph_construction(dataset, graph_cfg, cfg):
    
    graph_type = graph_cfg.get("type", None)
    
    data_set = cfg.get("set", None)

    if graph_type == 'correlation':
        edge_threshold =  graph_cfg.get("edge_threshold", None)
        corelation_base = graph_cfg.get("corelation_base", None)
        correlation_graphs = []

        for i in range(len(dataset)):
            if corelation_base == "fft":
                data = np.asarray(dataset[i][0].fft.T, dtype=np.float32) 
            elif corelation_base == "time":
                data = np.asarray(dataset[i][0].time.T, dtype=np.float32) 
            else:
                raise ValueError("corelation_base must be either 'fft' or 'time'.")
            adj_matrix = safe_corrcoef(data)
            adj_matrix = np.where(adj_matrix > edge_threshold, adj_matrix, 0)
            np.fill_diagonal(adj_matrix, 0)


            adj_tensor = torch.tensor(adj_matrix)
            edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_tensor)

            x_tensor = torch.from_numpy(np.asarray(dataset[i][0].fft.T, dtype=np.float32))  # shape: (n_channels, signal_len)
            time_tensor = torch.from_numpy(np.asarray(dataset[i][0].time.T, dtype=np.float32))  # shape: (n_channels, signal_len)
            pyg_graph = torch_geometric.data.Data(x=x_tensor,time= time_tensor, edge_index=edge_index, edge_attr=edge_weight) # edge_weight=edge_weight
            if data_set != "test":
                pyg_graph.y = torch.tensor(dataset[i][1], dtype=torch.int64)
            else:
                pyg_graph.id = dataset[i][1]

            correlation_graphs.append(pyg_graph)
            
        return correlation_graphs
    else: raise ValueError("Only correlation supported for fusion.")



def safe_corrcoef(X, eps=1e-8):
    X = X - np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std < eps] = eps  # Avoid division by zero
    X_norm = X / std
    return np.dot(X_norm, X_norm.T) / X.shape[1]

def get_transform(tfm_name: str) -> callable:
    """Get the transform function based on the name provided.

    Args:
        tfm_name (str): Name of the transform function to be used.

    Returns:
        function: The transform function corresponding to the name provided.
    """
    if tfm_name == "fft":
        return fft_filtering
    elif tfm_name == "time":
        return time_filtering
    elif tfm_name == "clean":
        return clean_input
    elif tfm_name == "psd":
        return power_spectral_density
    elif tfm_name == "wavelet":
        return wavelet_transform_filtering
    elif tfm_name == "fusion":
        return fusion_transform
    else:
        return None

def graph_signal_summary(signals):
    summary_array = np.stack([
        np.mean(signals, axis=1),
        np.median(signals, axis=1),
        np.max(signals, axis=1),
        np.min(signals, axis=1),
        np.std(signals, axis=1),
    ], axis=1)

    return summary_array


def compute_band_energy(eeg_data, fs = 250):
    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
    }

    n_channels = eeg_data.shape[0]
    energy_matrix = np.zeros((n_channels, 4))

    for ch in range(n_channels):
        # Welch returns PSD estimate
        freqs, psd = signal.welch(eeg_data[ch], fs=fs, nperseg=fs)

        for i, band in enumerate(freq_bands):
            idx_band = np.logical_and(freqs >= freq_bands[band][0], freqs <= freq_bands[band][1])
            # Integrate PSD over band: power ≈ energy
            band_power = np.trapz(psd[idx_band], freqs[idx_band])
            energy_matrix[ch, i] = band_power
            
    return energy_matrix




