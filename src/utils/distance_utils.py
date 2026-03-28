import os
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
from scipy.stats import wasserstein_distance
from scipy import linalg
from tqdm import tqdm
import matplotlib.pyplot as plt

### PCA ### 

def conduct_pca(features, test_preds, n_pc=5):
    '''
    Function to conduct PCA on the data and get results_df with first n PC's and different attributes (with standard scaling).
    args:
    features: torch.tensor where each row corresponds to an image and each col a different feature
    test_preds: dataframe with the predictions and metadata
    n_pc: number of principal components (actually for now i don't support =/= 5)
    returns:
    results_df: dataframe with the first n principal components and different attributes
    (also prints pca explained variance)
    '''
    # Convert torch tensor to numpy array
    X = features.numpy()
    
    # Standard scaling
    std_scaler = StandardScaler()
    scaled_X = std_scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=n_pc)
    pconp = pca.fit_transform(scaled_X)
    # print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    sum_explained_variance = sum(pca.explained_variance_ratio_)
    
    # Create a DataFrame for principal components
    pca_columns = [f'PC{i}' for i in range(n_pc)]
    pca_df = pd.DataFrame(pconp, columns=pca_columns)
    
    # Merge principal components with test_preds
    results_df = pd.concat([pca_df, test_preds.reset_index(drop=True)], axis=1)

    return results_df, pca.explained_variance_ratio_

def determine_pca_components(features, metadata_df, variance_explained=0.7,n_pc=100):
    results_df, explained_var_ratio = conduct_pca(features, metadata_df, n_pc=n_pc)

    var_ratio_array = np.asarray(explained_var_ratio)[:n_pc]
    cumulative_variance = np.cumsum(explained_var_ratio)
    n_components_threshold= np.argmax(cumulative_variance >= variance_explained) + 1

    return var_ratio_array, n_components_threshold

### WD ###
def calc_wd_df(experiment_dict,attributes,avg_components):

    results = {attr: {"WD_g0_vs_g1": [], "WD_g0_vs_all": [], "WD_g1_vs_all": []} for attr in attributes}

    pc_columns = ['PC' + str(n) for n in range(math.ceil(avg_components))]

    for seed in experiment_dict.keys():
        results_df = experiment_dict[seed]

        positive_results_df = results_df[results_df['target'] == 1]
        negative_results_df = results_df[results_df['target'] == 0]

        seed_results = {attr: {"WD_g0_vs_g1": [], "WD_g0_vs_all": [], "WD_g1_vs_all": []} for attr in attributes}

        for pc in pc_columns:
            for attribute in attributes:
                # Group 0 vs Group 1 (per-class, then average)
                pos_g0 = positive_results_df.loc[positive_results_df[attribute] == 0, pc]
                pos_g1 = positive_results_df.loc[positive_results_df[attribute] == 1, pc]
                neg_g0 = negative_results_df.loc[negative_results_df[attribute] == 0, pc]
                neg_g1 = negative_results_df.loc[negative_results_df[attribute] == 1, pc]

                wd_g0_vs_g1 = np.mean([
                    wasserstein_distance(pos_g0, pos_g1),
                    wasserstein_distance(neg_g0, neg_g1)
                ])

                # Group 0 vs All
                all_pos = positive_results_df[pc]
                all_neg = negative_results_df[pc]

                wd_g0_vs_all = np.mean([
                    wasserstein_distance(pos_g0, all_pos),
                    wasserstein_distance(neg_g0, all_neg)
                ])

                wd_g1_vs_all = np.mean([
                    wasserstein_distance(pos_g1, all_pos),
                    wasserstein_distance(neg_g1, all_neg)
                ])

                seed_results[attribute]["WD_g0_vs_g1"].append(wd_g0_vs_g1)
                seed_results[attribute]["WD_g0_vs_all"].append(wd_g0_vs_all)
                seed_results[attribute]["WD_g1_vs_all"].append(wd_g1_vs_all)

        # average over PCs for this seed
        for attribute in attributes:
            results[attribute]["WD_g0_vs_g1"].append(np.mean(seed_results[attribute]["WD_g0_vs_g1"]))
            results[attribute]["WD_g0_vs_all"].append(np.mean(seed_results[attribute]["WD_g0_vs_all"]))
            results[attribute]["WD_g1_vs_all"].append(np.mean(seed_results[attribute]["WD_g1_vs_all"]))

    # ---- aggregate over seeds (now std is across seeds only) ----
    rows = []
    for attr, metrics in results.items():
        row = {
            "WD_g0_vs_g1_mean": np.mean(metrics["WD_g0_vs_g1"]),
            "WD_g0_vs_g1_std": np.std(metrics["WD_g0_vs_g1"], ddof=0),

            "WD_g0_vs_all_mean": np.mean(metrics["WD_g0_vs_all"]),
            "WD_g0_vs_all_std": np.std(metrics["WD_g0_vs_all"], ddof=0),

            "WD_g1_vs_all_mean": np.mean(metrics["WD_g1_vs_all"]),
            "WD_g1_vs_all_std": np.std(metrics["WD_g1_vs_all"], ddof=0),
        }
        rows.append(row)

    wd_df = pd.DataFrame(rows, index=[a.replace("_binary", "") for a in attributes])

    return wd_df

def compute_attribute_wd(experiment_dict, attribute,pc_columns=['PC0', 'PC1', 'PC2', 'PC3']):
    """
    Compute WD for a given attribute across seeds, proportions, and PCs.
    Returns a dataframe with WD values.
    """
    data = []
    if True:
        for prop in experiment_dict[attribute].keys():
            for seed in experiment_dict[attribute][prop].keys():
                results_df = experiment_dict[attribute][prop][seed]

                positive_results_df = results_df[results_df['target'] == 1]
                neg_results_df = results_df[results_df['target'] == 0]

                for pc in pc_columns:
                    seed_results_list = [prop, seed, pc]

                    pos_dist = wasserstein_distance(
                        positive_results_df.loc[positive_results_df[attribute] == 0, pc],
                        positive_results_df.loc[positive_results_df[attribute] != 0, pc]
                    )
                    neg_dist = wasserstein_distance(
                        neg_results_df.loc[neg_results_df[attribute] == 0, pc],
                        neg_results_df.loc[neg_results_df[attribute] != 0, pc]
                    )

                    seed_results_list.extend([pos_dist, neg_dist])
                    data.append(seed_results_list)

    df_cols = ['Proportion of group 1','Seed','PC','WD pos samples','WD neg samples']
    return pd.DataFrame(data, columns=df_cols)


### FD ###

def compute_frechet_distance(X, Y):
    """Fréchet distance between two Gaussian distributions fitted on X and Y."""
    mu1, mu2 = np.mean(X, axis=0), np.mean(Y, axis=0)
    sigma1, sigma2 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Numerical issues: if imaginary part due to sqrtm, discard it
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

### MMD ###
def rbf_kernel(x, y, gamma=None):
    """Compute RBF kernel between x and y."""
    if gamma is None:
        gamma = 1.0 / x.shape[1]  # default: 1/d
    x_norm = np.sum(x ** 2, axis=1).reshape(-1, 1)
    y_norm = np.sum(y ** 2, axis=1).reshape(1, -1)
    dist = x_norm + y_norm - 2 * np.dot(x, y.T)
    return np.exp(-gamma * dist)

def compute_mmd(X, Y, gamma=None):
    """Maximum Mean Discrepancy (squared)."""
    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

### TVD ###

def compute_tvd(X, Y, bins=50):
    """
    Compute Total Variation Distance (TVD) between two sets of samples.
    X, Y: arrays of shape (n_samples, n_features)
    bins: number of bins per feature (histogram-based approximation)
    """
    # Flatten if 1D
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    tvds = []
    for d in range(X.shape[1]):  # per feature dimension
        x_hist, bin_edges = np.histogram(X[:, d], bins=bins, density=True)
        y_hist, _ = np.histogram(Y[:, d], bins=bin_edges, density=True)

        # Normalize histograms to sum to 1 (prob dist)
        x_hist = x_hist / np.sum(x_hist)
        y_hist = y_hist / np.sum(y_hist)

        tvd = 0.5 * np.sum(np.abs(x_hist - y_hist))
        tvds.append(tvd)

    return np.mean(tvds)  # average across features

# def compute_tvd(X, Y, bins=50):
# ''' this code can be used when computing TV on full feature space to avoid error where numpy can't create bints bc feature columns are near constant'''
#     X = np.asarray(X)
#     Y = np.asarray(Y)
#     if X.ndim == 1: X = X[:, None]
#     if Y.ndim == 1: Y = Y[:, None]
#     if X.shape[1] != Y.shape[1]:
#         raise ValueError(f"Feature dims differ: {X.shape[1]} vs {Y.shape[1]}")

#     tvds = []
#     for d in range(X.shape[1]):
#         x = X[:, d]
#         y = Y[:, d]

#         x = x[np.isfinite(x)]
#         y = y[np.isfinite(y)]
#         if x.size == 0 or y.size == 0:
#             continue

#         both = np.concatenate([x, y])
#         lo = both.min()
#         hi = both.max()

#         if np.allclose(lo, hi):
#             tvd_d = 0.0  # both groups share the same constant value
#         else:
#             edges = np.histogram_bin_edges(both, bins=bins)
#             x_hist, _ = np.histogram(x, bins=edges, density=False)
#             y_hist, _ = np.histogram(y, bins=edges, density=False)
#             px = x_hist / x.size
#             py = y_hist / y.size
#             tvd_d = 0.5 * np.abs(px - py).sum()

#         tvds.append(tvd_d)

#     return np.nan if len(tvds) == 0 else float(np.mean(tvds))

### GENERAL FUNCTIONS

def calc_distance_df(list_of_runs, experiment_dict, test_metadata_df,attributes,distance_metric='FD',distance_function=compute_frechet_distance,dimension_reduction=True,avg_components=10):
    results = {
    attr: {
        "g0_vs_g1": [],
        "g0_vs_all": [],
        "g1_vs_all": [],
    }
    for attr in attributes
}
    for i, seed in enumerate(experiment_dict.keys()):
        if dimension_reduction:
            results_df = experiment_dict[seed]

            pc_columns = ['PC' + str(n) for n in range(math.ceil(avg_components))]

            features = results_df[pc_columns].to_numpy()

            class_mask = results_df["Y"].values

        else: 
            run_folder = list_of_runs[i]
            features = torch.load(os.path.join(run_folder, "features.pt"))
            features = features.cpu().numpy()

            test_preds = pd.read_csv(os.path.join(run_folder, "preds.csv"))
            results_df = pd.merge(test_preds, test_metadata_df, on="id", how="left")

            class_mask = results_df["Y"].values

        all_neg = features[class_mask == 0]
        all_pos = features[class_mask == 1]

        for attr in attributes:
            group_mask = results_df[attr].values  # 0 or 1

            neg_group_0 = features[(group_mask == 0) & (class_mask == 0)]
            neg_group_1 = features[(group_mask == 1) & (class_mask == 0)]
            pos_group_0 = features[(group_mask == 0) & (class_mask == 1)]
            pos_group_1 = features[(group_mask == 1) & (class_mask == 1)]

            g0_vs_g1 = np.average([
                    distance_function(neg_group_0, neg_group_1),
                    distance_function(pos_group_0, pos_group_1)
                ],weights=[1,1]) # instead of .nanmean
            
            g0_vs_all = np.average([
                    distance_function(neg_group_0, all_neg),
                    distance_function(pos_group_0, all_pos)
                ],weights=[1,1]) # instead of .nanmean
            
            g1_vs_all = np.average([
                    distance_function(neg_group_1, all_neg),
                    distance_function(pos_group_1, all_pos)
                ],weights=[1,1]) # instead of .nanmean

            results[attr]["g0_vs_g1"].append(g0_vs_g1)
            results[attr]["g0_vs_all"].append(g0_vs_all)
            results[attr]["g1_vs_all"].append(g1_vs_all)

    rows = []
    for attr, metrics in results.items():
        row = {
            f"{distance_metric}_g0_vs_g1_mean": np.mean(metrics["g0_vs_g1"]),
            f"{distance_metric}_g0_vs_g1_std": np.std(metrics["g0_vs_g1"]),

            f"{distance_metric}_g0_vs_all_mean": np.mean(metrics["g0_vs_all"]),
            f"{distance_metric}_g0_vs_all_std": np.std(metrics["g0_vs_all"]),

            f"{distance_metric}_g1_vs_all_mean": np.mean(metrics["g1_vs_all"]),
            f"{distance_metric}_g1_vs_all_std": np.std(metrics["g1_vs_all"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows, index=results.keys())
    df.index = df.index.str.replace("_binary", "", regex=False)

    return df

def plot_attributes(attribute_distances, attributes, pretrained_tvs=None):
    fig, axes = plt.subplots(1, len(attributes), figsize=(6 * len(attributes), 5), sharey=True)

    if len(attributes) == 1:
        axes = [axes]

    for ax, attr in zip(axes, attributes):
        summary = attribute_distances[attr]
        ax.errorbar(
            summary['Proportion of group 1'],
            summary['TV_mean'],
            yerr=summary['TV_std'],
            fmt='o-', capsize=4
        )

        if pretrained_tvs and attr in pretrained_tvs:
            ax.axhline(
                y=pretrained_tvs[attr],
                color='steelblue',
                linestyle=':',
                linewidth=3,
                label='Pre-trained model latent separation'
            )
            ax.legend(fontsize=12)

        ax.set_xlabel('Proportion of group 1 in fine-tuning data', fontsize=14)
        ax.set_title(f"Separation of {attr}", fontsize=16)
        ax.grid(True)

    axes[0].set_ylabel('Mean TV', fontsize=14)
    plt.tight_layout()
    plt.show()




            

