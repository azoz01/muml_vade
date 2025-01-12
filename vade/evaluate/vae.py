import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    rand_score,
    silhouette_score,
)
from torch.utils.data import DataLoader

from vade.device import DEVICE
from vade.model.vae import VAE


def evaluate_vae(
    model: VAE,
    dataloader: DataLoader,
    output_root_path: Path,
    dataset: Literal["HAR", "MNIST"],
) -> None:
    assert isinstance(model, VAE), "This result supports only VAE"
    model = model.to(DEVICE)
    # Extract predictions
    predictions = []
    latent_representations = []
    true_labels = []
    data = []
    for X, y in dataloader:
        X = X.to(DEVICE)
        data.append(X)
        with torch.no_grad():
            predictions.append(model(X))
            latent_representations.append(model.mu_encoder(model.encode(X)))
        true_labels.append(y)
    predictions = torch.concat(predictions).cpu().numpy()
    latent_representations = torch.concat(latent_representations).cpu().numpy()
    true_labels = torch.concat(true_labels).cpu().numpy()
    X = torch.concat(data)
    n_clusters = np.unique(true_labels).shape[0]

    # Perform clustering
    clustering = KMeans(n_clusters=n_clusters, random_state=123)
    clusters = clustering.fit_predict(latent_representations)
    clustering_metrics = dict()

    # Clustering vs true classess
    clustering_metrics["rand_score"] = rand_score(true_labels, clusters)

    # Clustering
    clustering_metrics["silhouette"] = silhouette_score(
        latent_representations, clusters
    ).item()
    clustering_metrics["CH"] = calinski_harabasz_score(
        latent_representations, clusters
    ).item()

    with open(output_root_path / "clustering_metric.json", "w") as f:
        json.dump(clustering_metrics, f)

    # Cluster visualization
    stratified_sample_X = []
    stratified_sample_true_labels = []
    for i in range(n_clusters):
        stratified_sample_X.append(
            latent_representations[true_labels == i][:100]
        )
        stratified_sample_true_labels.append(
            true_labels[true_labels == i][:100]
        )
    stratified_sample_X = np.concat(stratified_sample_X)
    stratified_sample_true_labels = np.concat(stratified_sample_true_labels)
    reduced_data = TSNE(2).fit_transform(stratified_sample_X)
    sns.scatterplot(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        hue=stratified_sample_true_labels.tolist(),
        palette="tab10",
    )
    plt.savefig(output_root_path / "clusters.png")

    # Observation from each cluster
    if dataset == "MNIST":
        fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(16, 16))
        for a in ax.flat:
            a.tick_params(
                left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False,
            )
        plt.subplots_adjust(wspace=0, hspace=0)
        for idx in range(25):
            output = (
                torch.randn(size=(model.layers_sizes[-1],)).cuda().double()
            )
            output = model.reparametrize(
                model.mu_encoder(output), model.logvar_encoder(output)
            )
            output = model.decode(output)
            output_img = output.reshape(28, 28).cpu().detach().numpy()
            ax_to_plot = ax[idx // 5][idx % 5]
            ax_to_plot.imshow(output_img, cmap="gray")
        fig.tight_layout()
        fig.savefig(output_root_path / "generation.png")
