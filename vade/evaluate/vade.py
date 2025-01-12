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
from vade.model.vade import VADE


def evaluate_vade(
    model: VADE,
    dataloader: DataLoader,
    output_root_path: Path,
    dataset: Literal["HAR", "MNIST"],
) -> None:
    assert isinstance(model, VADE), "This result supports only VADE"
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
    classes = model.classify(X)
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
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        hue=stratified_sample_true_labels.tolist(),
        palette="tab10",
    )
    sns.scatterplot(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        color="black",
        s=100,
        label="VADE",
    )
    sns.scatterplot(
        x=reduced_data[:, 0],
        y=reduced_data[:, 1],
        color="red",
        s=100,
        label="KMeans",
    )
    plt.savefig(output_root_path / "clusters.png")
    (output_root_path / "generation").mkdir()

    # Observation from each cluster
    if dataset == "MNIST":
        for class_ in range(10):
            fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(9, 9))
            for a in ax.flat:
                a.tick_params(
                    left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False,
                )
            plt.subplots_adjust(wspace=0, hspace=0)
            for idx in range(12):
                r = torch.randn(size=(model.layers_sizes[-1],)).cuda()
                z = r * torch.exp(model.logvar[class_]) + model.mu[class_]
                output = model.decode(z)
                output_img = output.reshape(28, 28).cpu().detach().numpy()
                ax_to_plot = ax[idx // 4][idx % 4]
                ax_to_plot.imshow(output_img, cmap="gray")
            fig.tight_layout()
            fig.savefig(output_root_path / "generation" / f"{class_=}")

    # Clusters avg distance
    within_distances = []
    between_distances = []
    for class_ in range(model.n_classes):
        r = torch.randn(size=(model.layers_sizes[-1],)).cuda()
        with torch.no_grad():
            z = (r * torch.exp(model.logvar[class_])) + model.mu[class_]
            output = model.decode(z)

        dist_within = torch.cdist(
            output.unsqueeze(0), X[classes == class_].double()
        ).mean()
        if dist_within > 0:
            within_distances.append(dist_within)
        dist_between = torch.cdist(
            output.unsqueeze(0), X[classes != class_].double()
        ).mean()
        if dist_between > 0:
            between_distances.append(dist_between)
    within_distances = torch.mean(torch.stack(within_distances))
    between_distances = torch.mean(torch.stack(between_distances))

    with open(output_root_path / "generated_distances.json", "w") as f:
        json.dump(
            {
                "avg_distance_from_same_cluster": within_distances.item(),
                "avg_distance_from_other_clusters": between_distances.item(),
            },
            f,
        )
