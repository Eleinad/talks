from sklearn.cluster import HDBSCAN
import umap
from sklearn.metrics import silhouette_score
from optuna import trial
import optuna


# Optuna objective function: runs one trial of UMAP + HDBSCAN with sampled hyperparameters
# and returns the silhouette score (higher = better cluster separation).
# Called repeatedly by optuna.Study.optimize() in main() and in ClipFaissIndex.find_best_bertopic_parameters().
def objective_fn_umap_hdbscan(trial, data):
    silh_scores = {}
    try:

        # UMAP hyperparameters
        # Upper bound is len(data)/2 to keep the search space meaningful relative to dataset size
        n_neighbors = trial.suggest_int("n_neighbors", 2, int(len(data)/2))
        n_components = trial.suggest_int("n_components", 2, int(len(data)/2))
        min_dist = trial.suggest_float("min_dist", 0.0, 0.99)  # 1.0 excluded: would spread all points uniformly, destroying cluster structure
        metric_umap = trial.suggest_categorical("umap_metric", ["euclidean", "cosine"])

        # HDBSCAN hyperparameters
        min_cluster_size = trial.suggest_int("min_cluster_size", 2, int(len(data)/2))  # minimum number of points for a group to be considered a valid cluster
        min_samples = trial.suggest_int("min_samples", 2, int(len(data)/2))  # minimum number of neighbors a point needs to be treated as a core point
        cluster_selection_epsilon = trial.suggest_float("cluster_selection_epsilon", 0, .1)  # clusters closer than this distance are merged; kept small to avoid over-merging
        metric = trial.suggest_categorical("metric", ["euclidean", "cosine"])

        # Step 1: reduce high-dimensional CLIP embeddings before clustering.
        # HDBSCAN degrades in high dimensions (curse of dimensionality), so UMAP runs first.
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric_umap
        )
        reduced_data = reducer.fit_transform(data)

        # Step 2: cluster the reduced embeddings
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric
        )
        labels = clusterer.fit_predict(reduced_data)

        # Penalize degenerate results: only one cluster found, or every point labeled as noise (-1)
        if len(set(labels)) <= 1 or all(label == -1 for label in labels):
            silh_scores[trial.number] = -1
            return -1.0

        # Silhouette score measures cluster cohesion vs. separation; range [-1, 1], higher is better.
        # Scored on the original (non-reduced) data so UMAP distortions don't inflate the metric.
        silh_score=silhouette_score(data, labels)
        # if silh_scores.get((min_cluster_size,min_samples,cluster_selection_epsilon,metric)) == None:
        #   silh_scores[(min_cluster_size,min_samples,cluster_selection_epsilon,metric)]=[]
        # silh_scores[(min_cluster_size,min_samples,cluster_selection_epsilon,metric)].append(silh_score)

        return silh_score

    except Exception as e:
        print("Trial failed:", e)
        return -1.0