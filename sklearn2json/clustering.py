import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


def serialize_dbscan_clustering(model):
    serialized_model = {
        "meta": "dbscan",
        "components_": model.components_.tolist(),
        "core_sample_indices_": model.core_sample_indices_.tolist(),
        "labels_": model.labels_.tolist(),
        "n_features_in_": model.n_features_in_,
        "params": model.get_params(),
    }

    return serialized_model


def deserialize_dbscan_clustering(model_dict):
    model = DBSCAN(**model_dict["params"])

    model.components_ = np.array(model_dict["components_"])
    model.labels_ = np.array(model_dict["labels_"])
    model.core_sample_indices_ = np.array(model_dict["core_sample_indices_"])
    model.n_features_in_ = model_dict["n_features_in_"]

    return model


def serialize_k_means(model):
    serialized_model = {
        "meta": "kmeans",
        "cluster_centers_": model.cluster_centers_.tolist(),
        "labels_": model.labels_.tolist(),
        "n_features_in_": model.n_features_in_,
        "_n_init": model._n_init,
        "_tol": model._tol,
        "_n_threads": model._n_threads,
        "inertia_": model.inertia_,
        "n_iter_": model.n_iter_,
        "_algorithm": model._algorithm,
        "params": model.get_params(),
    }

    return serialized_model


def deserialize_k_means(model_dict):
    model = KMeans(**model_dict["params"])
    model.cluster_centers_ = np.array(model_dict["cluster_centers_"])
    model.labels_ = np.array(model_dict["labels_"])
    model.inertia_ = model_dict["inertia_"]
    model.n_iter_ = model_dict["n_iter_"]
    model.n_features_in_ = model_dict["n_features_in_"]
    model._n_init = model_dict["_n_init"]
    model._tol = np.float64(model_dict["_tol"])
    model._n_threads = model_dict["_n_threads"]
    model._algorithm = model_dict["_algorithm"]
    return model
