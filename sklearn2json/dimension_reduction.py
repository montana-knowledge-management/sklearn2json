import numpy as np
from sklearn.decomposition import TruncatedSVD


def serialize_lsa(model):
    serialized_model = {
        "meta": "lsa",
        "components_": model.components_.tolist(),
        "explained_variance_ratio_": model.explained_variance_ratio_.tolist(),
        "explained_variance_": model.explained_variance_.tolist(),
        "singular_values_": model.singular_values_.tolist(),
        "n_features_in_": model.n_features_in_,
        "params": model.get_params(),
    }

    return serialized_model


def deserialize_lsa(model_dict):
    model = TruncatedSVD(**model_dict["params"])
    model.components_ = np.array(model_dict["components_"])
    model.explained_variance_ratio_ = np.array(model_dict["explained_variance_ratio_"])
    model.explained_variance_ = np.array(model_dict["explained_variance_"])
    model.singular_values_ = np.array(model_dict["singular_values_"])
    model.n_features_in_ = model_dict["n_features_in_"]
    return model
