import numpy as np
from sklearn.tree._tree import Tree


def serialize_tree(tree):
    serialized_tree = tree.__getstate__()

    dtypes = serialized_tree["nodes"].dtype
    serialized_tree["nodes"] = serialized_tree["nodes"].tolist()
    serialized_tree["values"] = serialized_tree["values"].tolist()

    return serialized_tree, dtypes


def deserialize_tree(tree_dict, n_features, n_classes, n_outputs):
    tree_dict["nodes"] = [tuple(lst) for lst in tree_dict["nodes"]]

    names = [
        "left_child",
        "right_child",
        "feature",
        "threshold",
        "impurity",
        "n_node_samples",
        "weighted_n_node_samples",
    ]
    tree_dict["nodes"] = np.array(
        tree_dict["nodes"], dtype=np.dtype({"names": names, "formats": tree_dict["nodes_dtype"]})
    )
    tree_dict["values"] = np.array(tree_dict["values"])

    tree = Tree(n_features, np.array([n_classes], dtype=np.intp), n_outputs)
    tree.__setstate__(tree_dict)

    return tree
