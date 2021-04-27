from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import Tree
import numpy as np

def serialize_tree(tree):
    serialized_tree = tree.__getstate__()

    dtypes = serialized_tree['nodes'].dtype
    serialized_tree['nodes'] = serialized_tree['nodes'].tolist()
    serialized_tree['values'] = serialized_tree['values'].tolist()

    return serialized_tree, dtypes


def deserialize_tree(tree_dict, n_features, n_classes, n_outputs):
    tree_dict['nodes'] = [tuple(lst) for lst in tree_dict['nodes']]

    names = ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples',
             'weighted_n_node_samples']
    tree_dict['nodes'] = np.array(tree_dict['nodes'],
                                  dtype=np.dtype({'names': names, 'formats': tree_dict['nodes_dtype']}))
    tree_dict['values'] = np.array(tree_dict['values'])

    tree = Tree(n_features, np.array([n_classes], dtype=np.intp), n_outputs)
    tree.__setstate__(tree_dict)

    return tree


def serialize_decision_tree_regressor(model):
    tree, dtypes = serialize_tree(model.tree_)
    serialized_model = {
        'meta': 'decision-tree-regression',
        'feature_importances_': model.feature_importances_.tolist(),
        'max_features_': model.max_features_,
        'n_features_': model.n_features_,
        'n_features_in_': model.n_features_in_,
        'n_outputs_': model.n_outputs_,
        'tree_': tree
    }

    tree_dtypes = []
    for i in range(0, len(dtypes)):
        tree_dtypes.append(dtypes[i].str)

    serialized_model['tree_']['nodes_dtype'] = tree_dtypes

    return serialized_model


def deserialize_decision_tree_regressor(model_dict):
    deserialized_decision_tree = DecisionTreeRegressor()

    deserialized_decision_tree.max_features_ = model_dict['max_features_']
    deserialized_decision_tree.n_features_ = model_dict['n_features_']
    deserialized_decision_tree.n_outputs_ = model_dict['n_outputs_']
    deserialized_decision_tree.n_features_in_ = model_dict['n_features_in_']
    tree = deserialize_tree(model_dict['tree_'], model_dict['n_features_'], 1, model_dict['n_outputs_'])
    deserialized_decision_tree.tree_ = tree

    return deserialized_decision_tree
