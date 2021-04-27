from classification import serialize_tree, deserialize_tree
from sklearn.tree import DecisionTreeRegressor


def serialize_decision_tree_regressor(model):
    tree, dtypes = serialize_tree(model.tree_)
    serialized_model = {
        'meta': 'decision-tree-regression',
        'feature_importances_': model.feature_importances_.tolist(),
        'max_features_': model.max_features_,
        'n_features_': model.n_features_,
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

    tree = deserialize_tree(model_dict['tree_'], model_dict['n_features_'], 1, model_dict['n_outputs_'])
    deserialized_decision_tree.tree_ = tree

    return deserialized_decision_tree
