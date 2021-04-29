import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn2json.classification import serialize_bernoulli_nb, deserialize_bernoulli_nb, deserialize_gaussian_nb, \
    serialize_gaussian_nb, deserialize_complement_nb, serialize_complement_nb, deserialize_multinomial_nb, \
    serialize_multinomial_nb, serialize_linear_svm, deserialize_linear_svm, deserialize_gradient_boosting, \
    serialize_gradient_boosting, serialize_gradient_boosting_dummy_classifier, serialize_svm, deserialize_svm, \
    deserialize_random_forest, serialize_random_forest, serialize_lda, deserialize_lda, serialize_qda, deserialize_qda, \
    deserialize_logistic_regression, serialize_logistic_regression
from sklearn2json.label_encoders import deserialize_label_binarizer, serialize_label_binarizer
from sklearn2json.clustering import serialize_dbscan_clustering, deserialize_dbscan_clustering, serialize_k_means, \
    deserialize_k_means
from sklearn2json.dimension_reduction import serialize_lsa, deserialize_lsa
from sklearn2json.regression import deserialize_decision_tree_regressor, serialize_decision_tree_regressor, \
    serialize_linear_regressor, deserialize_linear_regressor, deserialize_lasso_regressor, serialize_lasso_regressor, \
    deserialize_elastic_regressor, serialize_elastic_regressor, deserialize_ridge_regressor, serialize_ridge_regressor, \
    serialize_svr, deserialize_svr, serialize_mlp_regressor, deserialize_mlp_regressor, \
    deserialize_random_forest_regressor, serialize_random_forest_regressor, deserialize_gradient_boosting_regressor, \
    serialize_gradient_boosting_regressor
from sklearn2json.vectorizer import serialize_tfidf, deserialize_tfidf
from sklearn import discriminant_analysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, \
    GradientBoostingRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn2json.csr import serialize_csr_matrix, deserialize_csr_matrix
from scipy.sparse import csr_matrix

def deserialize_model(model_dict):
    if model_dict['meta'] == 'svm':
        return deserialize_svm(model_dict)
    elif model_dict['meta'] == 'tfidf':
        return deserialize_tfidf(model_dict)
    elif model_dict['meta'] == 'multinomial-nb':
        return deserialize_multinomial_nb(model_dict)
    elif model_dict["meta"] == "linear_svm":
        return deserialize_linear_svm(model_dict)
    elif model_dict['meta'] == 'bernoulli-nb':
        return deserialize_bernoulli_nb(model_dict)
    elif model_dict['meta'] == 'gaussian-nb':
        return deserialize_gaussian_nb(model_dict)
    elif model_dict['meta'] == 'complement-nb':
        return deserialize_complement_nb(model_dict)
    elif model_dict['meta'] == 'lr':
        return deserialize_logistic_regression(model_dict)
    elif model_dict['meta'] == 'lda':
        return deserialize_lda(model_dict)
    elif model_dict['meta'] == 'qda':
        return deserialize_qda(model_dict)
    elif model_dict['meta'] == 'gb':
        return deserialize_gradient_boosting(model_dict)
    elif model_dict['meta'] == "random_forest":
        return deserialize_random_forest(model_dict)
    elif model_dict["meta"] == "kmeans":
        return deserialize_k_means(model_dict)
    elif model_dict["meta"] == "lsa":
        return deserialize_lsa(model_dict)
    elif model_dict["meta"] == "dbscan":
        return deserialize_dbscan_clustering(model_dict)
    elif model_dict["meta"] == 'decision-tree-regression':
        return deserialize_decision_tree_regressor(model_dict)
    elif model_dict["meta"] == 'linear-regression':
        return deserialize_linear_regressor(model_dict)
    elif model_dict["meta"] == "lasso-regression":
        return deserialize_lasso_regressor(model_dict)
    elif model_dict["meta"] == 'elasticnet-regression':
        return deserialize_elastic_regressor(model_dict)
    elif model_dict["meta"] == "ridge-regression":
        return deserialize_ridge_regressor(model_dict)
    elif model_dict["meta"] == "svr":
        return deserialize_svr(model_dict)
    elif model_dict["meta"] == "mlp-regression":
        return deserialize_mlp_regressor(model_dict)
    elif model_dict["meta"] == "rf-regression":
        return deserialize_random_forest_regressor(model_dict)
    elif model_dict["meta"] == 'gb-regression':
        return deserialize_gradient_boosting_regressor(model_dict)
    elif model_dict["meta"] == 'label-binarizer':
        return deserialize_label_binarizer(model_dict)
    elif model_dict["meta"] == 'csr':
        return deserialize_csr_matrix(model_dict)
    else:
        raise ValueError("Model type cannot be found in deserialize_model function. Please implement it!")


def from_json(file_name):
    with open(file_name, 'r') as model_json:
        model_dict = json.load(model_json)
        return deserialize_model(model_dict)


def serialize_model(model):
    if isinstance(model, SVC):
        return serialize_svm(model)
    elif isinstance(model, TfidfVectorizer):
        return serialize_tfidf(model)
    elif isinstance(model, MultinomialNB):
        return serialize_multinomial_nb(model)
    elif isinstance(model, BernoulliNB):
        return serialize_bernoulli_nb(model)
    elif isinstance(model, GaussianNB):
        return serialize_gaussian_nb(model)
    elif isinstance(model, ComplementNB):
        return serialize_complement_nb(model)
    elif isinstance(model, LogisticRegression):
        return serialize_logistic_regression(model)
    elif isinstance(model, discriminant_analysis.LinearDiscriminantAnalysis):
        return serialize_lda(model)
    elif isinstance(model, discriminant_analysis.QuadraticDiscriminantAnalysis):
        return serialize_qda(model)
    elif isinstance(model, GradientBoostingClassifier):
        return serialize_gradient_boosting(model)
    elif isinstance(model, RandomForestClassifier):
        return serialize_random_forest(model)
    elif isinstance(model, LinearSVC):
        return serialize_linear_svm(model)
    elif isinstance(model, KMeans):
        return serialize_k_means(model)
    elif isinstance(model, TruncatedSVD):
        return serialize_lsa(model)
    elif isinstance(model, DBSCAN):
        return serialize_dbscan_clustering(model)
    elif isinstance(model, DecisionTreeRegressor):
        return serialize_decision_tree_regressor(model)
    elif isinstance(model, LinearRegression):
        return serialize_linear_regressor(model)
    elif isinstance(model, Lasso):
        return serialize_lasso_regressor(model)
    elif isinstance(model, ElasticNet):
        return serialize_elastic_regressor(model)
    elif isinstance(model, Ridge):
        return serialize_ridge_regressor(model)
    elif isinstance(model, SVR):
        return serialize_svr(model)
    elif isinstance(model, MLPRegressor):
        return serialize_mlp_regressor(model)
    elif isinstance(model, RandomForestRegressor):
        return serialize_random_forest_regressor(model)
    elif isinstance(model, GradientBoostingRegressor):
        return serialize_gradient_boosting_regressor(model)
    elif isinstance(model, LabelBinarizer):
        return serialize_label_binarizer(model)
    elif isinstance(model, csr_matrix):
        return serialize_csr_matrix(model)
    else:
        raise ValueError("Model type cannot be found in serialize_model function. Please implement it!")


def to_json(model, model_name):
    with open(model_name, 'w') as model_json:
        json.dump(serialize_model(model), model_json)
