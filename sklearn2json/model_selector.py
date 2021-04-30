from scipy.sparse import csr_matrix
from sklearn import discriminant_analysis
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn2json.classification import deserialize_bernoulli_nb
from sklearn2json.classification import deserialize_complement_nb
from sklearn2json.classification import deserialize_decision_tree
from sklearn2json.classification import deserialize_gaussian_nb
from sklearn2json.classification import deserialize_gradient_boosting
from sklearn2json.classification import deserialize_lda
from sklearn2json.classification import deserialize_linear_svm
from sklearn2json.classification import deserialize_logistic_regression
from sklearn2json.classification import deserialize_mlp
from sklearn2json.classification import deserialize_multinomial_nb
from sklearn2json.classification import deserialize_perceptron
from sklearn2json.classification import deserialize_qda
from sklearn2json.classification import deserialize_random_forest
from sklearn2json.classification import deserialize_svm
from sklearn2json.classification import serialize_bernoulli_nb
from sklearn2json.classification import serialize_complement_nb
from sklearn2json.classification import serialize_decision_tree
from sklearn2json.classification import serialize_gaussian_nb
from sklearn2json.classification import serialize_gradient_boosting
from sklearn2json.classification import serialize_lda
from sklearn2json.classification import serialize_linear_svm
from sklearn2json.classification import serialize_logistic_regression
from sklearn2json.classification import serialize_mlp
from sklearn2json.classification import serialize_multinomial_nb
from sklearn2json.classification import serialize_perceptron
from sklearn2json.classification import serialize_qda
from sklearn2json.classification import serialize_random_forest
from sklearn2json.classification import serialize_svm
from sklearn2json.clustering import deserialize_dbscan_clustering
from sklearn2json.clustering import deserialize_k_means
from sklearn2json.clustering import serialize_dbscan_clustering
from sklearn2json.clustering import serialize_k_means
from sklearn2json.csr import deserialize_csr_matrix
from sklearn2json.csr import serialize_csr_matrix
from sklearn2json.dimension_reduction import deserialize_lsa
from sklearn2json.dimension_reduction import serialize_lsa
from sklearn2json.label_encoders import deserialize_label_binarizer
from sklearn2json.label_encoders import serialize_label_binarizer
from sklearn2json.regression import deserialize_decision_tree_regressor
from sklearn2json.regression import deserialize_elastic_regressor
from sklearn2json.regression import deserialize_gradient_boosting_regressor
from sklearn2json.regression import deserialize_lasso_regressor
from sklearn2json.regression import deserialize_linear_regressor
from sklearn2json.regression import deserialize_mlp_regressor
from sklearn2json.regression import deserialize_random_forest_regressor
from sklearn2json.regression import deserialize_ridge_regressor
from sklearn2json.regression import deserialize_svr
from sklearn2json.regression import serialize_decision_tree_regressor
from sklearn2json.regression import serialize_elastic_regressor
from sklearn2json.regression import serialize_gradient_boosting_regressor
from sklearn2json.regression import serialize_lasso_regressor
from sklearn2json.regression import serialize_linear_regressor
from sklearn2json.regression import serialize_mlp_regressor
from sklearn2json.regression import serialize_random_forest_regressor
from sklearn2json.regression import serialize_ridge_regressor
from sklearn2json.regression import serialize_svr
from sklearn2json.vectorizer import deserialize_tfidf
from sklearn2json.vectorizer import serialize_tfidf

# Dictionary containing mapping for model deserialization (loading model from json), classification category
classification_dict = {
    "svm": deserialize_svm,
    "multinomial-nb": deserialize_multinomial_nb,
    "linear_svm": deserialize_linear_svm,
    "bernoulli-nb": deserialize_bernoulli_nb,
    "gaussian-nb": deserialize_gaussian_nb,
    "complement-nb": deserialize_complement_nb,
    "lr": deserialize_logistic_regression,
    "lda": deserialize_lda,
    "qda": deserialize_qda,
    "gb": deserialize_gradient_boosting,
    "random_forest": deserialize_random_forest,
    "perceptron": deserialize_perceptron,
    "mlp": deserialize_mlp,
    "decision-tree": deserialize_decision_tree,
}
# Dictionary containing mapping for model deserialization (loading model from json), regression category
regression_dict = {
    "decision-tree-regression": deserialize_decision_tree_regressor,
    "linear-regression": deserialize_linear_regressor,
    "lasso-regression": deserialize_lasso_regressor,
    "elasticnet-regression": deserialize_elastic_regressor,
    "ridge-regression": deserialize_ridge_regressor,
    "svr": deserialize_svr,
    "mlp-regression": deserialize_mlp_regressor,
    "rf-regression": deserialize_random_forest_regressor,
    "gb-regression": deserialize_gradient_boosting_regressor,
}
# Dictionary containing mapping for model deserialization (loading model from json), clustering category
clustering_dict = {"kmeans": deserialize_k_means, "dbscan": deserialize_dbscan_clustering}
# Dictionary containing mapping for model deserialization (loading model from json), dimension reduction category
dimension_reduction_dict = {"lsa": deserialize_lsa}
# Dictionary containing mapping for model deserialization (loading model from json), vectorizer category
vectorizer_dict = {"tfidf": deserialize_tfidf}
# Dictionary containing mapping for model deserialization (loading model from json), label encoder category
label_encoder_dict = {"label-binarizer": deserialize_label_binarizer}
# Dictionary containing mapping for model deserialization (loading model from json), sparse matrix category
csr_dict = {"csr": deserialize_csr_matrix}


def deserialize_model(model_dict):
    """
    Function for selecting the right deserialization function based on loaded json model's "meta" key.
    :param model_dict: JSON model to be loaded.
    :return: Loaded JSON model
    """
    if model_dict["meta"] in classification_dict:
        return classification_dict[model_dict["meta"]](model_dict)
    elif model_dict["meta"] in regression_dict:
        return regression_dict[model_dict["meta"]](model_dict)
    elif model_dict["meta"] in clustering_dict:
        return clustering_dict[model_dict["meta"]](model_dict)
    elif model_dict["meta"] in dimension_reduction_dict:
        return dimension_reduction_dict[model_dict["meta"]](model_dict)
    elif model_dict["meta"] in vectorizer_dict:
        return vectorizer_dict[model_dict["meta"]](model_dict)
    elif model_dict["meta"] in label_encoder_dict:
        return label_encoder_dict[model_dict["meta"]](model_dict)
    elif model_dict["meta"] in csr_dict:
        return csr_dict[model_dict["meta"]](model_dict)
    else:
        raise ValueError("Model type cannot be found in deserialize_model function. Please implement it!")


# Dictionary containing mapping for model serialization (saving model to json), classification category
classification_serialization_dict = {
    SVC: serialize_svm,
    MultinomialNB: serialize_multinomial_nb,
    BernoulliNB: serialize_bernoulli_nb,
    GaussianNB: serialize_gaussian_nb,
    ComplementNB: serialize_complement_nb,
    LogisticRegression: serialize_logistic_regression,
    discriminant_analysis.LinearDiscriminantAnalysis: serialize_lda,
    discriminant_analysis.QuadraticDiscriminantAnalysis: serialize_qda,
    GradientBoostingClassifier: serialize_gradient_boosting,
    RandomForestClassifier: serialize_random_forest,
    LinearSVC: serialize_linear_svm,
    Perceptron: serialize_perceptron,
    MLPClassifier: serialize_mlp,
    DecisionTreeClassifier: serialize_decision_tree,
}

# Dictionary containing mapping for model serialization (saving model to json), regression category
regression_serialization_dict = {
    SVR: serialize_svr,
    DecisionTreeRegressor: serialize_decision_tree_regressor,
    LinearRegression: serialize_linear_regressor,
    Lasso: serialize_lasso_regressor,
    ElasticNet: serialize_elastic_regressor,
    Ridge: serialize_ridge_regressor,
    MLPRegressor: serialize_mlp_regressor,
    RandomForestRegressor: serialize_random_forest_regressor,
    GradientBoostingRegressor: serialize_gradient_boosting_regressor,
}
# Dictionary containing mapping for model serialization (saving model to json), clustering category
clustering_serialization_dict = {KMeans: serialize_k_means, DBSCAN: serialize_dbscan_clustering}
# Dictionary containing mapping for model serialization (saving model to json), dimension reduction category
dimred_serialization_dict = {TruncatedSVD: serialize_lsa}
# Dictionary containing mapping for model serialization (saving model to json), label encoding category
label_encoder_serialization_dict = {LabelBinarizer: serialize_label_binarizer}
# Dictionary containing mapping for model serialization (saving model to json), vectorizer category
vectorization_serialization_dict = {TfidfVectorizer: serialize_tfidf}
# Dictionary containing mapping for model serialization (saving model to json), sparse matrix category
csr_serialization_dict = {csr_matrix: serialize_csr_matrix}


def select_serialization_model(model, serialization_dict):
    """
    Finds the appropriate serializer function for a given sklearn model.
    :param model: sklearn model to be saved
    :param serialization_dict: dictionary containing mapping for sklearn classes to serialization functions
    :return:
    """
    for class_name in serialization_dict:
        if isinstance(model, class_name):
            return serialization_dict[class_name](model)


def serialize_model(model):
    """
    Saves (serializes) sklearn model into json.
    :param model: sklearn model
    :return: model in JSON format
    """
    if any(isinstance(model, key) for key in classification_serialization_dict):
        return select_serialization_model(model, classification_serialization_dict)
    elif any(isinstance(model, key) for key in regression_serialization_dict):
        return select_serialization_model(model, regression_serialization_dict)
    elif any(isinstance(model, key) for key in clustering_serialization_dict):
        return select_serialization_model(model, clustering_serialization_dict)
    elif any(isinstance(model, key) for key in dimred_serialization_dict):
        return select_serialization_model(model, dimred_serialization_dict)
    elif any(isinstance(model, key) for key in vectorization_serialization_dict):
        return select_serialization_model(model, vectorization_serialization_dict)
    elif any(isinstance(model, key) for key in label_encoder_serialization_dict):
        return select_serialization_model(model, label_encoder_serialization_dict)
    elif any(isinstance(model, key) for key in csr_serialization_dict):
        return select_serialization_model(model, csr_serialization_dict)
    else:
        raise ValueError("Model type cannot be found in serialize_model function. Please implement it!")
