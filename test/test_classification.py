import unittest
from sklearn.svm import SVC, LinearSVC
import sklearn2json
from os import remove
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree
from sklearn import svm, discriminant_analysis, dummy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, _gb_losses
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier

X = [[0, 2, 1],
     [3, 3, 3],
     [1, 1, 1]]
y = [0, 1, 1]
new_x = [[0, 0, 0]]


def helper_test(model):
    # save data
    sklearn2json.to_json(model, "model.json")

    # load the saved matrix from the saved object
    test_model = sklearn2json.from_json("model.json")
    print(
        "Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
    for key, value in model.__dict__.items():
        if not isinstance(value, type(test_model.__dict__.get(key))):
            print(key)
            print(value, test_model.__dict__.get(key))
            print(type(value), type(test_model.__dict__.get(key)))
    return model, test_model


class ClassificationTestCase(unittest.TestCase):
    def test_base(self, model=discriminant_analysis.LinearDiscriminantAnalysis(), X=X, y=y, new_x=new_x, exclude_keys=[]):
        model.fit(X, y)
        model, test_model = helper_test(model)
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()-exclude_keys), set(test_model.__dict__.keys()))
        self.assertEqual(model.predict(new_x), test_model.predict(new_x))
        remove("model.json")
        return model, test_model

    def test_to_json_from_json_svc(self):
        svc = SVC(degree=4)
        svc, test_svc = self.test_base(model=svc)
        # the attributes of the svc class should be identical
        self.assertEqual(svc.class_weight_.all(), test_svc.class_weight_.all())
        self.assertEqual(svc.classes_.all(), test_svc.classes_.all())
        self.assertEqual(svc.support_vectors_.all(), test_svc.support_vectors_.all())
        self.assertEqual(svc.degree, test_svc.degree)

    def test_to_json_from_json_multinomial_nb(self):
        mnb = MultinomialNB()
        mnb, test_mnb = self.test_base(mnb)

        # the attributes of the mnb class should be identical
        self.assertListEqual(list(mnb.classes_), list(test_mnb.classes_))
        self.assertListEqual(list(mnb.class_count_), list(test_mnb.class_count_))
        self.assertListEqual(list(mnb.class_log_prior_), list(test_mnb.class_log_prior_))

        for idx in range(len(mnb.feature_count_)):
            self.assertListEqual(list(mnb.feature_count_[idx]), list(test_mnb.feature_count_[idx]))

        for idx in range(len(mnb.feature_count_)):
            self.assertListEqual(list(mnb.feature_log_prob_[idx]), list(test_mnb.feature_log_prob_[idx]))

    def test_to_json_from_json_bernoulli_nb(self):
        model = BernoulliNB(alpha=0.5)
        self.test_base(model)

    def test_to_json_from_json_linear_svc(self):
        svc = LinearSVC()
        svc, test_svc = self.test_base(svc)
        # # the attributes of the svc class should be identical
        self.assertEqual(svc.coef_.all(), test_svc.coef_.all())
        self.assertEqual(svc.classes_.all(), test_svc.classes_.all())
        self.assertEqual(svc.intercept_.all(), test_svc.intercept_.all())
        self.assertEqual(svc.n_iter_, test_svc.n_iter_)

    def test_to_json_from_json_gaussian_nb(self):
        model = GaussianNB()
        self.test_base(model)

    def test_to_json_from_json_complement_nb(self):
        model = ComplementNB()
        self.test_base(model)

    def test_to_json_from_json_logistic_regression(self):
        model = LogisticRegression()
        self.test_base(model)

    def test_to_json_from_json_linear_discriminant_analysis(self):
        model = discriminant_analysis.LinearDiscriminantAnalysis()
        self.test_base(model)

    def test_to_json_from_json_quadratic_discriminant_analysis(self):
        model = discriminant_analysis.QuadraticDiscriminantAnalysis()
        X = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
        y = [1, 1, 1, 2, 2, 2]
        new_x = [[0, 0]]
        self.test_base(model, X, y, new_x)

    def test_to_json_from_json_gradient_boosting(self):
        model = GradientBoostingClassifier()
        self.test_base(model, exclude_keys=["_rng"])


    def test_to_json_from_json_random_forest(self):
        model = RandomForestClassifier()
        self.test_base(model)

    def test_to_json_from_json_perceptron(self):
        model = Perceptron()
        self.test_base(model, exclude_keys=["loss_function_"])

    def test_to_json_from_json_mlp(self):
        model = MLPClassifier()
        self.test_base(model, exclude_keys=["_random_state", "_optimizer"])

    def test_to_json_from_json_decision_tree(self):
        model = DecisionTreeClassifier()
        self.test_base(model, exclude_keys=[])
