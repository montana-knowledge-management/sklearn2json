import unittest
from os import remove
from test.helper_test import print_differences

from sklearn import discriminant_analysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# from sklearn.tree._tree import Tree

input_x = [[0, 2, 1], [3, 3, 3], [1, 1, 1]]
output_y = [0, 1, 1]
new_x = [[0, 0, 0]]


class ClassificationTestCase(unittest.TestCase):
    def base(self, model, x_input, y_output, new_x_input, exclude_keys=None):
        if not exclude_keys:
            exclude_keys = []
        model.fit(x_input, y_output)
        model, test_model = print_differences(model)
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys() - exclude_keys), set(test_model.__dict__.keys()))
        self.assertEqual(model.predict(new_x_input), test_model.predict(new_x_input))
        remove("model.json")
        return model, test_model

    def test_to_json_from_json_svc(self):
        svc = SVC(degree=4)
        svc, test_svc = self.base(svc, input_x, output_y, new_x)
        # the attributes of the svc class should be identical
        self.assertEqual(svc.class_weight_.all(), test_svc.class_weight_.all())
        self.assertEqual(svc.classes_.all(), test_svc.classes_.all())
        self.assertEqual(svc.support_vectors_.all(), test_svc.support_vectors_.all())
        self.assertEqual(svc.degree, test_svc.degree)

    def test_to_json_from_json_multinomial_nb(self):
        mnb = MultinomialNB()
        mnb, test_mnb = self.base(mnb, input_x, output_y, new_x)

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
        self.base(model, input_x, output_y, new_x)

    def test_to_json_from_json_linear_svc(self):
        svc = LinearSVC()
        svc, test_svc = self.base(svc, input_x, output_y, new_x)
        # # the attributes of the svc class should be identical
        self.assertEqual(svc.coef_.all(), test_svc.coef_.all())
        self.assertEqual(svc.classes_.all(), test_svc.classes_.all())
        self.assertEqual(svc.intercept_.all(), test_svc.intercept_.all())
        self.assertEqual(svc.n_iter_, test_svc.n_iter_)

    def test_to_json_from_json_gaussian_nb(self):
        model = GaussianNB()
        self.base(model, input_x, output_y, new_x)

    def test_to_json_from_json_complement_nb(self):
        model = ComplementNB()
        self.base(model, input_x, output_y, new_x)

    def test_to_json_from_json_logistic_regression(self):
        model = LogisticRegression()
        self.base(model, input_x, output_y, new_x)

    def test_to_json_from_json_linear_discriminant_analysis(self):
        model = discriminant_analysis.LinearDiscriminantAnalysis()
        self.base(model, input_x, output_y, new_x)

    def test_to_json_from_json_quadratic_discriminant_analysis(self):
        model = discriminant_analysis.QuadraticDiscriminantAnalysis()
        X = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
        y = [1, 1, 1, 2, 2, 2]
        new_x = [[0, 0]]
        self.base(model, X, y, new_x)

    def test_to_json_from_json_gradient_boosting(self):
        model = GradientBoostingClassifier()
        self.base(model, input_x, output_y, new_x, exclude_keys=["_rng"])

    def test_to_json_from_json_random_forest(self):
        model = RandomForestClassifier()
        self.base(model, input_x, output_y, new_x)

    def test_to_json_from_json_perceptron(self):
        model = Perceptron()
        self.base(model, input_x, output_y, new_x, exclude_keys=["loss_function_"])

    def test_to_json_from_json_mlp(self):
        model = MLPClassifier()
        self.base(
            model,
            input_x,
            output_y,
            new_x,
            exclude_keys=["_random_state", "_optimizer"],
        )

    def test_to_json_from_json_decision_tree(self):
        model = DecisionTreeClassifier()
        self.base(model, input_x, output_y, new_x)
