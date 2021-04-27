import unittest
from sklearn.svm import SVC, LinearSVC
from src.utils import sklearn2json
from os import remove
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree
from sklearn import svm, discriminant_analysis, dummy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, _gb_losses
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from nltk.tokenize import word_tokenize
import numpy
from sklearn.decomposition import TruncatedSVD


class TestSklearn2Json(unittest.TestCase):
    def test_to_json_from_json_SVC_1(self):
        svc = SVC(degree=4)
        X = [[0, 2, 1],
             [3, 3, 3]]
        y = [0, 1]

        svc.fit(X, y)
        # save data
        sklearn2json.to_json(svc, "svc_model.json")

        # load the saved matrix from the saved object
        test_svc = sklearn2json.from_json("svc_model.json")

        # the attributes of the svc class should be identical
        self.assertEqual(svc.class_weight_.all(), test_svc.class_weight_.all())
        self.assertEqual(svc.classes_.all(), test_svc.classes_.all())
        self.assertEqual(svc.support_vectors_.all(), test_svc.support_vectors_.all())
        self.assertEqual(svc.degree, test_svc.degree)
        print("Missing keys from saved model: {}\n".format(set(svc.__dict__.keys()) - set(test_svc.__dict__.keys())))
        for key, value in svc.__dict__.items():
            if type(value) != type(test_svc.__dict__.get(key)):
                print(key)
                print(value, test_svc.__dict__.get(key))
                print(type(value), type(test_svc.__dict__.get(key)))
        self.assertEqual(svc.get_params(), test_svc.get_params())
        self.assertEqual(set(svc.__dict__.keys()), set(test_svc.__dict__.keys()))
        remove("svc_model.json")



    def test_to_json_from_json_Multinomial_NB(self):
        mnb = MultinomialNB()
        X = [[0, 2, 1],
             [3, 3, 3],
             [1, 1, 1]]
        y = [0, 1, 1]

        mnb.fit(X, y)
        # save data
        sklearn2json.to_json(mnb, "mnb_model.json")

        # load the saved matrix from the saved object
        test_mnb = sklearn2json.from_json("mnb_model.json")

        # the attributes of the mnb class should be identical
        self.assertListEqual(list(mnb.classes_), list(test_mnb.classes_))
        self.assertListEqual(list(mnb.class_count_), list(test_mnb.class_count_))
        self.assertListEqual(list(mnb.class_log_prior_), list(test_mnb.class_log_prior_))

        for idx in range(len(mnb.feature_count_)):
            self.assertListEqual(list(mnb.feature_count_[idx]), list(test_mnb.feature_count_[idx]))

        for idx in range(len(mnb.feature_count_)):
            self.assertListEqual(list(mnb.feature_log_prob_[idx]), list(test_mnb.feature_log_prob_[idx]))

        # print("{}\n{}".format(mnb.__dict__, test_mnb.__dict__))
        print("{}\n".format(set(mnb.__dict__.keys()) - set(test_mnb.__dict__.keys())))
        for key, value in mnb.__dict__.items():
            if type(value) != type(test_mnb.__dict__.get(key)):
                print(value, test_mnb.__dict__.get(key))
                print(type(value), type(test_mnb.__dict__.get(key)))
        self.assertEqual(mnb.get_params(), test_mnb.get_params())
        self.assertEqual(set(mnb.__dict__.keys()), set(test_mnb.__dict__.keys()))
        # self.assertDictEqual(mnb.__dict__, test_mnb.__dict__)
        remove("mnb_model.json")

    def test_to_json_from_json_Bernoulli_NB(self):
        model = BernoulliNB(alpha=0.5)
        X = [[0, 2, 1],
             [3, 3, 3],
             [1, 1, 1]]
        y = [0, 1, 1]

        model.fit(X, y)
        # save data
        sklearn2json.to_json(model, "model.json")

        # load the saved matrix from the saved object
        test_model = sklearn2json.from_json("model.json")
        print(
            "Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
        for key, value in model.__dict__.items():
            if type(value) != type(test_model.__dict__.get(key)):
                print(key)
                print(value, test_model.__dict__.get(key))
                print(type(value), type(test_model.__dict__.get(key)))
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()), set(test_model.__dict__.keys()))

        remove("model.json")

    def test_to_json_from_json_SVC(self):
        svc = LinearSVC()
        X = [[0, 2, 1],
             [3, 3, 3]]
        y = [0, 1]

        svc.fit(X, y)
        # save data
        sklearn2json.to_json(svc, "lin_svc_model.json")

        # load the saved matrix from the saved object
        test_svc = sklearn2json.from_json("lin_svc_model.json")

        # the attributes of the svc class should be identical
        self.assertEqual(svc.coef_.all(), test_svc.coef_.all())
        self.assertEqual(svc.classes_.all(), test_svc.classes_.all())
        self.assertEqual(svc.intercept_.all(), test_svc.intercept_.all())
        self.assertEqual(svc.n_iter_, test_svc.n_iter_)
        remove("lin_svc_model.json")

    def test_to_json_from_json_Gaussian_NB(self):
        model = GaussianNB()
        X = [[0, 2, 1],
             [3, 3, 3],
             [1, 1, 1]]
        y = [0, 1, 1]

        model.fit(X, y)
        # save data
        sklearn2json.to_json(model, "model.json")

        # load the saved matrix from the saved object
        test_model = sklearn2json.from_json("model.json")
        print(
            "Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
        for key, value in model.__dict__.items():
            if type(value) != type(test_model.__dict__.get(key)):
                print(key)
                print(value, test_model.__dict__.get(key))
                print(type(value), type(test_model.__dict__.get(key)))
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()), set(test_model.__dict__.keys()))

        remove("model.json")

    def test_to_json_from_json_Complement_NB(self):
        model = GaussianNB()
        X = [[0, 2, 1],
             [3, 3, 3],
             [1, 1, 1]]
        y = [0, 1, 1]

        model.fit(X, y)
        # save data
        sklearn2json.to_json(model, "model.json")

        # load the saved matrix from the saved object
        test_model = sklearn2json.from_json("model.json")
        print(
            "Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
        for key, value in model.__dict__.items():
            if type(value) != type(test_model.__dict__.get(key)):
                print(key)
                print(value, test_model.__dict__.get(key))
                print(type(value), type(test_model.__dict__.get(key)))
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()), set(test_model.__dict__.keys()))

        remove("model.json")

    def test_to_json_from_json_Logistic_regression(self):
        model = LogisticRegression()
        X = [[0, 2, 1],
             [3, 3, 3],
             [1, 1, 1]]
        y = [0, 1, 1]

        model.fit(X, y)
        # save data
        sklearn2json.to_json(model, "model.json")

        # load the saved matrix from the saved object
        test_model = sklearn2json.from_json("model.json")
        print(
            "Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
        for key, value in model.__dict__.items():
            if type(value) != type(test_model.__dict__.get(key)):
                print(key)
                print(value, test_model.__dict__.get(key))
                print(type(value), type(test_model.__dict__.get(key)))
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()), set(test_model.__dict__.keys()))

        remove("model.json")

    def test_to_json_from_json_linear_discriminant_analysis(self):
        model = discriminant_analysis.LinearDiscriminantAnalysis()
        X = [[0, 2, 1],
             [3, 3, 3],
             [1, 1, 1]]
        y = [0, 1, 1]

        model.fit(X, y)
        # save data
        sklearn2json.to_json(model, "model.json")

        # load the saved matrix from the saved object
        test_model = sklearn2json.from_json("model.json")
        print(
            "Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
        for key, value in model.__dict__.items():
            if type(value) != type(test_model.__dict__.get(key)):
                print(key)
                print(value, test_model.__dict__.get(key))
                print(type(value), type(test_model.__dict__.get(key)))
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()), set(test_model.__dict__.keys()))

        remove("model.json")

    def test_to_json_from_json_quadratic_discriminant_analysis(self):
        model = discriminant_analysis.QuadraticDiscriminantAnalysis()
        X = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
        y = [1, 1, 1, 2, 2, 2]

        model.fit(X, y)
        # save data
        sklearn2json.to_json(model, "model.json")

        # load the saved matrix from the saved object
        test_model = sklearn2json.from_json("model.json")
        print(
            "Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
        for key, value in model.__dict__.items():
            if type(value) != type(test_model.__dict__.get(key)):
                print(key)
                print(value, test_model.__dict__.get(key))
                print(type(value), type(test_model.__dict__.get(key)))
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()), set(test_model.__dict__.keys()))

        remove("model.json")

    def test_to_json_from_json_gradient_boosting(self):
        model = GradientBoostingClassifier()
        X = [[0, 2, 1],
             [3, 3, 3],
             [1, 1, 1]]
        y = [0, 1, 1]

        model.fit(X, y)
        # save data
        sklearn2json.to_json(model, "model.json")

        # load the saved matrix from the saved object
        test_model = sklearn2json.from_json("model.json")
        print(
            "Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
        for key, value in model.__dict__.items():
            if type(value) != type(test_model.__dict__.get(key)):
                print(key)
                print(value, test_model.__dict__.get(key))
                print(type(value), type(test_model.__dict__.get(key)))
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()) - {"_rng"}, set(test_model.__dict__.keys()))

        remove("model.json")

    def test_to_json_from_json_random_forest(self):
        model = RandomForestClassifier()
        X = [[0, 2, 1],
             [3, 3, 3],
             [1, 1, 1]]
        y = [0, 1, 1]

        model.fit(X, y)
        # save data
        sklearn2json.to_json(model, "model.json")

        # load the saved matrix from the saved object
        test_model = sklearn2json.from_json("model.json")
        print(
            "Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
        for key, value in model.__dict__.items():
            if type(value) != type(test_model.__dict__.get(key)):
                print(key)
                print(value, test_model.__dict__.get(key))
                print(type(value), type(test_model.__dict__.get(key)))
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()), set(test_model.__dict__.keys()))

        remove("model.json")

    def test_to_json_from_json_kmeans(self):
        model = KMeans(n_clusters=2)
        X = [[0, 2, 1],
             [3, 3, 3],
             [1, 1, 1],
             [2, 2, 2],
             [1, 3, 2],
             [1, 5, 4]
             ]

        model.fit(X)
        # save data
        sklearn2json.to_json(model, "model.json")

        # load the saved matrix from the saved object
        test_model = sklearn2json.from_json("model.json")
        print(
            "Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
        for key, value in model.__dict__.items():
            if type(value) != type(test_model.__dict__.get(key)):
                print(key)
                print(value, test_model.__dict__.get(key))
                print(type(value), type(test_model.__dict__.get(key)))
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()), set(test_model.__dict__.keys()))

        remove("model.json")

    def test_to_json_from_json_dbscan(self):
        model = DBSCAN()
        X = [[0, 2, 1],
             [3, 3, 3],
             [1, 1, 1],
             [2, 2, 2],
             [1, 3, 2],
             [1, 5, 4]
             ]
        new_x = [[0, 1, 2]]
        model.fit(X)
        # save data
        sklearn2json.to_json(model, "model.json")

        # load the saved matrix from the saved object
        test_model = sklearn2json.from_json("model.json")
        print(
            "Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
        for key, value in model.__dict__.items():
            if type(value) != type(test_model.__dict__.get(key)):
                print(key)
                print(value, test_model.__dict__.get(key))
                print(type(value), type(test_model.__dict__.get(key)))
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()), set(test_model.__dict__.keys()))
        self.assertListEqual(model.fit_predict(new_x).tolist(), test_model.fit_predict(new_x).tolist())
        remove("model.json")

    def test_to_json_from_json_lsa(self):
        model = TruncatedSVD(n_components=3)
        X = [[0, 2, 1, 2],
             [3, 3, 3, 2],
             [1, 1, 1, 2],
             [2, 2, 2, 1],
             [1, 3, 2, 1],
             [1, 5, 4, 5]]
        new_x = [[0, 1, 2, 3]]
        model.fit(X)
        # save data
        sklearn2json.to_json(model, "model.json")

        # load the saved matrix from the saved object
        test_model = sklearn2json.from_json("model.json")
        print(
            "Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
        for key, value in model.__dict__.items():
            if type(value) != type(test_model.__dict__.get(key)):
                print(key)
                print(value, test_model.__dict__.get(key))
                print(type(value), type(test_model.__dict__.get(key)))
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()), set(test_model.__dict__.keys()))
        self.assertListEqual(model.transform(new_x).tolist(), test_model.transform(new_x).tolist())
        remove("model.json")
