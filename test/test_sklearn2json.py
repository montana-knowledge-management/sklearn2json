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

    def test_tfidf_default_settings(self):
        """
        Test saving tfidf model weights in case of default parameters.
        """
        tfidf_vectorizer = TfidfVectorizer(decode_error="ignore", encoding="ascii", strip_accents="ascii",
                                           lowercase=False, analyzer="char_wb", stop_words=["a", "stb"],
                                           token_pattern=r"(?u)\b[a-z]+?\b", ngram_range=(1, 3), max_df=0.99,
                                           min_df=0.01, binary=True, norm="l1",

                                           smooth_idf=False,
                                           sublinear_tf=True

                                           )
        test_data = ["The quick brown fox jumps over the lazy dog.",
                     "When a dog barks it won't bite you.",
                     "This one is a dog, the other is a puppy."]
        tfidf_vectorizer.fit_transform(test_data)
        sklearn2json.to_json(tfidf_vectorizer, "tfidf_vectorizer.json")
        loaded_tfidf_model = sklearn2json.from_json("tfidf_vectorizer.json")
        self.assertDictEqual(tfidf_vectorizer.vocabulary_, loaded_tfidf_model.vocabulary_)
        self.assertEqual(set(tfidf_vectorizer.idf_), set(loaded_tfidf_model.idf_))
        self.assertEqual(tfidf_vectorizer.stop_words_, loaded_tfidf_model.stop_words_)
        self.assertEqual(tfidf_vectorizer.fixed_vocabulary_, loaded_tfidf_model.fixed_vocabulary_)
        self.assertEqual("ascii", loaded_tfidf_model.encoding)
        self.assertEqual("ignore", loaded_tfidf_model.decode_error)
        self.assertEqual("ascii", loaded_tfidf_model.strip_accents)
        self.assertFalse(loaded_tfidf_model.lowercase)
        self.assertEqual(tfidf_vectorizer.analyzer, loaded_tfidf_model.analyzer)
        self.assertEqual(tfidf_vectorizer.stop_words, loaded_tfidf_model.stop_words)
        self.assertEqual(r"(?u)\b[a-z]+?\b", loaded_tfidf_model.token_pattern)
        self.assertEqual((1, 3), loaded_tfidf_model.ngram_range)
        self.assertEqual(0.99, loaded_tfidf_model.max_df)
        self.assertEqual(0.01, loaded_tfidf_model.min_df)
        self.assertTrue(loaded_tfidf_model.binary)
        self.assertEqual("l1", loaded_tfidf_model.norm)

        self.assertFalse(loaded_tfidf_model.smooth_idf)
        self.assertTrue(loaded_tfidf_model.sublinear_tf)

        remove("tfidf_vectorizer.json")

    def test_tfidf_parameters_2(self):
        """
        Test saving tfidf model weights in case of default parameters.
        """
        tfidf_vectorizer = TfidfVectorizer(
            vocabulary={'the': 14, 'quick': 13, 'brown': 2, 'fox': 4, 'jumps': 7, 'over': 11, 'lazy': 8, 'dog': 3,
                        'when': 16, 'barks': 0, 'it': 6, 'won': 17, 'bite': 1, 'you': 18, 'this': 15, 'one': 9, 'is': 5,
                        'other': 10, 'puppy': 12, "catdoll": 19},
            use_idf=False,
            max_features=100
        )
        test_data = ["The quick brown fox jumps over the lazy dog.",
                     "When a dog barks it won't bite you.",
                     "This one is a dog, the other is a puppy and catdoll."]
        tfidf_vectorizer.fit_transform(test_data)
        sklearn2json.to_json(tfidf_vectorizer, "tfidf_vectorizer.json")
        loaded_tfidf_model = sklearn2json.from_json("tfidf_vectorizer.json")
        self.assertEqual(19, loaded_tfidf_model.vocabulary["catdoll"])
        self.assertFalse(loaded_tfidf_model.use_idf)
        self.assertEqual(100, loaded_tfidf_model.max_features)
        remove("tfidf_vectorizer.json")

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
