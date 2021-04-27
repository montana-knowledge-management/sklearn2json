import unittest
import sklearn2json
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from os import remove


class ClusteringTestCase(unittest.TestCase):
    def test_to_json_from_json_kmeans(self):
        model = KMeans(n_clusters=2)
        X = [[0, 2, 1],
             [3, 3, 3],
             [1, 1, 1],
             [2, 2, 2],
             [1, 3, 2],
             [1, 5, 4]]

        model.fit(X)
        new_x = [[0, 1, 2]]
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
        self.assertListEqual(model.predict(new_x).tolist(), test_model.predict(new_x).tolist())
        remove("model.json")

    def test_to_json_from_json_dbscan(self):
        model = DBSCAN()
        X = [[0, 2, 1],
             [3, 3, 3],
             [1, 1, 1],
             [2, 2, 2],
             [1, 3, 2],
             [1, 5, 4]]
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
