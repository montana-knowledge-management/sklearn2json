import unittest
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from os import remove
from test.helper_test import print_differences

input_x = [[0, 2, 1],
           [3, 3, 3],
           [1, 1, 1],
           [2, 2, 2],
           [1, 3, 2],
           [1, 5, 4]]

new_x = [[0, 1, 2]]


class ClusteringTestCase(unittest.TestCase):
    def test_base(self, model=KMeans(n_clusters=2), X=input_x, exclude_keys=[]):
        model.fit(X)
        model, test_model = print_differences(model)
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys() - exclude_keys), set(test_model.__dict__.keys()))
        remove("model.json")
        return model, test_model

    def test_to_json_from_json_kmeans(self):
        model = KMeans(n_clusters=2)
        model, test_model = self.test_base(model)
        self.assertEqual(model.predict(new_x), test_model.predict(new_x))

    def test_to_json_from_json_dbscan(self):
        model = DBSCAN()
        model, test_model = self.test_base(model)
        self.assertListEqual(model.fit_predict(new_x).tolist(), test_model.fit_predict(new_x).tolist())
