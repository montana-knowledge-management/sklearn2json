import unittest
import sklearn2json
from sklearn.decomposition import TruncatedSVD
from os import remove


class DimensionReductionTestCase(unittest.TestCase):
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
        for orig, saved in zip(model.transform(new_x).tolist()[0], test_model.transform(new_x).tolist()[0]):
            self.assertAlmostEqual(orig, saved)
        remove("model.json")
