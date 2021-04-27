import unittest
import sklearn2json
from os import remove
from sklearn.tree import DecisionTreeRegressor


class RegressionTestCase(unittest.TestCase):
    def test_to_json_from_json_kmeans(self):
        model = DecisionTreeRegressor()
        X = [[0, 2, 1],
             [3, 3, 3],
             [1, 1, 1],
             [2, 2, 2],
             [1, 3, 2],
             [1, 5, 4]]
        y = [1, 3, 1.5, 3, 3.6, 4.1]
        model.fit(X, y)
        new_x = [[0, 1, 2]]
        # save data
        sklearn2json.to_json(model, "model.json")

        # load the saved matrix from the saved object
        test_model = sklearn2json.from_json("model.json")
        print(
            "Missing keys from saved model: {}\n".format(
                set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
        for key, value in model.__dict__.items():
            if type(value) != type(test_model.__dict__.get(key)):
                print(key)
                print(value, test_model.__dict__.get(key))
                print(type(value), type(test_model.__dict__.get(key)))
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()), set(test_model.__dict__.keys()))
        self.assertListEqual(model.predict(new_x).tolist(), test_model.predict(new_x).tolist())
        remove("model.json")
