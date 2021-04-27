import unittest
import sklearn2json
from os import remove
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR

input_x = [[0, 2, 1],
           [3, 3, 3],
           [1, 1, 1],
           [2, 2, 2],
           [1, 3, 2],
           [1, 5, 4]]
output_y = [1, 3, 1.5, 3, 3.6, 4.1]
new_x = [[0, 1, 2]]


def helper_test(model):
    # save data
    sklearn2json.to_json(model, "model.json")

    # load the saved matrix from the saved object
    test_model = sklearn2json.from_json("model.json")
    print(
        "Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
    for key, value in model.__dict__.items():
        if not isinstance(value, type(test_model.__dict__.get(key))):
            print(
                "Key name: {}\n"
                "Original value:{}; Saved reloaded value: {}\n"
                "Original type:{}; Saved reloaded type: {}".format(key, value, test_model.__dict__.get(key),
                                                                     type(value), type(test_model.__dict__.get(key))))
    return model, test_model


class RegressionTestCase(unittest.TestCase):
    def test_base(self, model=DecisionTreeRegressor(), X=input_x, y=output_y, new_x=new_x):
        model.fit(X, y)
        model, test_model = helper_test(model)
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()), set(test_model.__dict__.keys()))
        self.assertEqual(model.predict(new_x), test_model.predict(new_x))
        remove("model.json")
        return model, test_model

    def test_to_json_from_json_decision_tree_regressor(self):
        model = DecisionTreeRegressor()
        self.test_base(model)

    def test_to_json_from_json_linear_regression(self):
        model = LinearRegression()
        self.test_base(model)

    def test_to_json_from_json_lasso_regression(self):
        model = Lasso()
        self.test_base(model)

    def test_to_json_from_json_elastic_net(self):
        model = ElasticNet()
        self.test_base(model)

    def test_to_json_from_json_ridge(self):
        model = Ridge()
        self.test_base(model)

    def test_to_json_from_json_svr(self):
        model = SVR()
        self.test_base(model)
