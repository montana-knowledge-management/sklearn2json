import unittest
import sklearn2json
from os import remove
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from test.helper_test import print_differences

input_x = [[0, 2, 1],
           [3, 3, 3],
           [1, 1, 1],
           [2, 2, 2],
           [1, 3, 2],
           [1, 5, 4]]
output_y = [1, 3, 1.5, 3, 3.6, 4.1]
new_x = [[0, 1, 2]]
additional_input_x = [[10, 20, 5]]
additional_output_y = [9]


class RegressionTestCase(unittest.TestCase):
    def test_base(self, model=DecisionTreeRegressor(), X=input_x, y=output_y, new_x=new_x, exclude_keys=[]):
        model.fit(X, y)
        model, test_model = print_differences(model)
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys() - exclude_keys), set(test_model.__dict__.keys()))
        self.assertEqual(model.predict(new_x), test_model.predict(new_x))
        remove("model.json")
        return model, test_model

    def test_with_training(self, model=DecisionTreeRegressor(), X=input_x, y=output_y, new_x=new_x):
        model.fit(X, y)
        model, test_model = print_differences(model)
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys()), set(test_model.__dict__.keys()))
        self.assertEqual(model.predict(new_x), test_model.predict(new_x))

        # Training
        model.fit(additional_input_x, additional_output_y)
        test_model.fit(additional_input_x, additional_output_y)
        model, test_model = print_differences(model)
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

    def test_to_json_from_json_random_forest_regressor(self):
        model = RandomForestRegressor()
        self.test_base(model)

    def test_to_json_from_json_mlp_regressor(self):
        model = MLPRegressor(random_state=1)
        self.test_base(model, exclude_keys=['_random_state', '_optimizer'])

    def test_to_json_from_json_svr(self):
        model = SVR()
        self.test_base(model)

    def test_to_json_from_json_gradient_boosting_regressor(self):
        model = GradientBoostingRegressor()
        self.test_base(model, exclude_keys=["_rng"])

    # def test_serilaize_tree(self):
