import unittest
from sklearn.decomposition import TruncatedSVD
from os import remove
from test.helper_test import print_differences

input_x = [
    [0, 2, 1, 2],
    [3, 3, 3, 2],
    [1, 1, 1, 2],
    [2, 2, 2, 1],
    [1, 3, 2, 1],
    [1, 5, 4, 5],
]
new_x = [[0, 1, 2, 3]]


class DimensionReductionTestCase(unittest.TestCase):
    def base(self, model, x_input, exclude_keys=None):
        if not exclude_keys:
            exclude_keys = []
        model.fit(x_input)
        model, test_model = print_differences(model)
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys() - exclude_keys), set(test_model.__dict__.keys()))
        remove("model.json")
        return model, test_model

    def test_to_json_from_json_lsa(self):
        model = TruncatedSVD(n_components=3)
        model, test_model = self.base(model, input_x)
        for orig, saved in zip(model.transform(new_x).tolist()[0], test_model.transform(new_x).tolist()[0]):
            self.assertAlmostEqual(orig, saved)
