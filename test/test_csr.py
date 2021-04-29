import unittest
import numpy as np
import sklearn2json
from scipy.sparse import csr_matrix
from os import remove

row = np.array([0, 1, 2, 0])
col = np.array([0, 1, 1, 0])
data = np.array([1, 2, 4, 8])
matrix = csr_matrix((data, (row, col)), shape=(3, 3))


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


class CsrTestCase(unittest.TestCase):
    def test_csr_matrix(self, matrix=matrix, exclude_keys=[]):
        model, test_model = helper_test(matrix)
        self.assertListEqual(model.toarray().tolist(), test_model.toarray().tolist())
        remove("model.json")
        return model, test_model
