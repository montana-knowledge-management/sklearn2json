import unittest
from os import remove
from test.helper_test import print_differences

import numpy as np
from scipy.sparse import csr_matrix

row = np.array([0, 1, 2, 0])
col = np.array([0, 1, 1, 0])
data = np.array([1, 2, 4, 8])
matrix = csr_matrix((data, (row, col)), shape=(3, 3))


class CsrTestCase(unittest.TestCase):
    def test_csr_matrix(self, model=matrix):
        model, test_model = print_differences(model)
        self.assertListEqual(model.toarray().tolist(), test_model.toarray().tolist())
        remove("model.json")
        return model, test_model
