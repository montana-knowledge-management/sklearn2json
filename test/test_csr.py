import unittest

import unittest
from sklearn.preprocessing import LabelBinarizer
import sklearn2json
from os import remove

labels = [1, 2, 6, 4, 2, 3, 6, 4, 2, 3, 5]


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
    def test_base(self, model=LabelBinarizer(), labels=labels, exclude_keys=[]):
        model.fit(labels)
        model, test_model = helper_test(model)
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys() - exclude_keys), set(test_model.__dict__.keys()))
        remove("model.json")
        return model, test_model


