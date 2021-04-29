import unittest
from sklearn.preprocessing import LabelBinarizer
from os import remove
from test.helper_test import print_differences

labels = [1, 2, 6, 4, 2, 3, 6, 4, 2, 3, 5]


class LabelEncoderTestCase(unittest.TestCase):
    def test_base(self, model=LabelBinarizer(), labels=labels, exclude_keys=[]):
        model.fit(labels)
        model, test_model = print_differences(model)
        self.assertEqual(model.get_params(), test_model.get_params())
        self.assertEqual(set(model.__dict__.keys() - exclude_keys), set(test_model.__dict__.keys()))
        remove("model.json")
        return model, test_model

    def test_label_binarizer(self):
        model, test_model = self.test_base()
        self.assertListEqual(model.classes_.tolist(), test_model.classes_.tolist())
        self.assertListEqual(model.transform(model.classes_.tolist()).tolist(),
                             test_model.transform(model.classes_.tolist()).tolist())
