import numpy as np
from sklearn.preprocessing import LabelBinarizer


def serialize_label_binarizer(label_binarizer):
    serialized_label_binarizer = {
        "meta": "label-binarizer",
        "neg_label": label_binarizer.neg_label,
        "pos_label": label_binarizer.pos_label,
        "sparse_output": label_binarizer.sparse_output,
        "y_type_": label_binarizer.y_type_,
        "sparse_input_": label_binarizer.sparse_input_,
        "classes_": label_binarizer.classes_.tolist(),
    }

    return serialized_label_binarizer


def deserialize_label_binarizer(label_binarizer_dict):
    label_binarizer = LabelBinarizer()
    label_binarizer.neg_label = label_binarizer_dict["neg_label"]
    label_binarizer.pos_label = label_binarizer_dict["pos_label"]
    label_binarizer.sparse_output = label_binarizer_dict["sparse_output"]
    label_binarizer.y_type_ = label_binarizer_dict["y_type_"]
    label_binarizer.sparse_input_ = label_binarizer_dict["sparse_input_"]
    label_binarizer.classes_ = np.array(label_binarizer_dict["classes_"])

    return label_binarizer
