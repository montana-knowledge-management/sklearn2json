import json

from sklearn2json.model_selector import deserialize_model
from sklearn2json.model_selector import serialize_model


def from_json(file_name):
    with open(file_name) as model_json:
        model_dict = json.load(model_json)
        return deserialize_model(model_dict)


def to_json(model, model_name):
    with open(model_name, "w") as model_json:
        json.dump(serialize_model(model), model_json)
