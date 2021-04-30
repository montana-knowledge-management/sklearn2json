import sklearn2json


def print_differences(model):
    """
    A helper function to print the differences between saved and original sklearn models.
    :param model: an sklearn model e.g. SVC, LinearSVC
    :return: model: original model,
             test_model: saved and loaded sklearn model
    """
    # save data
    sklearn2json.to_json(model, "model.json")

    # load the saved matrix from the saved object
    test_model = sklearn2json.from_json("model.json")
    print("Missing keys from saved model: {}\n".format(set(model.__dict__.keys()) - set(test_model.__dict__.keys())))
    for key, value in model.__dict__.items():
        if not isinstance(value, type(test_model.__dict__.get(key))):
            print(
                "Key name: {}\n"
                "Original value:{}; Saved reloaded value: {}\n"
                "Original type:{}; Saved reloaded type: {}".format(
                    key,
                    value,
                    test_model.__dict__.get(key),
                    type(value),
                    type(test_model.__dict__.get(key)),
                )
            )
    return model, test_model
