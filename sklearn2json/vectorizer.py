import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def serialize_tfidf(model):
    """
    Serializes a TfidfVectorizer model to dictionary.
    WARNING: The following parameter names are not possible to be serialized: preporcessor, tokenizer
    :param model: TfidfVectorizer model from sklearn.
    :return: dictionary containing the TfidfVectorizer model parameters and attributes
    """
    serialized_model = {
        "meta": "tfidf",
        "vocabulary_": model.vocabulary_,
        "fixed_vocabulary_": model.fixed_vocabulary_,
        "encoding": model.encoding,
        "decode_error": model.decode_error,
        "strip_accents": model.strip_accents,
        "lowercase": model.lowercase,
        "tokenizer": model.tokenizer,
        "analyzer": model.analyzer,
        "stop_words": model.stop_words,
        "token_pattern": model.token_pattern,
        "ngram_range": list(model.ngram_range),
        "max_df": model.max_df,
        "min_df": model.min_df,
        "max_features": model.max_features,
        "vocabulary": model.vocabulary,
        "binary": model.binary,
        "norm": model.norm,
        "use_idf": model.use_idf,
        "smooth_idf": model.smooth_idf,
        "sublinear_tf": model.sublinear_tf,
    }
    if model.use_idf:
        serialized_model.update({"idf_": model.idf_.tolist()})
    if not model.vocabulary:
        serialized_model.update({"stop_words_": list(model.stop_words_)})
    return serialized_model


def deserialize_tfidf(tfidf_model):
    model = TfidfVectorizer()

    # Loading attributes
    if tfidf_model["use_idf"]:
        model.idf_ = np.array(tfidf_model["idf_"])  # .astype(float64)
    model.vocabulary_ = tfidf_model["vocabulary_"]
    model.fixed_vocabulary_ = tfidf_model["fixed_vocabulary_"]
    if not tfidf_model["vocabulary"]:
        model.stop_words_ = set(tfidf_model["stop_words_"])
    # Loading parameters
    model.encoding = tfidf_model["encoding"]
    model.decode_error = tfidf_model["decode_error"]
    model.strip_accents = tfidf_model["strip_accents"]
    model.lowercase = tfidf_model["lowercase"]
    model.tokenizer = tfidf_model["tokenizer"]
    model.analyzer = tfidf_model["analyzer"]
    model.stop_words = tfidf_model["stop_words"]
    model.token_pattern = tfidf_model["token_pattern"]
    model.ngram_range = tuple(tfidf_model["ngram_range"])
    model.max_df = tfidf_model["max_df"]
    model.min_df = tfidf_model["min_df"]
    model.max_features = tfidf_model["max_features"]
    model.vocabulary = tfidf_model["vocabulary"]
    model.binary = tfidf_model["binary"]
    model.norm = tfidf_model["norm"]
    model.use_idf = tfidf_model["use_idf"]
    model.smooth_idf = tfidf_model["smooth_idf"]
    model.sublinear_tf = tfidf_model["sublinear_tf"]
    return model
