import unittest
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn2json
from os import remove

class VectorizerTestCase(unittest.TestCase):
    def test_tfidf_default_settings(self):
        """
        Test saving tfidf model weights in case of default parameters.
        """
        tfidf_vectorizer = TfidfVectorizer(decode_error="ignore", encoding="ascii", strip_accents="ascii",
                                           lowercase=False, analyzer="char_wb", stop_words=["a", "stb"],
                                           token_pattern=r"(?u)\b[a-z]+?\b", ngram_range=(1, 3), max_df=0.99,
                                           min_df=0.01, binary=True, norm="l1",

                                           smooth_idf=False,
                                           sublinear_tf=True

                                           )
        test_data = ["The quick brown fox jumps over the lazy dog.",
                     "When a dog barks it won't bite you.",
                     "This one is a dog, the other is a puppy."]
        tfidf_vectorizer.fit_transform(test_data)
        sklearn2json.to_json(tfidf_vectorizer, "tfidf_vectorizer.json")
        loaded_tfidf_model = sklearn2json.from_json("tfidf_vectorizer.json")
        self.assertDictEqual(tfidf_vectorizer.vocabulary_, loaded_tfidf_model.vocabulary_)
        self.assertEqual(set(tfidf_vectorizer.idf_), set(loaded_tfidf_model.idf_))
        self.assertEqual(tfidf_vectorizer.stop_words_, loaded_tfidf_model.stop_words_)
        self.assertEqual(tfidf_vectorizer.fixed_vocabulary_, loaded_tfidf_model.fixed_vocabulary_)
        self.assertEqual("ascii", loaded_tfidf_model.encoding)
        self.assertEqual("ignore", loaded_tfidf_model.decode_error)
        self.assertEqual("ascii", loaded_tfidf_model.strip_accents)
        self.assertFalse(loaded_tfidf_model.lowercase)
        self.assertEqual(tfidf_vectorizer.analyzer, loaded_tfidf_model.analyzer)
        self.assertEqual(tfidf_vectorizer.stop_words, loaded_tfidf_model.stop_words)
        self.assertEqual(r"(?u)\b[a-z]+?\b", loaded_tfidf_model.token_pattern)
        self.assertEqual((1, 3), loaded_tfidf_model.ngram_range)
        self.assertEqual(0.99, loaded_tfidf_model.max_df)
        self.assertEqual(0.01, loaded_tfidf_model.min_df)
        self.assertTrue(loaded_tfidf_model.binary)
        self.assertEqual("l1", loaded_tfidf_model.norm)

        self.assertFalse(loaded_tfidf_model.smooth_idf)
        self.assertTrue(loaded_tfidf_model.sublinear_tf)

        remove("tfidf_vectorizer.json")

    def test_tfidf_parameters_2(self):
        """
        Test saving tfidf model weights in case of default parameters.
        """
        tfidf_vectorizer = TfidfVectorizer(
            vocabulary={'the': 14, 'quick': 13, 'brown': 2, 'fox': 4, 'jumps': 7, 'over': 11, 'lazy': 8, 'dog': 3,
                        'when': 16, 'barks': 0, 'it': 6, 'won': 17, 'bite': 1, 'you': 18, 'this': 15, 'one': 9, 'is': 5,
                        'other': 10, 'puppy': 12, "catdoll": 19},
            use_idf=False,
            max_features=100
        )
        test_data = ["The quick brown fox jumps over the lazy dog.",
                     "When a dog barks it won't bite you.",
                     "This one is a dog, the other is a puppy and catdoll."]
        tfidf_vectorizer.fit_transform(test_data)
        sklearn2json.to_json(tfidf_vectorizer, "tfidf_vectorizer.json")
        loaded_tfidf_model = sklearn2json.from_json("tfidf_vectorizer.json")
        self.assertEqual(19, loaded_tfidf_model.vocabulary["catdoll"])
        self.assertFalse(loaded_tfidf_model.use_idf)
        self.assertEqual(100, loaded_tfidf_model.max_features)
        remove("tfidf_vectorizer.json")


if __name__ == '__main__':
    unittest.main()
