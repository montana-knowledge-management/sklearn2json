[![tests](https://github.com/montana-knowledge-management/sklearn2json/actions/workflows/ci.yml/badge.svg)](https://github.com/robust/actions)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=montana-knowledge-management_sklearn2json&metric=coverage)](https://sonarcloud.io/summary/new_code?id=montana-knowledge-management_sklearn2json)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=montana-knowledge-management_sklearn2json&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=montana-knowledge-management_sklearn2json)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=montana-knowledge-management_sklearn2json&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=montana-knowledge-management_sklearn2json)

## Sklearn2JSON

Python library for exporting Scikit-Learn models to JSON.

## Install

python3.8+:

```shell
poetry add sklearn2json@2023.3
```

python3.7 - 3.8:

```shell
poetry add sklearn2json@2021.5
```

## Why to use Sklearn2JSON?

Scikit-learn has an integrated function to export models. This functionality is dependent on the Pickle or Joblib (based
on Pickle) libraries. Pickle module can serialize almost any Python object, without any boilerplate or even
white-/black-listing (in the typical case). However, it is unsafe because it constructs arbitrary Python objects by
invoking random functions. Therefore, if one inner variable is refactored in your deployed machine learning model, it
cannot be loaded with the newest version of `scikit-learn`. This can cause a maintenance overhead and headaches for
deployed machine learning-based projects.

Moreover, serializing model files with Pickle provides a simple attack vector for malicious users-- they give an
attacker the ability to execute arbitrary code wherever the file is deserialized ([example][example]).

Sklearn2JSON is a safe and transparent solution for exporting scikit-learn model files.

#### Safe

Export model files to 100% JSON which cannot execute code on deserialization.

#### Transparent

Model files are serialized in JSON (i.e., not binary), so you have the ability to see exactly what's inside.

#### Easy care

The library saves the mathematical parameters (inner variables) of the trained machine learning model, which can be used
in the future with newer version of scikit-learn even when the class parameters have changed a bit.

## Example Usage

```
    import sklearn2json as sk
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0).fit(X, y)

    sk.to_json(model, file_name)
    deserialized_model = sk.from_json(file_name)

    deserialized_model.predict(X)
```

## Features

The list of supported models is growing. If you have a request for a model or feature, please reach out to
sklearn2json@docutent.org.

Sklearn2JSON is licensed under the terms and conditions of
the [GNU Affero General Public License, Version 3.0](https://www.gnu.org/licenses/agpl-3.0.html).

If you would like to use Sklearn2JSON in a proprietary software project, then it is possible to enter into a licensing
agreement which makes Sklearn2JSON available under the terms and conditions of
the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause) instead.

## Contributing

For guidance on setting up a development environment and how to make a contribution to Sklearn2JSON, see
the [contributing guidelines][contributing].

# Additional information #

The concept of the project is based on the sklearn-json project. SkLearn2JSON is developed, supported and maintained by
MONTANA Knowledge Management LTD, Hungary. In order to grow the community of contributors and users, and allow the
maintainers to devote more time to the projects, [please donate today] [donate].

Interested in using [Docutent](https://github.com/docutent) software in your company? Please
contact [info@docutent.org](mailto:info@docutent.org)

## Supported scikit-learn Models

* **Classification**

    * sklearn.linear_model.LogisticRegression
    * sklearn.linear_model.Perceptron
    * sklearn.discriminant_analysis.LinearDiscriminantAnalysis
    * sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
    * sklearn.svm.SVC
    * sklearn.svm.LinearSVC
    * sklearn.naive_bayes.GaussianNB
    * sklearn.naive_bayes.MultinomialNB
    * sklearn.naive_bayes.ComplementNB
    * sklearn.naive_bayes.BernoulliNB
    * sklearn.tree.DecisionTreeClassifier
    * sklearn.ensemble.RandomForestClassifier
    * sklearn.ensemble.GradientBoostingClassifier
    * sklearn.neural_network.MLPClassifier


* **Regression**

    * sklearn.linear_model.ElasticNet
    * sklearn.linear_model.LinearRegression
    * sklearn.linear_model.Ridge
    * sklearn.linear_model.Lasso
    * sklearn.svm.SVR
    * sklearn.tree.DecisionTreeRegressor
    * sklearn.ensemble.RandomForestRegressor
    * sklearn.ensemble.GradientBoostingRegressor
    * sklearn.neural_network.MLPRegressor


* **Clustering**

    * sklearn.cluster.KMeans
    * sklearn.cluster.DBSCAN


* **Other**

    * sklearn.feature_extraction.text.TfidfVectorizer
    * sklearn.decomposition.TruncatedSVD
    * sklearn.preprocessing.LabelBinarizer
    * scipy.sparse.csr_matrix

[example]: https://www.smartfile.com/blog/python-pickle-security-problems-and-solutions/

[contributing]: CONTRIBUTING.md

[donate]: https://donate.org
