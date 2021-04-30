[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Sonarcloud Status](https://sonarcloud.io/api/project_badges/measure?project=com.lapots.breed.judge:judge-rule-engine&metric=alert_status)](https://sonarcloud.io/dashboard?id=com.lapots.breed.judge:judge-rule-engine)
## Sklearn2JSON

Python library for converting Scikit-Learn models to JSON.

## Why Sklearn2JSON?

Scikit-learn has an integrated function to export models. This functionality is dependent on the Pickle or Joblib (based
on Pickle) libraries. One main problem with this solution is that these solutions freezes the machine learning classes as it is. 
Therefore if you have a production code or a two years old model 

Serializing model files with Pickle provide a simple attack vector for malicious users-- they give an attacker the
ability to execute arbitrary code wherever the file is
deserialized. (![For an example]:https://www.smartfile.com/blog/python-pickle-security-problems-and-solutions/).

Sklearn2JSON is a safe and transparent solution for exporting scikit-learn model files.

### Safe

Export model files to 100% JSON which cannot execute code on deserialization.

### Transparent

Model files are serialized in JSON (i.e., not binary), so you have the ability to see exactly what's inside.

## Install

```
$ pip install sklearn2json
```
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

SkLearn2PMML is licensed under the terms and conditions of
the [GNU Affero General Public License, Version 3.0](https://www.gnu.org/licenses/agpl-3.0.html).

If you would like to use SkLearn2PMML in a proprietary software project, then it is possible to enter into a licensing
agreement which makes SkLearn2PMML available under the terms and conditions of
the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause) instead.

# Additional information #

The concept of the project is based on the sklearn-json project SkLearn2JSON is developed and maintained by MONTANA
Knowledge Management LTD, Hungary.

Interested in using [Docutent](https://github.com/docutent) software in your company? Please
contact [info@docutent.org](mailto:info@docutent.org)

## Supported scikit-learn Models

* **Classification**

  * sklearn.linear_model.LogisticRegression
  * sklearn.linear_model.Perceptron
  * sklearn.discriminant_analysis.LinearDiscriminantAnalysis
  * sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
  * sklearn.svm.SVC
  * sklearn.naive_bayes.GaussianNB
  * sklearn.naive_bayes.MultinomialNB
  * sklearn.naive_bayes.ComplementNB
  * sklearn.naive_bayes.BernoulliNB
  * sklearn.tree.DecisionTreeClassifier
  * sklearn.ensemble.RandomForestClassifier
  * sklearn.ensemble.GradientBoostingClassifier
  * sklearn.neural_network.MLPClassifier


* **Regression**

  * sklearn.linear_model.LinearRegression
  * sklearn.linear_model.Ridge
  * sklearn.linear_model.Lasso
  * sklearn.svm.SVR
  * sklearn.tree.DecisionTreeRegressor
  * sklearn.ensemble.RandomForestRegressor
  * sklearn.ensemble.GradientBoostingRegressor
  * sklearn.neural_network.MLPRegressor
