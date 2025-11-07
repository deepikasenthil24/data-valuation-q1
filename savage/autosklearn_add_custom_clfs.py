from typing import Optional, List

FEAT_TYPE_TYPE = Optional[List[str]]

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import autosklearn.classification
import autosklearn.pipeline.components.classification
from autosklearn.pipeline.components.classification import (
    AutoSklearnClassificationAlgorithm,
)
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS, SPARSE


class CustomLogisticRegression(AutoSklearnClassificationAlgorithm):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        # Initialize the Logistic Regression with fixed hyperparameters
        self.estimator = LogisticRegression(
            C=1.0,  # Inverse of regularization strength; smaller values specify stronger regularization
            penalty='l2',  # Regularization type
            solver='lbfgs',  # Solver for optimization
            max_iter=100,  # Maximum number of iterations taken for the solvers to converge
            random_state=self.random_state
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError("Estimator not fitted.")
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError("Estimator not fitted.")
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "LR",
            "name": "Logistic Regression",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        # Return an empty configuration space since we are not tuning any hyperparameters
        return ConfigurationSpace()



class CustomRandomForest(AutoSklearnClassificationAlgorithm):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        self.estimator = RandomForestClassifier(
            n_estimators=100, max_features='sqrt', random_state=self.random_state)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if not self.estimator:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if not self.estimator:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "RF",
            "name": "Random Forest Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        return ConfigurationSpace()

class CustomDecisionTree(AutoSklearnClassificationAlgorithm):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        self.estimator = DecisionTreeClassifier(random_state=self.random_state)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if not self.estimator:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if not self.estimator:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "DT",
            "name": "Decision Tree Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        return ConfigurationSpace()

class CustomMLPClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        self.estimator = MLPClassifier(hidden_layer_sizes=(10,), random_state=self.random_state)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if not self.estimator:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if not self.estimator:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "MLP",
            "name": "Multilayer Perceptron Classifier",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": False,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        return ConfigurationSpace()

    
class CustomSVM(AutoSklearnClassificationAlgorithm):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
#         # Initialize the SGDClassifier with hinge loss to simulate SVM
#         self.estimator = SGDClassifier(
#             loss='hinge',  # Hinge loss is equivalent to a linear SVM
#             penalty='l2',  # L2 regularization is standard for SVMs
#             alpha=0.0001,  # Learning rate
#             max_iter=300,  # Maximum number of iterations
#             tol=1e-3,  # Stopping criterion
#             early_stopping=True,
#             random_state=self.random_state
#         )
        sgd_clf = SGDClassifier(loss='hinge', random_state=self.random_state, max_iter=100, tol=1e-3, early_stopping=True)
        clf = CalibratedClassifierCV(sgd_clf, method='sigmoid', cv=5)
        self.estimator = clf
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if not self.estimator:
            raise NotImplementedError("Estimator not fitted.")
        return self.estimator.predict(X)

    def predict_proba(self, X):
        raise NotImplementedError("SGDClassifier with hinge loss does not support probability estimates.")

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "SVM",
            "name": "Stochastic Gradient Descent SVM",
            "handles_regression": False,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": False,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        # Return an empty configuration space since we are not tuning any hyperparameters
        return ConfigurationSpace()
    
    
def add_clf(clf_name):
    classifiers = {
        'CustomLogisticRegression': CustomLogisticRegression,
        'CustomRandomForest': CustomRandomForest,
        'CustomDecisionTree': CustomDecisionTree,
        'CustomMLPClassifier': CustomMLPClassifier,
        'CustomSVM': CustomSVM
    }

    if clf_name in classifiers:
        autosklearn.pipeline.components.classification.add_classifier(classifiers[clf_name])
        print(f"{clf_name} added to auto-sklearn.")
    else:
        print(f"Classifier {clf_name} is not recognized.")