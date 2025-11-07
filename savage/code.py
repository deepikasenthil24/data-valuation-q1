import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

def pipeline(X_train, y_train, X_test):
    """
    Imputes missing values, standardizes the data, and trains a Logistic Regression model.
    """
    imputer = IterativeImputer(random_state=42)
    model = LogisticRegression(random_state=42, max_iter=1000)

    X_train_imputed = imputer.fit_transform(X_train)
    
    ss = StandardScaler()
    ss.fit(X_train_imputed)
    
    model.fit(ss.transform(X_train_imputed), y_train)
    
    return model.predict_proba(ss.transform(X_test))

def auc(X_test, y_test, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).
    """
    return roc_auc_score(y_test, y_pred[:, 1])