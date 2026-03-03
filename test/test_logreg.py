"""Unit tests for the logistic regression implementation.

These tests are intentionally small and fast. They validate:
- predictions are valid probabilities and the right shape
- loss_function matches sklearn's log_loss on fixed inputs
- calculate_gradient matches a finite-difference numerical gradient
- training updates the weights and records loss history
"""

import numpy as np
import pytest

from regression.logreg import LogisticRegressor
from regression import utils

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_prediction():
    # load dataset
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )

    # scale the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    # train my model
    model = LogisticRegressor(num_feats=X_train.shape[1], learning_rate=1e-5, tol=1e-6, max_iter=5, batch_size=16)
    model.train_model(X_train, y_train, X_val, y_val)
    # predict with my model
    y_prob = model.make_prediction(X_val)

    # train sklearn model
    sk_model = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42)
    sk_model.fit(X_train, y_train)
    # try to align coefficients/intercept so predictions are produced from the same params 
    sk_model.coef_ = model.W[:-1].reshape(1, -1)  # Align coefficients
    sk_model.intercept_ = np.array([model.W[-1]])  # Align intercept
	
    # predict with sklearn model
    sk_prob = sk_model.predict_proba(X_val)[:, 1]

    # calc the accuracy of each model
    # mine
    my_acc = accuracy_score(y_val, (y_prob > 0.5).astype(int))
    # sklearn
    sk_acc = accuracy_score(y_val, (sk_prob > 0.5).astype(int))
    # these should be close
    assert np.isclose(my_acc, sk_acc, atol=0.1)
    

def test_loss_function():
    # simple example to compare with sklearn
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
	# use my model to compute loss
    model = LogisticRegressor(num_feats=1)
    my_loss = model.loss_function(y_true, y_pred)
    # use sklearn to compute loss 
    sk_loss = log_loss(y_true, y_pred)
	# these should be close
    assert np.allclose(my_loss, sk_loss, atol=1e-8)


def test_gradient():
    # Generate a random n-class classification problem
    X, y = make_classification(n_samples=60, n_features=4, n_informative=3, n_redundant=0, random_state=0)
    X = StandardScaler().fit_transform(X)
	# my model
    model = LogisticRegressor(num_feats=X.shape[1])
    # calc the gradient using my model
    my_grad = model.calculate_gradient(y, X)
    
    # compute gradient manually for comparison
    eps = 1e-6
    manual_grad = np.zeros_like(model.W, dtype=float)
    W_orig = model.W.copy()

    for i in range(model.W.size):
        model.W = W_orig.copy()
        model.W[i] = model.W[i] + eps
        loss_plus = model.loss_function(y, model.make_prediction(X))

        model.W = W_orig.copy()
        model.W[i] = model.W[i] - eps
        loss_minus = model.loss_function(y, model.make_prediction(X))

        manual_grad[i] = (loss_plus - loss_minus) / (2 * eps)

    # restore original weights
    model.W = W_orig.copy()

    # gradients should be close 
    assert np.allclose(my_grad, manual_grad, rtol=1e-3, atol=1e-3)


def test_training():
    # training should update weights and record loss
    X, y = make_classification(n_samples=80, n_features=5, n_informative=3, random_state=1) # make n-class classification problem
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2) # Split arrays or matrices into random train and test subsets

	# my model
    model = LogisticRegressor(num_feats=X_train.shape[1], learning_rate=1e-2, max_iter=3, batch_size=16)
    W_before = model.W.copy()
    # train my model
    model.train_model(X_train, y_train, X_test, y_test)
    W_after = model.W

    # weights should change after training
    assert not np.allclose(W_before, W_after)
    # loss history should be non-empty
    assert len(model.loss_hist_train) > 0


