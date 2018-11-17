import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Taken from Assignment 3
def getaccuracy(ytest, predictions):
	correct = 0
	for i in range(len(ytest)):
		if ytest[i] == predictions[i]:
		correct += 1
	return (correct/float(len(ytest))) * 100.0

#Taken from Assignment 3
def geterror(ytest, predictions):
	return (100.0-getaccuracy(ytest, predictions))

#Prints confusion matrix for binary classification algorithm
def print_conf_matrix(algo_name, conf_mat):
	print(30*'-')
	print(f'Confusion Matrix for {algo_name}')
	print(conf_mat)

#SVM, where the kernel is passed from main
#To Do: Cross validation to find best kernel
def run_svm(train_X, train_y, test_X, test_y, kernel = 'rbf', gamma='auto', random_state = None):

	clf = SVC(kernel = kernel, gamma = gamma, random_state = random_state)
	clf.fit(train_X, train_y)

	pred_y = clf.predict(test_X)
	print_conf_matrix('SVM', pd.crosstab(test_y, pred_y, margins = True))
	print('Error for SVM: ' + geterror(test_y, pred_y))
	return

#Naive Bayes prediction assuming features from bernoulli distribution
def run_nb(train_X, train_y, test_X, test_y):

	clf = BernoulliNB(alpha = 1.0, fit_prior = True)
	clf.fit(train_X, train_y)

	pred_y = clf.predict(test_X)
	print_conf_matrix('Naive Bayes', pd.crosstab(test_y, pred_y, margins = True))
	print('Error for Naive Bayes: ' + geterror(test_y, pred_y))
	return

#logistic regression with l1 regularization, where the regularization parameter is passed from main
#To Do: Cross validation to find best parameter
def run_lr(train_X, train_y, test_X, test_y, reg_param = 1.0, solver = 'liblinear', random_state = None):

	clf = LogisticRegression(penalty = 'l1', C = 1.0/reg_param, solver = solver, random_state = random_state)
	clf.fit(train_X, train_y)

	pred_y = clf.predict(test_X)
	print_conf_matrix('Logistic Regression', pd.crosstab(test_y, pred_y, margins = True))
	print('Error for Logistic Regression: ' + geterror(test_y, pred_y))
	return

def cross_validate_error(alg, X, y, kf):
    # returns cross-validation error
    # assumes we have been given an alg from scikit learn, so that we can call fit() and predict()
    total_error = 0
    for train_ix, test_ix in kf.split(X):
        alg.fit(X[train_ix], y[train_ix])
        preds = alg.predict(X[test_ix])
        total_error += sum([a == b for a, b in zip(y[test_ix], preds)])
    return total_error
