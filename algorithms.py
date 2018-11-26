import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.metrics import f1_score


#Prints confusion matrix for binary classification algorithm
def print_conf_matrix(algo_name, conf_mat):
	print(30*'-')
	print(f'Confusion Matrix for {algo_name}')
	print(conf_mat)

#SVM, where the kernel is passed from main
def run_svm(train_X, train_y, test_X, test_y, kernel = 'rbf', gamma='auto', random_state = None):

	clf = SVC(kernel = kernel, gamma = gamma, random_state = random_state, class_weight = "balanced")
	clf.fit(train_X, train_y)

	pred_y = clf.predict(test_X)
	print_conf_matrix('SVM', pd.crosstab(test_y, pred_y, margins = True))
	print('F1 score for SVM: ' + str(f1_score(test_y, pred_y)))
	return

#Naive Bayes prediction assuming features from bernoulli distribution
def run_nb(train_X, train_y, test_X, test_y):

	clf = BernoulliNB(alpha = 1.0, fit_prior = True)
	clf.fit(train_X, train_y)

	pred_y = clf.predict(test_X)
	print_conf_matrix('Naive Bayes', pd.crosstab(test_y, pred_y, margins = True)))
	print('F1 score for Naive Bayes: ' + str(f1_score(test_y, pred_y)))
	return

#logistic regression with l1 regularization, where the regularization parameter is passed from main
def run_lr(train_X, train_y, test_X, test_y, reg_param = 1.0, solver = 'liblinear', random_state = None):

	clf = LogisticRegression(penalty = 'l1', C = 1.0/reg_param, solver = solver, random_state = random_state, class_weight = "balanced")
	clf.fit(train_X, train_y)

	pred_y = clf.predict(test_X)
	print_conf_matrix('Logistic Regression', pd.crosstab(test_y, pred_y, margins = True))
	print('F1 score for Logistic Regression: ' + str(f1_score(test_y, pred_y)))
	return

"""def cross_validate(algs, X, y, kf):
	# assumes we have been given algs from scikit learn, so that we can call fit() and predict()
	# assumes algs is a dictionary
	# returns the best name, algorithm pair
	scores = {}
	ixs = kf.split(X, y)
	# convert to numpy representation for multidim indexing
	X = X.values
	y = y.values
	for name, alg in algs.items():
		y_pred = np.array([])
		for train_ix, test_ix in ixs:
			#print(X)
			alg.fit(X[train_ix, :], y[train_ix])
			np.concatenate(y_pred, alg.predict(X[test_ix, :]))
		scores[name] = f1_score(y_pred)
	best_score = np.Inf
	best_name = ""
	for name, score in scores.items():
		print(name + " got a score of: " + str(score))
		if best_score <= score:
			best_score = score
			best_name = name
	return {name: algs[name]}"""
