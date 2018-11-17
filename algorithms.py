import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def print_conf_matrix(algo_name, conf_mat):
	print(30*'-')
	print(f'Confusion Matrix for {algo_name}')
	print(conf_mat)

def run_svm(train_X, train_y, test_X, test_y, kernel = 'rbf', gamma='auto', random_state = None):
	
	clf = SVC(kernel = kernel, gamma = gamma, random_state = random_state)
	clf.fit(train_X, train_y)

	pred_y = clf.predict(test_X)
	print_conf_matrix('SVM', pd.crosstab(test_y, pred_y, margins = True))
	return

def run_nb(train_X, train_y, test_X, test_y):

	clf = BernoulliNB(alpha = 1.0, fit_prior = True)
	clf.fit(train_X, train_y)

	pred_y = clf.predict(test_X)
	print_conf_matrix('Naive Bayes', pd.crosstab(test_y, pred_y, margins = True))
	return

def run_lr(train_X, train_y, test_X, test_y, solver = 'liblinear', random_state = None):

	clf = LogisticRegression(C = 1.0, solver = solver, random_state = random_state)
	clf.fit(train_X, train_y)

	pred_y = clf.predict(test_X)
	print_conf_matrix('Logistic Regression', pd.crosstab(test_y, pred_y, margins = True))
	return