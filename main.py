import numpy as np
import pandas as pd

import algorithms

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#from imblearn.over_sampling import SMOTE
#from sklearn.model_selection import KFold

#from sklearn.naive_bayes import BernouilliNB


# set the random seed
np.random.seed(566)

## Prepare the data
## Prepare the data
# import data
all_data = pd.read_csv("all-data.csv")
all_data['Output'] = all_data.apply(lambda r: 1 if r['Outcome'] == 'Active' else 0, axis = 1)
all_data.drop('Outcome', inplace = True, axis = 1)



# split the data into test and train sets
test_size = 0.1 # percent of the dataset that should be set aside for test
n_splits = 10 # number of splits for cross-validation
X_data, y_data = all_data.loc[:, all_data.columns != 'Output'], all_data['Output']

train_X, test_X, train_y, test_y = train_test_split(X_data, y_data, test_size = test_size, shuffle = True)
#print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)

p = all_data.shape[1] - 1 # number of features

majority_data = all_data.loc[all_data["Output"] == 0]
n_minority = all_data.loc[all_data["Output"] == 1].shape[0] # number of minority samples
print("Total samples = " + str(all_data.shape[0]))
print("There are " + str(n_minority) + " minority samples.")


# cross-validation for the regular train/test split
kf = StratifiedKFold(n_splits = n_splits)
kf.get_n_splits(train_X)

# try oversampling the minority class
# NOTE: Should only oversample on the training data to ensure independence between training and test sets
# Use SMOTE to ensure that we are not overfitting?
# Test set might have radically different distribution than train set
#sm = SMOTE(ratio = 1.0)
#oversampled_data =
#oversampled_train_X, oversampled

# try undersampling the majority class
# get as many majority samples as n_minority
# both train and test sets should be relatively balanced
"""undersampled_data = majority_data.sample(n = n_minority)
undersampled_train_X, undersampled_train_y, undersampled_test_X, undersampled_test_y = train_test_split(undersampled_data[:, 1:p], undersampled[:, -1], test_size = test_size, shuffle = True)
undersampled_kf = KFold(n_splits = n_splits)
undersampled_kf.get_splits(undersampled_train_X)"""



# Run the experiments
# cross-validation experiments

# cross-validation parameters
regwgts = np.array([np.Inf] + [1 / i for i in np.arange(0.2, 1.02, 0.2)])#np.arange(1, 11, 2)
kernels = ["linear", "rbf"]
gammas = np.arange(0.1, 1.1, 0.1)
lr_params = {"C":regwgts}
svm_params = {"gamma": gammas}
#for regwgt in regwgts:
    #lr_algs["Logistic Regression: regwgt = " + str(regwgt)] = LogisticRegression(penalty = 'l1', C = 1.0/regwgt, solver = "liblinear", random_state = None, class_weight = "balanced")
#for gamma in gammas:
    #svm_algs["SVM: RBG gamma = " + str(gamma)] = SVC(kernel = "rbf", gamma = gamma, random_state = None, class_weight = "balanced")

# run cross-validation
#lr_best = algorithms.cross_validate(lr_algs, train_X, train_y, kf)
#svm_best = algorithms.cross_validate(svm_algs, train_X, train_y, kf)

#print(lr_best.items())
#print(svm_best.items())

print("Starting logistic regression CV")
lr = LogisticRegression(penalty = "l1", solver = "liblinear", random_state = None, class_weight = "balanced")
lr_cv = GridSearchCV(lr, lr_params, cv = 10)
lr_cv.fit(train_X, train_y)
print(lr_cv.get_params())

print("Starting SVM CV")
svm = SVC(kernel = "rbf", random_state = None, class_weight = "balanced")
svm_cv = GridSearchCV(svm, svm_params, cv = 10)
svm_cv.fit(train_X, train_y)
print(svm_cv.get_params())


# final experiments (e.g., to get standard error)
print("Starting final experiments")
algorithms.run_svm(train_X, train_y, test_X, test_y)

algorithms.run_nb(train_X, train_y, test_X, test_y)

algorithms.run_lr(train_X, train_y, test_X, test_y)
